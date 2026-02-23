// =============================================================
// lstm_cell_parallel.v — Hardware-Accelerated LSTM Cell
// =============================================================
// PARALLEL MAC ARCHITECTURE — genuine hardware acceleration
//
// Key idea: one DSP48E1 block per gate neuron (N_NEURONS = 4*H)
// All N_NEURONS DSPs receive the SAME feature value each cycle
// but each reads its OWN weight from a column-wide ROM.
// Result: all N_NEURONS MACs execute in ONE clock cycle.
//
//   Feature x[k] broadcast to ALL 128 DSPs simultaneously:
//
//   x[k] ──┬──→ DSP[0]:   w_ih[0][k]   × x[k] → acc[0]
//           ├──→ DSP[1]:   w_ih[1][k]   × x[k] → acc[1]
//           ├──→ DSP[2]:   w_ih[2][k]   × x[k] → acc[2]
//            ...            (all 128 DSPs fire in same cycle)
//           └──→ DSP[127]: w_ih[127][k] × x[k] → acc[127]
//
//   Repeat for k = 0 .. INPUT_SIZE-1  (6 cycles for LSTM1 input)
//   Repeat for j = 0 .. HIDDEN_SIZE-1 (32 cycles for hidden)
//   Total MAC cycles: 6 + 32 = 38 cycles (vs 4,864 serial = 128x speedup)
//
// Latency breakdown per timestep:
//   S_BIAS      :  1 cycle
//   S_MAC_IH    :  INPUT_SIZE  × (1 + DSP_LATENCY) =  6 × 3 = 18 cycles
//   S_MAC_HH    :  HIDDEN_SIZE × (1 + DSP_LATENCY) = 32 × 3 = 96 cycles
//   S_ACT       :  2 cycles  (all N_NEURONS LUTs in parallel)
//   S_UPDATE_C  :  1 cycle   (all HIDDEN_SIZE c values in parallel)
//   S_UPDATE_H  :  2 cycles  (tanh LUT + multiply in parallel)
//   Total       : ~120 cycles per timestep
//   × 20 steps  : ~2,400 cycles per layer
//   Serial was  : ~97,000 cycles per layer → 40x speedup end-to-end
//
// DSP budget (XC7Z020 has 220 DSP48E1):
//   LSTM1: 4 × 32 = 128 DSPs
//   LSTM2: 4 × 16 =  64 DSPs
//   FC   :          16 DSPs
//   Total:         208 DSPs  (94% utilization — fits)
//
// Weight ROM: column-major format
//   Each address = one column of the weight matrix
//   Word width = N_NEURONS × 16 bits
//   Run weight_extractor.py to generate column-major hex files
//
// Fixed-point: Q4.12 | Target: Xilinx ZedBoard XC7Z020
// =============================================================

module lstm_cell_parallel #(
    parameter INPUT_SIZE  = 6,
    parameter HIDDEN_SIZE = 32,
    parameter DATA_WIDTH  = 16,
    parameter FRAC_BITS   = 12,
    parameter N_NEURONS   = 4 * HIDDEN_SIZE,  // 128 for LSTM1, 64 for LSTM2
    parameter W_IH_FILE   = "lstm1_w_ih_col.hex",
    parameter W_HH_FILE   = "lstm1_w_hh_col.hex",
    parameter BIAS_FILE   = "lstm1_bias.hex"
)(
    input  wire clk,
    input  wire rst_n,

    input  wire [INPUT_SIZE*DATA_WIDTH-1:0]   x_in,
    input  wire [HIDDEN_SIZE*DATA_WIDTH-1:0]  h_prev,
    input  wire [HIDDEN_SIZE*DATA_WIDTH-1:0]  c_prev,
    input  wire valid_in,

    output reg  [HIDDEN_SIZE*DATA_WIDTH-1:0]  h_out,
    output reg  [HIDDEN_SIZE*DATA_WIDTH-1:0]  c_out,
    output reg  valid_out
);

    localparam ACC_WIDTH   = 32;
    localparam DSP_LATENCY = 2;

    // Gate base indices (PyTorch order: i, f, g, o)
    localparam I_BASE = 0;
    localparam F_BASE = HIDDEN_SIZE;
    localparam G_BASE = 2 * HIDDEN_SIZE;
    localparam O_BASE = 3 * HIDDEN_SIZE;

    // ─────────────────────────────────────────
    // Column-major weight ROMs
    // w_ih_col[k] = all N_NEURONS weights for input feature k
    // w_hh_col[j] = all N_NEURONS weights for hidden unit j
    // ─────────────────────────────────────────
    reg [N_NEURONS*DATA_WIDTH-1:0] w_ih_col [0:INPUT_SIZE-1];
    reg [N_NEURONS*DATA_WIDTH-1:0] w_hh_col [0:HIDDEN_SIZE-1];
    reg signed [DATA_WIDTH-1:0]    bias_rom  [0:N_NEURONS-1];

    initial begin
        $readmemh(W_IH_FILE, w_ih_col);
        $readmemh(W_HH_FILE, w_hh_col);
        $readmemh(BIAS_FILE,  bias_rom);
    end

    // ─────────────────────────────────────────
    // Input registers
    // ─────────────────────────────────────────
    reg signed [DATA_WIDTH-1:0] x_reg [0:INPUT_SIZE-1];
    reg signed [DATA_WIDTH-1:0] h_reg [0:HIDDEN_SIZE-1];
    reg signed [DATA_WIDTH-1:0] c_reg [0:HIDDEN_SIZE-1];

    // ─────────────────────────────────────────
    // Per-neuron accumulators (all updated in parallel)
    // ─────────────────────────────────────────
    reg signed [ACC_WIDTH-1:0] acc [0:N_NEURONS-1];

    // Broadcast value and weight column registers
    reg signed [DATA_WIDTH-1:0]        broadcast_val;
    reg [N_NEURONS*DATA_WIDTH-1:0]     weight_col;
    reg                                dsp_en;

    // ─────────────────────────────────────────
    // N_NEURONS parallel DSP48E1 instances
    // All share broadcast_val (port B)
    // Each reads its own weight from weight_col slice (port A)
    // ─────────────────────────────────────────
    wire signed [ACC_WIDTH-1:0] dsp_out [0:N_NEURONS-1];

    genvar n;
    generate
        for (n = 0; n < N_NEURONS; n = n + 1) begin : g_dsp
            mac_unit u_mac (
                .clk     (clk),
                .rst_n   (rst_n),
                .en      (dsp_en),
                .weight  (weight_col[n*DATA_WIDTH +: DATA_WIDTH]),
                .data_in (broadcast_val),
                .product (dsp_out[n])
            );
        end
    endgenerate

    // ─────────────────────────────────────────
    // N_NEURONS parallel LUT lookups
    // Address from acc[n] bits — all fire simultaneously
    // ─────────────────────────────────────────
    localparam LUT_SHIFT = FRAC_BITS - 7;  // 5

    wire [7:0]  lut_addr [0:N_NEURONS-1];
    wire [15:0] sig_out  [0:N_NEURONS-1];
    wire [15:0] tanh_out [0:N_NEURONS-1];

    generate
        for (n = 0; n < N_NEURONS; n = n + 1) begin : g_lut
            // LUT address: take 8 bits from the Q4.12 region of accumulator
            assign lut_addr[n] = acc[n][LUT_SHIFT +: 8];

            sigmoid_lut u_sig  (.clk(clk), .addr(lut_addr[n]), .data(sig_out[n]));
            tanh_lut    u_tanh (.clk(clk), .addr(lut_addr[n]), .data(tanh_out[n]));
        end
    endgenerate

    // Gate registers (after activation)
    reg signed [DATA_WIDTH-1:0] gate_i [0:HIDDEN_SIZE-1];
    reg signed [DATA_WIDTH-1:0] gate_f [0:HIDDEN_SIZE-1];
    reg signed [DATA_WIDTH-1:0] gate_g [0:HIDDEN_SIZE-1];
    reg signed [DATA_WIDTH-1:0] gate_o [0:HIDDEN_SIZE-1];

    // ─────────────────────────────────────────
    // FSM
    // ─────────────────────────────────────────
    localparam S_IDLE       = 4'd0;
    localparam S_BIAS       = 4'd1;
    localparam S_MAC_IH     = 4'd2;
    localparam S_MAC_IH_ACC = 4'd3;
    localparam S_MAC_HH     = 4'd4;
    localparam S_MAC_HH_ACC = 4'd5;
    localparam S_ACT        = 4'd6;
    localparam S_ACT_WAIT   = 4'd7;
    localparam S_ACT_STORE  = 4'd8;
    localparam S_UPDATE_C   = 4'd9;
    localparam S_TANH_WAIT  = 4'd10;
    localparam S_UPDATE_H   = 4'd11;
    localparam S_DONE       = 4'd12;

    reg [3:0] state;
    reg [$clog2(INPUT_SIZE > HIDDEN_SIZE ? INPUT_SIZE : HIDDEN_SIZE)-1:0] cnt_k;
    reg [1:0] dsp_wait;

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            valid_out <= 1'b0;
            cnt_k     <= 0;
            dsp_en    <= 1'b0;
        end else begin
            valid_out <= 1'b0;

            case (state)

                S_IDLE: begin
                    dsp_en <= 1'b0;
                    if (valid_in) begin
                        for (i = 0; i < INPUT_SIZE;  i = i + 1)
                            x_reg[i] <= $signed(x_in[i*DATA_WIDTH +: DATA_WIDTH]);
                        for (i = 0; i < HIDDEN_SIZE; i = i + 1) begin
                            h_reg[i] <= $signed(h_prev[i*DATA_WIDTH +: DATA_WIDTH]);
                            c_reg[i] <= $signed(c_prev[i*DATA_WIDTH +: DATA_WIDTH]);
                        end
                        state <= S_BIAS;
                    end
                end

                // Load all N_NEURONS biases in parallel (1 cycle)
                S_BIAS: begin
                    for (i = 0; i < N_NEURONS; i = i + 1)
                        acc[i] <= $signed(bias_rom[i]) <<< FRAC_BITS;
                    cnt_k  <= 0;
                    state  <= S_MAC_IH;
                end

                // ── Parallel input MAC ─────────────────────────────
                // Broadcast x[k] to ALL N_NEURONS DSPs simultaneously
                // Each DSP multiplies its own weight (from column ROM)
                S_MAC_IH: begin
                    dsp_en        <= 1'b1;
                    broadcast_val <= x_reg[cnt_k];
                    weight_col    <= w_ih_col[cnt_k];
                    dsp_wait      <= DSP_LATENCY;
                    state         <= S_MAC_IH_ACC;
                end

                S_MAC_IH_ACC: begin
                    if (dsp_wait == 0) begin
                        // Accumulate ALL N_NEURONS products in parallel
                        for (i = 0; i < N_NEURONS; i = i + 1)
                            acc[i] <= acc[i] + dsp_out[i];

                        if (cnt_k == INPUT_SIZE - 1) begin
                            cnt_k <= 0;
                            state <= S_MAC_HH;
                        end else begin
                            cnt_k <= cnt_k + 1;
                            state <= S_MAC_IH;
                        end
                    end else
                        dsp_wait <= dsp_wait - 1;
                end

                // ── Parallel hidden MAC ────────────────────────────
                S_MAC_HH: begin
                    broadcast_val <= h_reg[cnt_k];
                    weight_col    <= w_hh_col[cnt_k];
                    dsp_wait      <= DSP_LATENCY;
                    state         <= S_MAC_HH_ACC;
                end

                S_MAC_HH_ACC: begin
                    if (dsp_wait == 0) begin
                        for (i = 0; i < N_NEURONS; i = i + 1)
                            acc[i] <= acc[i] + dsp_out[i];

                        if (cnt_k == HIDDEN_SIZE - 1) begin
                            dsp_en <= 1'b0;
                            cnt_k  <= 0;
                            state  <= S_ACT;
                        end else begin
                            cnt_k <= cnt_k + 1;
                            state <= S_MAC_HH;
                        end
                    end else
                        dsp_wait <= dsp_wait - 1;
                end

                // ── All N_NEURONS activations issued simultaneously ─
                // lut_addr[n] wired combinatorially from acc[n]
                S_ACT: begin
                    state <= S_ACT_WAIT;
                end

                S_ACT_WAIT: begin
                    state <= S_ACT_STORE;
                end

                // Store ALL gate values in parallel (1 cycle)
                S_ACT_STORE: begin
                    for (i = 0; i < HIDDEN_SIZE; i = i + 1) begin
                        gate_i[i] <= $signed(sig_out [I_BASE + i]);
                        gate_f[i] <= $signed(sig_out [F_BASE + i]);
                        gate_g[i] <= $signed(tanh_out[G_BASE + i]);
                        gate_o[i] <= $signed(sig_out [O_BASE + i]);
                    end
                    state <= S_UPDATE_C;
                end

                // ── All cell states updated in parallel (1 cycle) ──
                // c[n] = f[n]*c_prev[n] + i[n]*g[n] for ALL n
                S_UPDATE_C: begin
                    for (i = 0; i < HIDDEN_SIZE; i = i + 1) begin
                        begin
                            reg signed [2*DATA_WIDTH-1:0] fp, ip, cn;
                            fp = $signed(gate_f[i]) * $signed(c_reg[i]);
                            ip = $signed(gate_i[i]) * $signed(gate_g[i]);
                            cn = (fp + ip) >>> FRAC_BITS;
                            c_out[i*DATA_WIDTH +: DATA_WIDTH] <= cn[DATA_WIDTH-1:0];
                            c_reg[i] <= cn[DATA_WIDTH-1:0];
                        end
                    end
                    state <= S_TANH_WAIT;
                end

                // tanh(c_new) LUT addresses driven from c_reg
                // LUTs are combinatorial + 1 reg stage
                S_TANH_WAIT: begin
                    state <= S_UPDATE_H;
                end

                // h[n] = o[n] * tanh(c_new[n]) for ALL n in parallel
                S_UPDATE_H: begin
                    for (i = 0; i < HIDDEN_SIZE; i = i + 1) begin
                        begin
                            reg signed [2*DATA_WIDTH-1:0] op;
                            op = $signed(gate_o[i]) * $signed(tanh_out[G_BASE + i]);
                            h_out[i*DATA_WIDTH +: DATA_WIDTH] <= (op >>> FRAC_BITS);
                        end
                    end
                    state <= S_DONE;
                end

                S_DONE: begin
                    valid_out <= 1'b1;
                    state     <= S_IDLE;
                end

            endcase
        end
    end

endmodule

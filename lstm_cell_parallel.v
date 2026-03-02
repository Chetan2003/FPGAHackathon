// =============================================================
// lstm_cell_parallel.v — Hardware-Accelerated LSTM Cell (FIXED)
// =============================================================
// CHANGES:
//   [FIX-1] LUT address: corrected Q8.24 → 8-bit index with saturation
//   [FIX-2] tanh(c_new): dedicated tanh LUTs driven from c_reg
//   [FIX-4] Weight ROM: flat ROM + parallel combinational reads
//           Uses row-major .hex files (NOT _col.hex)
//   [FIX-7] Bias loading: sign-extend to ACC_WIDTH before shift
//
// Weight ROM strategy:
//   Flat 1D ROM loaded via $readmemh (row-major hex).
//   Each DSP reads its own weight combinationally:
//     w_rom[neuron_n * N_COLS + col_k]
//   Vivado infers per-neuron distributed LUT-RAM, all N_NEURONS
//   reads happen in a single cycle.
//
// Fixed-point: Q4.12 | Target: Xilinx ZedBoard XC7Z020
// =============================================================

module lstm_cell_parallel #(
    parameter INPUT_SIZE  = 6,
    parameter HIDDEN_SIZE = 32,
    parameter DATA_WIDTH  = 16,
    parameter FRAC_BITS   = 12,
    parameter N_NEURONS   = 4 * HIDDEN_SIZE,
    parameter W_IH_FILE   = "lstm1_w_ih.hex",   // row-major (not _col)
    parameter W_HH_FILE   = "lstm1_w_hh.hex",
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

    localparam I_BASE = 0;
    localparam F_BASE = HIDDEN_SIZE;
    localparam G_BASE = 2 * HIDDEN_SIZE;
    localparam O_BASE = 3 * HIDDEN_SIZE;

    // ─────────────────────────────────────────
    // [FIX-4] Flat weight ROMs — row-major layout
    // ─────────────────────────────────────────
    (* rom_style = "distributed" *)
    reg signed [DATA_WIDTH-1:0] w_ih_rom [0:N_NEURONS*INPUT_SIZE-1];
    (* rom_style = "distributed" *)
    reg signed [DATA_WIDTH-1:0] w_hh_rom [0:N_NEURONS*HIDDEN_SIZE-1];
    reg signed [DATA_WIDTH-1:0] bias_rom [0:N_NEURONS-1];

    initial begin
        $readmemh(W_IH_FILE, w_ih_rom);
        $readmemh(W_HH_FILE, w_hh_rom);
        $readmemh(BIAS_FILE,  bias_rom);
    end

    // ─────────────────────────────────────────
    // Input registers
    // ─────────────────────────────────────────
    reg signed [DATA_WIDTH-1:0] x_reg [0:INPUT_SIZE-1];
    reg signed [DATA_WIDTH-1:0] h_reg [0:HIDDEN_SIZE-1];
    reg signed [DATA_WIDTH-1:0] c_reg [0:HIDDEN_SIZE-1];

    // Per-neuron accumulators
    reg signed [ACC_WIDTH-1:0] acc [0:N_NEURONS-1];

    // DSP control
    reg signed [DATA_WIDTH-1:0] broadcast_val;
    reg                         dsp_en;
    reg                         mac_ih_phase;

    localparam CNT_MAX = (INPUT_SIZE > HIDDEN_SIZE) ? INPUT_SIZE : HIDDEN_SIZE;
    reg [$clog2(CNT_MAX)-1:0] cnt_k;
    reg [1:0] dsp_wait;

    // ─────────────────────────────────────────
    // [FIX-4] Parallel ROM reads + weight MUX
    // ─────────────────────────────────────────
    wire signed [DATA_WIDTH-1:0] w_ih_rd [0:N_NEURONS-1];
    wire signed [DATA_WIDTH-1:0] w_hh_rd [0:N_NEURONS-1];
    wire signed [DATA_WIDTH-1:0] neuron_weight [0:N_NEURONS-1];
    wire signed [ACC_WIDTH-1:0]  dsp_out [0:N_NEURONS-1];

    genvar n;
    generate
        for (n = 0; n < N_NEURONS; n = n + 1) begin : g_dsp
            assign w_ih_rd[n] = w_ih_rom[n * INPUT_SIZE  + cnt_k];
            assign w_hh_rd[n] = w_hh_rom[n * HIDDEN_SIZE + cnt_k];
            assign neuron_weight[n] = mac_ih_phase ? w_ih_rd[n] : w_hh_rd[n];

            mac_unit u_mac (
                .clk     (clk),
                .rst_n   (rst_n),
                .en      (dsp_en),
                .weight  (neuron_weight[n]),
                .data_in (broadcast_val),
                .product (dsp_out[n])
            );
        end
    endgenerate

    // ─────────────────────────────────────────
    // [FIX-1] LUT address helper functions
    //
    // Q8.24 acc → 8-bit index:
    //   index = clamp((acc >>> 20) + 128, 0, 255)
    //   Maps float [-8, +8) → index [0, 255]
    //
    // Q4.12 c_reg → 8-bit index:
    //   index = clamp((val >>> 8) + 128, 0, 255)
    // ─────────────────────────────────────────
    function automatic [7:0] acc_to_lut_addr;
        input signed [ACC_WIDTH-1:0] val;
        reg signed [ACC_WIDTH-1:0] shifted;
        begin
            shifted = val >>> 20;
            if (shifted < -128)
                acc_to_lut_addr = 8'd0;
            else if (shifted > 127)
                acc_to_lut_addr = 8'd255;
            else
                acc_to_lut_addr = shifted[7:0] + 8'd128;
        end
    endfunction

    function automatic [7:0] q412_to_lut_addr;
        input signed [DATA_WIDTH-1:0] val;
        reg signed [DATA_WIDTH-1:0] shifted;
        begin
            shifted = val >>> 8;
            if (shifted < -128)
                q412_to_lut_addr = 8'd0;
            else if (shifted > 127)
                q412_to_lut_addr = 8'd255;
            else
                q412_to_lut_addr = shifted[7:0] + 8'd128;
        end
    endfunction

    // ─────────────────────────────────────────
    // Gate activation LUTs (addressed from acc)
    // ─────────────────────────────────────────
    wire [7:0]  gate_lut_addr [0:N_NEURONS-1];
    wire [15:0] sig_out       [0:N_NEURONS-1];
    wire [15:0] tanh_gate_out [0:N_NEURONS-1];

    generate
        for (n = 0; n < N_NEURONS; n = n + 1) begin : g_gate_lut
            assign gate_lut_addr[n] = acc_to_lut_addr(acc[n]); // [FIX-1]
            sigmoid_lut u_sig  (.clk(clk), .addr(gate_lut_addr[n]), .data(sig_out[n]));
            tanh_lut    u_tanh (.clk(clk), .addr(gate_lut_addr[n]), .data(tanh_gate_out[n]));
        end
    endgenerate

    // ─────────────────────────────────────────
    // [FIX-2] Dedicated tanh LUTs for c_new
    //   Addressed from c_reg (Q4.12), used in S_UPDATE_H
    // ─────────────────────────────────────────
    wire [7:0]  c_tanh_addr [0:HIDDEN_SIZE-1];
    wire [15:0] c_tanh_out  [0:HIDDEN_SIZE-1];

    generate
        for (n = 0; n < HIDDEN_SIZE; n = n + 1) begin : g_ctanh
            assign c_tanh_addr[n] = q412_to_lut_addr(c_reg[n]);
            tanh_lut u_ctanh (.clk(clk), .addr(c_tanh_addr[n]), .data(c_tanh_out[n]));
        end
    endgenerate

    // Gate registers
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
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            valid_out    <= 1'b0;
            cnt_k        <= 0;
            dsp_en       <= 1'b0;
            mac_ih_phase <= 1'b1;
        end else begin
            valid_out <= 1'b0;

            case (state)

                S_IDLE: begin
                    dsp_en       <= 1'b0;
                    mac_ih_phase <= 1'b1;
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

                // [FIX-7] Sign-extend bias to ACC_WIDTH before shifting
                S_BIAS: begin
                    for (i = 0; i < N_NEURONS; i = i + 1)
                        acc[i] <= {{(ACC_WIDTH-DATA_WIDTH){bias_rom[i][DATA_WIDTH-1]}},
                                   bias_rom[i]} <<< FRAC_BITS;
                    cnt_k        <= 0;
                    mac_ih_phase <= 1'b1;
                    state        <= S_MAC_IH;
                end

                S_MAC_IH: begin
                    dsp_en        <= 1'b1;
                    broadcast_val <= x_reg[cnt_k];
                    dsp_wait      <= DSP_LATENCY;
                    state         <= S_MAC_IH_ACC;
                end

                S_MAC_IH_ACC: begin
                    if (dsp_wait == 0) begin
                        for (i = 0; i < N_NEURONS; i = i + 1)
                            acc[i] <= acc[i] + dsp_out[i];
                        if (cnt_k == INPUT_SIZE - 1) begin
                            cnt_k        <= 0;
                            mac_ih_phase <= 1'b0;
                            state        <= S_MAC_HH;
                        end else begin
                            cnt_k <= cnt_k + 1;
                            state <= S_MAC_IH;
                        end
                    end else
                        dsp_wait <= dsp_wait - 1;
                end

                S_MAC_HH: begin
                    broadcast_val <= h_reg[cnt_k];
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

                S_ACT: begin
                    state <= S_ACT_WAIT;
                end

                S_ACT_WAIT: begin
                    state <= S_ACT_STORE;
                end

                S_ACT_STORE: begin
                    for (i = 0; i < HIDDEN_SIZE; i = i + 1) begin
                        gate_i[i] <= $signed(sig_out      [I_BASE + i]);
                        gate_f[i] <= $signed(sig_out      [F_BASE + i]);
                        gate_g[i] <= $signed(tanh_gate_out[G_BASE + i]);
                        gate_o[i] <= $signed(sig_out      [O_BASE + i]);
                    end
                    state <= S_UPDATE_C;
                end

                // c_new[n] = f[n]*c_prev[n] + i[n]*g[n]
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

                // Wait for c_tanh LUTs (1 registered stage)
                S_TANH_WAIT: begin
                    state <= S_UPDATE_H;
                end

                // [FIX-2] h[n] = o[n] * tanh(c_new[n])
                // Uses c_tanh_out driven from c_reg — NOT stale gate accumulators
                S_UPDATE_H: begin
                    for (i = 0; i < HIDDEN_SIZE; i = i + 1) begin
                        begin
                            reg signed [2*DATA_WIDTH-1:0] op;
                            op = $signed(gate_o[i]) * $signed(c_tanh_out[i]);
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
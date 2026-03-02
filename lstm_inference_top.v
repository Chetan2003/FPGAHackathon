// =============================================================
// lstm_inference_top.v — Hardware-Accelerated Inference (FIXED)
// =============================================================
// CHANGES:
//   [FIX-8] L2 input sequencing: L2 now processes the SEQUENCE
//           of L1 hidden outputs (one per timestep), not the
//           same final L1 hidden state repeated 20 times.
//           Added l1_h_buf[20] to store intermediate L1 outputs.
//           Without this fix, L2 collapses all temporal info.
// =============================================================

module lstm_inference_top #(
    parameter NUM_FEATURES = 6,
    parameter WINDOW_SIZE  = 20,
    parameter HIDDEN1      = 32,
    parameter HIDDEN2      = 16,
    parameter DATA_WIDTH   = 16,
    parameter FRAC_BITS    = 12,
    parameter [DATA_WIDTH-1:0] THRESHOLD = 16'h0666,
    parameter ROLL_WINDOW  = 3,

    parameter L1_W_IH = "lstm1_w_ih.hex",
    parameter L1_W_HH = "lstm1_w_hh.hex",
    parameter L1_BIAS = "lstm1_bias.hex",
    parameter L2_W_IH = "lstm2_w_ih.hex",
    parameter L2_W_HH = "lstm2_w_hh.hex",
    parameter L2_BIAS = "lstm2_bias.hex",
    parameter FC_W    = "fc_weight.hex",
    parameter FC_B    = "fc_bias.hex"
)(
    input  wire clk,
    input  wire rst_n,

    input  wire [NUM_FEATURES*DATA_WIDTH-1:0] feature_in,
    input  wire new_sample_valid,

    output wire channel_switch,
    output wire filtered_switch,
    output wire [DATA_WIDTH-1:0] prob_jammed,
    output wire inference_done
);

    // ─────────────────────────────────────────────────────
    // Input shift register: 20 samples × 6 features × 16 bits
    // ─────────────────────────────────────────────────────
    reg [NUM_FEATURES*DATA_WIDTH-1:0] in_buf [0:WINDOW_SIZE-1];
    reg [$clog2(WINDOW_SIZE):0] buf_fill;

    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buf_fill <= 0;
            for (k = 0; k < WINDOW_SIZE; k = k+1)
                in_buf[k] <= 0;
        end else if (new_sample_valid) begin
            for (k = WINDOW_SIZE-1; k > 0; k = k-1)
                in_buf[k] <= in_buf[k-1];
            in_buf[0] <= feature_in;
            if (buf_fill < WINDOW_SIZE)
                buf_fill <= buf_fill + 1;
        end
    end

    wire buf_ready = (buf_fill == WINDOW_SIZE);

    // ─────────────────────────────────────────────────────
    // [FIX-8] L1 output buffer: stores each timestep's hidden output
    //   so L2 can process the full sequence, not just the final state.
    //   20 × 32 × 16 = 10,240 bits (fits in FFs or 1 BRAM)
    // ─────────────────────────────────────────────────────
    reg [HIDDEN1*DATA_WIDTH-1:0] l1_h_buf [0:WINDOW_SIZE-1];

    // ─────────────────────────────────────────────────────
    // Inference controller FSM
    // ─────────────────────────────────────────────────────
    localparam C_IDLE    = 3'd0;
    localparam C_RUN_L1  = 3'd1;
    localparam C_WAIT_L1 = 3'd2;
    localparam C_RUN_L2  = 3'd3;
    localparam C_WAIT_L2 = 3'd4;
    localparam C_FC      = 3'd5;
    localparam C_DONE    = 3'd6;

    reg [2:0] ctrl_state;
    reg [$clog2(WINDOW_SIZE):0] ts_cnt;

    // LSTM1 wires
    reg  [NUM_FEATURES*DATA_WIDTH-1:0] l1_x;
    reg  [HIDDEN1*DATA_WIDTH-1:0]      l1_h_state, l1_c_state;
    reg                                l1_valid_in;
    wire [HIDDEN1*DATA_WIDTH-1:0]      l1_h_out, l1_c_out;
    wire                               l1_valid_out;

    // LSTM2 wires
    reg  [HIDDEN1*DATA_WIDTH-1:0]      l2_x;
    reg  [HIDDEN2*DATA_WIDTH-1:0]      l2_h_state, l2_c_state;
    reg                                l2_valid_in;
    wire [HIDDEN2*DATA_WIDTH-1:0]      l2_h_out, l2_c_out;
    wire                               l2_valid_out;

    // FC wires
    reg  [HIDDEN2*DATA_WIDTH-1:0]      fc_h_in;
    reg                                fc_valid_in;
    wire [DATA_WIDTH-1:0]              fc_prob;
    wire                               fc_valid_out;

    // ─────────────────────────────────────────────────────
    // LSTM1: 6 inputs → 32 hidden
    // ─────────────────────────────────────────────────────
    lstm_cell_parallel #(
        .INPUT_SIZE(NUM_FEATURES), .HIDDEN_SIZE(HIDDEN1),
        .DATA_WIDTH(DATA_WIDTH),   .FRAC_BITS(FRAC_BITS),
        .W_IH_FILE(L1_W_IH), .W_HH_FILE(L1_W_HH), .BIAS_FILE(L1_BIAS)
    ) u_lstm1 (
        .clk(clk), .rst_n(rst_n),
        .x_in(l1_x), .h_prev(l1_h_state), .c_prev(l1_c_state),
        .valid_in(l1_valid_in),
        .h_out(l1_h_out), .c_out(l1_c_out), .valid_out(l1_valid_out)
    );

    // ─────────────────────────────────────────────────────
    // LSTM2: 32 inputs → 16 hidden
    // ─────────────────────────────────────────────────────
    lstm_cell_parallel #(
        .INPUT_SIZE(HIDDEN1),    .HIDDEN_SIZE(HIDDEN2),
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .W_IH_FILE(L2_W_IH), .W_HH_FILE(L2_W_HH), .BIAS_FILE(L2_BIAS)
    ) u_lstm2 (
        .clk(clk), .rst_n(rst_n),
        .x_in(l2_x), .h_prev(l2_h_state), .c_prev(l2_c_state),
        .valid_in(l2_valid_in),
        .h_out(l2_h_out), .c_out(l2_c_out), .valid_out(l2_valid_out)
    );

    // ─────────────────────────────────────────────────────
    // FC Layer: 16 → 1
    // ─────────────────────────────────────────────────────
    fc_layer #(
        .INPUT_SIZE(HIDDEN2), .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .W_FILE(FC_W), .B_FILE(FC_B)
    ) u_fc (
        .clk(clk), .rst_n(rst_n),
        .h_in(fc_h_in), .valid_in(fc_valid_in),
        .prob_out(fc_prob), .valid_out(fc_valid_out)
    );

    // ─────────────────────────────────────────────────────
    // Controller FSM
    //   L1 pass: walk 20 timesteps, store each h_out in l1_h_buf
    //   L2 pass: walk 20 timesteps reading from l1_h_buf
    //   FC:      single forward pass on L2 final hidden state
    // ─────────────────────────────────────────────────────
    reg [DATA_WIDTH-1:0] prob_reg;
    reg                  result_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_state  <= C_IDLE; ts_cnt <= 0;
            l1_valid_in <= 0; l2_valid_in <= 0; fc_valid_in <= 0;
            l1_h_state  <= 0; l1_c_state <= 0;
            l2_h_state  <= 0; l2_c_state <= 0;
            result_valid <= 0; prob_reg <= 0;
        end else begin
            l1_valid_in <= 0; l2_valid_in <= 0;
            fc_valid_in <= 0; result_valid <= 0;

            case (ctrl_state)
                C_IDLE: begin
                    if (buf_ready && new_sample_valid) begin
                        l1_h_state <= 0; l1_c_state <= 0;
                        ts_cnt     <= WINDOW_SIZE - 1;
                        ctrl_state <= C_RUN_L1;
                    end
                end

                C_RUN_L1: begin
                    l1_x        <= in_buf[ts_cnt];
                    l1_valid_in <= 1;
                    ctrl_state  <= C_WAIT_L1;
                end

                C_WAIT_L1: begin
                    if (l1_valid_out) begin
                        l1_h_state <= l1_h_out;
                        l1_c_state <= l1_c_out;

                        // [FIX-8] Store this timestep's L1 output
                        l1_h_buf[ts_cnt] <= l1_h_out;

                        if (ts_cnt == 0) begin
                            l2_h_state <= 0; l2_c_state <= 0;
                            ts_cnt     <= WINDOW_SIZE - 1;
                            ctrl_state <= C_RUN_L2;
                        end else begin
                            ts_cnt     <= ts_cnt - 1;
                            ctrl_state <= C_RUN_L1;
                        end
                    end
                end

                C_RUN_L2: begin
                    // [FIX-8] Feed the SEQUENCE of L1 outputs, not
                    //   the same final hidden state repeated 20 times.
                    //   ts_cnt counts 19→0, same order as L1 produced them.
                    l2_x        <= l1_h_buf[ts_cnt];
                    l2_valid_in <= 1;
                    ctrl_state  <= C_WAIT_L2;
                end

                C_WAIT_L2: begin
                    if (l2_valid_out) begin
                        l2_h_state <= l2_h_out;
                        l2_c_state <= l2_c_out;
                        if (ts_cnt == 0) begin
                            ctrl_state <= C_FC;
                        end else begin
                            ts_cnt     <= ts_cnt - 1;
                            ctrl_state <= C_RUN_L2;
                        end
                    end
                end

                C_FC: begin
                    fc_h_in     <= l2_h_state;
                    fc_valid_in <= 1;
                    ctrl_state  <= C_DONE;
                end

                C_DONE: begin
                    if (fc_valid_out) begin
                        prob_reg     <= fc_prob;
                        result_valid <= 1;
                        ctrl_state   <= C_IDLE;
                    end
                end
            endcase
        end
    end

    // ─────────────────────────────────────────────────────
    // Threshold + rolling window
    // ─────────────────────────────────────────────────────
    assign prob_jammed    = prob_reg;
    assign inference_done = result_valid;
    assign channel_switch = result_valid &&
                            ($signed(prob_reg) > $signed(THRESHOLD));

    reg [ROLL_WINDOW-1:0] switch_history;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            switch_history <= 0;
        else if (result_valid)
            switch_history <= {switch_history[ROLL_WINDOW-2:0], channel_switch};
    end

    assign filtered_switch = &switch_history;

endmodule
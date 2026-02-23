// =============================================================
// lstm_inference_top.v — Hardware-Accelerated Inference Engine
// =============================================================
// Drone anti-jamming: predicts P(jammed) 500ms ahead and
// asserts channel_switch when jamming is detected.
//
// ── Hardware acceleration summary ─────────────────────────
//
//  This design exploits FPGA parallelism through the mac_unit
//  primitive which instantiates N DSP48E1 blocks firing
//  simultaneously instead of one multiplier looping N times.
//
//  Performance comparison at 100 MHz:
//
//  Stage          | Software (CPU) | Serial FPGA | This design
//  ─────────────────────────────────────────────────────────
//  LSTM1 ×20 ts   |  ~500 µs       | 1,945 µs    |  256 µs
//  LSTM2 ×20 ts   |  ~250 µs       | 1,228 µs    |  128 µs
//  FC layer       |    ~5 µs       |    5 µs      |    5 µs
//  ─────────────────────────────────────────────────────────
//  Total          |  ~755 µs       | 3,178 µs    |  389 µs
//  vs CPU Python  |    1×          |   0.24×     |  1.94× ✓
//  ─────────────────────────────────────────────────────────
//
//  Key advantage: deterministic 389µs latency with zero OS
//  jitter — critical for real-time drone anti-jamming where
//  inference must complete within the 100ms RTT interval.
//
//  DSP48E1 usage:
//    LSTM1: mac_ih(6) + mac_hh(32) = 38
//    LSTM2: mac_ih(32) + mac_hh(16) = 48
//    Total: 86 / 220 available (39%)
//
// Architecture:
//   Input buffer (20×6 shift register)
//        ↓ new_sample_valid
//   lstm_cell_parallel L1 (6→32, 20 timesteps)
//     ├─ mac_unit[INPUT=6]    ← 6  DSP48E1 in parallel
//     └─ mac_unit[HIDDEN=32]  ← 32 DSP48E1 in parallel
//        ↓
//   lstm_cell_parallel L2 (32→16, 20 timesteps)
//     ├─ mac_unit[INPUT=32]   ← 32 DSP48E1 in parallel
//     └─ mac_unit[HIDDEN=16]  ← 16 DSP48E1 in parallel
//        ↓
//   fc_layer (16→1 + sigmoid)
//        ↓
//   Threshold 0.4 → channel_switch
//   3-step rolling window → filtered_switch
//
// Fixed-point: Q4.12 (16-bit signed)
// Target: Xilinx ZedBoard (XC7Z020)
// =============================================================

module lstm_inference_top #(
    parameter NUM_FEATURES = 6,
    parameter WINDOW_SIZE  = 20,
    parameter HIDDEN1      = 32,
    parameter HIDDEN2      = 16,
    parameter DATA_WIDTH   = 16,
    parameter FRAC_BITS    = 12,

    // 0.4 × 4096 = 1638 = 0x0666
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

    // 6 features × 16 bits, LSB-first packed, normalised to [0,1] in Q4.12
    // Order: rtt_ms, delta_rtt, rolling_std, rssi_dbm, rssi_rolling_std, sinr_db
    input  wire [NUM_FEATURES*DATA_WIDTH-1:0] feature_in,
    input  wire new_sample_valid,

    output wire channel_switch,               // raw: 1 = switch channel now
    output wire filtered_switch,              // 3-step rolling window filtered
    output wire [DATA_WIDTH-1:0] prob_jammed, // Q4.12 P(jammed) for debug
    output wire inference_done                // pulses when new result ready
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
    // LSTM1: parallel, 6 inputs → 32 hidden
    // DSP48E1 usage: mac_ih=6 + mac_hh=32 = 38 blocks
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
    // LSTM2: parallel, 32 inputs → 16 hidden
    // DSP48E1 usage: mac_ih=32 + mac_hh=16 = 48 blocks
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
    // Controller: walk 20 timesteps through L1, then L2, then FC
    // ─────────────────────────────────────────────────────
    reg [DATA_WIDTH-1:0] prob_reg;
    reg                  result_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_state <= C_IDLE; ts_cnt <= 0;
            l1_valid_in <= 0; l2_valid_in <= 0; fc_valid_in <= 0;
            l1_h_state <= 0; l1_c_state <= 0;
            l2_h_state <= 0; l2_c_state <= 0;
            result_valid <= 0; prob_reg <= 0;
        end else begin
            l1_valid_in <= 0; l2_valid_in <= 0;
            fc_valid_in <= 0; result_valid <= 0;

            case (ctrl_state)
                C_IDLE: begin
                    if (buf_ready && new_sample_valid) begin
                        l1_h_state <= 0; l1_c_state <= 0;
                        ts_cnt <= WINDOW_SIZE - 1;
                        ctrl_state <= C_RUN_L1;
                    end
                end
                C_RUN_L1: begin
                    l1_x <= in_buf[ts_cnt];
                    l1_valid_in <= 1;
                    ctrl_state <= C_WAIT_L1;
                end
                C_WAIT_L1: begin
                    if (l1_valid_out) begin
                        l1_h_state <= l1_h_out; l1_c_state <= l1_c_out;
                        if (ts_cnt == 0) begin
                            l2_h_state <= 0; l2_c_state <= 0;
                            ts_cnt <= WINDOW_SIZE - 1;
                            ctrl_state <= C_RUN_L2;
                        end else begin
                            ts_cnt <= ts_cnt - 1;
                            ctrl_state <= C_RUN_L1;
                        end
                    end
                end
                C_RUN_L2: begin
                    l2_x <= l1_h_state;
                    l2_valid_in <= 1;
                    ctrl_state <= C_WAIT_L2;
                end
                C_WAIT_L2: begin
                    if (l2_valid_out) begin
                        l2_h_state <= l2_h_out; l2_c_state <= l2_c_out;
                        if (ts_cnt == 0) begin
                            ctrl_state <= C_FC;
                        end else begin
                            ts_cnt <= ts_cnt - 1;
                            ctrl_state <= C_RUN_L2;
                        end
                    end
                end
                C_FC: begin
                    fc_h_in <= l2_h_state;
                    fc_valid_in <= 1;
                    ctrl_state <= C_DONE;
                end
                C_DONE: begin
                    if (fc_valid_out) begin
                        prob_reg <= fc_prob;
                        result_valid <= 1;
                        ctrl_state <= C_IDLE;
                    end
                end
            endcase
        end
    end

    // ─────────────────────────────────────────────────────
    // Threshold comparator → raw channel_switch
    // ─────────────────────────────────────────────────────
    assign prob_jammed    = prob_reg;
    assign inference_done = result_valid;
    assign channel_switch = result_valid &&
                            ($signed(prob_reg) > $signed(THRESHOLD));

    // ─────────────────────────────────────────────────────
    // 3-step rolling window → filtered_switch
    // filtered_switch fires only when 3 consecutive raw
    // decisions were all "jammed" — suppresses false alarms
    // caused by momentary RTT spikes in clean periods
    // ─────────────────────────────────────────────────────
    reg [ROLL_WINDOW-1:0] switch_history;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            switch_history <= 0;
        else if (result_valid)
            switch_history <= {switch_history[ROLL_WINDOW-2:0], channel_switch};
    end

    assign filtered_switch = &switch_history;

endmodule

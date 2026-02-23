// =============================================================
// fc_layer.v — Fully Connected Layer (16 → 1)
// =============================================================
// Computes: out = sigmoid(W * h + b)
// where h is the LSTM2 final hidden state (16 values)
//
// Output is P(jammed) in Q4.12 format (range 0.0 to 1.0)
// For channel switch decision: if out > THRESHOLD → switch
//
// Fixed-point: Q4.12 (16-bit signed)
// =============================================================

module fc_layer #(
    parameter  INPUT_SIZE = 16,
    parameter  DATA_WIDTH = 16,
    parameter  FRAC_BITS  = 12,
    parameter  W_FILE     = "fc_weight.hex",
    parameter  B_FILE     = "fc_bias.hex"
)(
    input  wire clk,
    input  wire rst_n,

    // Input: LSTM2 final hidden state (16 × 16 bits)
    input  wire [INPUT_SIZE*DATA_WIDTH-1:0] h_in,
    input  wire valid_in,

    // Output: P(jammed) in Q4.12
    output reg  [DATA_WIDTH-1:0] prob_out,
    output reg  valid_out
);
    localparam ACC_WIDTH = 32;
    localparam ACC_SHIFT = FRAC_BITS;  // 12

    // Weight and bias ROMs
    reg signed [DATA_WIDTH-1:0] fc_w [0:INPUT_SIZE-1];
    reg signed [DATA_WIDTH-1:0] fc_b;

    initial begin
        $readmemh(W_FILE, fc_w);
        $readmemh(B_FILE, fc_b);
    end

    // Input register
    reg signed [DATA_WIDTH-1:0] h_reg [0:INPUT_SIZE-1];

    // Accumulator
    reg signed [ACC_WIDTH-1:0] acc;

    // LUT interface
    reg  [7:0]  sig_addr;
    wire [15:0] sig_data;

    sigmoid_lut u_sigmoid (.clk(clk), .addr(sig_addr), .data(sig_data));

    // FSM
    localparam S_IDLE      = 3'd0;
    localparam S_BIAS      = 3'd1;
    localparam S_MAC       = 3'd2;
    localparam S_SIG_ADDR  = 3'd3;
    localparam S_SIG_WAIT  = 3'd4;
    localparam S_DONE      = 3'd5;

    reg [2:0] state;
    reg [$clog2(INPUT_SIZE)-1:0] cnt;

    // LUT address mapping (same as lstm_cell)
    localparam LUT_SHIFT = ACC_SHIFT - 7;  // 5

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            valid_out <= 1'b0;
            cnt       <= 0;
            acc       <= 0;
        end else begin
            valid_out <= 1'b0;

            case (state)

                S_IDLE: begin
                    if (valid_in) begin
                        for (i = 0; i < INPUT_SIZE; i = i + 1)
                            h_reg[i] <= $signed(h_in[i*DATA_WIDTH +: DATA_WIDTH]);
                        cnt   <= 0;
                        state <= S_BIAS;
                    end
                end

                // Initialise accumulator with bias (scaled)
                S_BIAS: begin
                    acc   <= ($signed(fc_b) <<< ACC_SHIFT);
                    cnt   <= 0;
                    state <= S_MAC;
                end

                // dot product: sum(W[i] * h[i])
                S_MAC: begin
                    begin
                        reg signed [2*DATA_WIDTH-1:0] product;
                        product = $signed(fc_w[cnt]) * $signed(h_reg[cnt]);
                        acc     <= acc + product;
                    end
                    if (cnt == INPUT_SIZE - 1) begin
                        state <= S_SIG_ADDR;
                    end else begin
                        cnt <= cnt + 1;
                    end
                end

                // Issue sigmoid LUT address
                S_SIG_ADDR: begin
                    begin
                        reg signed [ACC_WIDTH-1:0] shifted;
                        shifted  = acc >>> LUT_SHIFT;
                        sig_addr <= shifted[7:0];
                    end
                    state <= S_SIG_WAIT;
                end

                // Wait 1 cycle for LUT pipeline
                S_SIG_WAIT: begin
                    state <= S_DONE;
                end

                S_DONE: begin
                    prob_out  <= sig_data;   // Q4.12 probability
                    valid_out <= 1'b1;
                    state     <= S_IDLE;
                end

            endcase
        end
    end

endmodule

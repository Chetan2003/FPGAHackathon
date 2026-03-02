// =============================================================
// fc_layer.v — Fully Connected Layer (16 → 1) (FIXED)
// =============================================================
// CHANGES:
//   [FIX-1] LUT address: corrected Q8.24 → 8-bit index with saturation
//   [FIX-5] $readmemh: bias loaded into array, not scalar register
//   [FIX-7] Bias sign-extension before shift to prevent truncation
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
    input  wire [INPUT_SIZE*DATA_WIDTH-1:0] h_in,
    input  wire valid_in,
    output reg  [DATA_WIDTH-1:0] prob_out,
    output reg  valid_out
);

    localparam ACC_WIDTH = 32;

    // Weight and bias ROMs
    reg signed [DATA_WIDTH-1:0] fc_w [0:INPUT_SIZE-1];

    // [FIX-5] Bias must be in an array for $readmemh
    reg signed [DATA_WIDTH-1:0] fc_b_rom [0:0];
    wire signed [DATA_WIDTH-1:0] fc_b = fc_b_rom[0];

    initial begin
        $readmemh(W_FILE, fc_w);
        $readmemh(B_FILE, fc_b_rom);
    end

    // Input register
    reg signed [DATA_WIDTH-1:0] h_reg [0:INPUT_SIZE-1];

    // Accumulator
    reg signed [ACC_WIDTH-1:0] acc;

    // Sigmoid LUT interface
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

    // [FIX-1] LUT address: Q8.24 → 8-bit index with saturation
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

                // [FIX-7] Sign-extend bias to ACC_WIDTH before shifting
                S_BIAS: begin
                    acc   <= {{(ACC_WIDTH-DATA_WIDTH){fc_b[DATA_WIDTH-1]}},
                              fc_b} <<< FRAC_BITS;
                    cnt   <= 0;
                    state <= S_MAC;
                end

                // Dot product: sum(W[i] * h[i])
                S_MAC: begin
                    begin
                        reg signed [2*DATA_WIDTH-1:0] product;
                        product = $signed(fc_w[cnt]) * $signed(h_reg[cnt]);
                        acc     <= acc + product;
                    end
                    if (cnt == INPUT_SIZE - 1)
                        state <= S_SIG_ADDR;
                    else
                        cnt <= cnt + 1;
                end

                // [FIX-1] Corrected sigmoid LUT address
                S_SIG_ADDR: begin
                    sig_addr <= acc_to_lut_addr(acc);
                    state    <= S_SIG_WAIT;
                end

                S_SIG_WAIT: begin
                    state <= S_DONE;
                end

                S_DONE: begin
                    prob_out  <= sig_data;
                    valid_out <= 1'b1;
                    state     <= S_IDLE;
                end
            endcase
        end
    end

endmodule
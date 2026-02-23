// =============================================================
// tanh_lut.v — Tanh Activation LUT
// =============================================================
// 256-entry lookup table implementing tanh(x)
//
// Input  : 8-bit index (taken from bits [14:7] of Q4.12 accumulator)
// Output : Q4.12 tanh value (range -1.0 to +1.0)
//
// Index mapping:
//   index 0   → input = -8.0    → tanh ≈ -0.99999
//   index 128 → input =  0.0    → tanh =  0.0
//   index 255 → input = +7.9375 → tanh ≈ +0.99999
//
// To use: addr = accumulator[14:7]
// =============================================================

module tanh_lut (
    input  wire        clk,
    input  wire [7:0]  addr,   // 8-bit index
    output reg  [15:0] data    // Q4.12 tanh output (signed)
);
    reg [15:0] lut [0:255];

    initial $readmemh("tanh_lut.hex", lut);

    always @(posedge clk)
        data <= lut[addr];

endmodule

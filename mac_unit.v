// =============================================================
// mac_unit.v — Single DSP48E1 Multiply-Accumulate Unit
// =============================================================
// Explicit DSP48E1 primitive instantiation for XC7Z020.
// Forces Vivado to use a dedicated DSP block, not LUT logic.
//
// Operation: P = A * B  (2-cycle latency: AREG+BREG → MREG)
//   A (weight)  : 16-bit signed Q4.12
//   B (data_in) : 16-bit signed Q4.12
//   P (product) : 32-bit signed Q8.24  (rescale >>12 externally)
//
// DSP48E1 pipeline:
//   Cycle 1: A,B registers latch inputs
//   Cycle 2: M register latches A*B result
//   Cycle 3: product visible on output
//
// OPMODE = 7'b000_0101 → P = A * B (no cascade, no pre-add)
// =============================================================

module mac_unit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        en,

    input  wire [15:0] weight,      // Q4.12 signed weight
    input  wire [15:0] data_in,     // Q4.12 signed feature/hidden

    output wire [31:0] product      // Q8.24 signed product
);

    // Sign-extend to DSP48E1 port widths
    wire [29:0] dsp_a = {{14{weight[15]}},  weight};
    wire [17:0] dsp_b = {{ 2{data_in[15]}}, data_in};
    wire [47:0] dsp_p;

    assign product = dsp_p[31:0];

    DSP48E1 #(
        .AREG           (1),
        .BREG           (1),
        .CREG           (0),
        .DREG           (0),
        .ADREG          (0),
        .MREG           (1),
        .PREG           (0),
        .USE_MULT       ("MULTIPLY"),
        .USE_DPORT      ("FALSE"),
        .USE_SIMD       ("ONE48"),
        .A_INPUT        ("DIRECT"),
        .B_INPUT        ("DIRECT")
    ) dsp_inst (
        .CLK            (clk),
        .CEA1           (en),
        .CEA2           (en),
        .CEB1           (en),
        .CEB2           (en),
        .CEM            (en),
        .CEP            (1'b0),
        .RSTA           (!rst_n),
        .RSTB           (!rst_n),
        .RSTM           (!rst_n),
        .RSTP           (1'b0),
        .A              (dsp_a),
        .B              (dsp_b),
        .C              (48'b0),
        .D              (25'b0),
        .OPMODE         (7'b000_0101),
        .ALUMODE        (4'b0000),
        .CARRYIN        (1'b0),
        .CARRYINSEL     (3'b000),
        .INMODE         (5'b00000),
        .ACIN           (30'b0),
        .BCIN           (18'b0),
        .PCIN           (48'b0),
        .P              (dsp_p),
        .CARRYOUT       (),
        .CARRYCASCOUT   (),
        .MULTSIGNOUT    (),
        .ACOUT          (),
        .BCOUT          (),
        .PCOUT          ()
    );

endmodule

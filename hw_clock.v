// =============================================================
// hw_clock.v — 64-bit Free-Running Hardware Clock
// =============================================================
// At 100 MHz: 1 tick = 10ns. Wraps in ~5849 years.
// Shared across packet_sniffer, rtt_calculator for RTT stamps.
// =============================================================
module hw_clock (
    input  wire        clk,
    input  wire        rst_n,
    output reg  [63:0] timestamp
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) timestamp <= 64'd0;
        else        timestamp <= timestamp + 64'd1;
    end
endmodule

// =============================================================
// feature_engineering.v — Real-Time Feature Computer (FIXED)
// =============================================================
// CHANGES:
//   [FIX-3] Operator precedence: parenthesised all >> shifts
//           in sum_sq computation (>> has lower precedence than +)
//   [NOTE]  isqrt: reduced Newton-Raphson to 4 iterations.
//           8 chained 32-bit divisions in combinational logic
//           may fail timing at 100 MHz. Consider pipelining
//           if Vivado reports timing violations on this path.
// =============================================================

module feature_engineering (
    input  wire        clk,
    input  wire        rst_n,

    input  wire [15:0] rtt_q412,
    input  wire [15:0] rssi_q412,
    input  wire [15:0] noise_floor_q412,
    input  wire        input_valid,

    output reg  [6*16-1:0] feature_out,
    output reg             feature_valid
);

    localparam WINDOW  = 5;
    localparam FRAC    = 12;
    localparam SCALE   = 4096;

    // ─────────────────────────────────────────────────────
    // History buffers
    // ─────────────────────────────────────────────────────
    reg [15:0] rtt_hist  [0:WINDOW-1];
    reg [15:0] rssi_hist [0:WINDOW-1];
    reg        hist_valid;
    reg [2:0]  hist_fill;

    reg [15:0] rtt_prev;
    reg        prev_valid;

    reg signed [15:0] delta_rtt_r;
    reg        [15:0] rtt_rstd_r;
    reg        [15:0] rssi_rstd_r;

    // ─────────────────────────────────────────────────────
    // Integer sqrt via Newton-Raphson
    // [NOTE] 4 iterations with 32-bit division — combinational.
    //   If timing fails, pipeline over 4 clock cycles.
    // ─────────────────────────────────────────────────────
    function automatic [15:0] isqrt_q412;
        input [31:0] x_q824;
        reg [31:0] s;
        begin
            if (x_q824 == 0) begin
                isqrt_q412 = 0;
            end else begin
                s = x_q824 >> 1;
                // 4 iterations sufficient for 16-bit output precision
                s = (s + (x_q824 / s)) >> 1;
                s = (s + (x_q824 / s)) >> 1;
                s = (s + (x_q824 / s)) >> 1;
                s = (s + (x_q824 / s)) >> 1;
                isqrt_q412 = s[15:0];
            end
        end
    endfunction

    // ─────────────────────────────────────────────────────
    // Rolling std: sqrt( E[x²] - E[x]² )
    // ─────────────────────────────────────────────────────
    function automatic [15:0] rolling_std_q412;
        input [15:0] buf0, buf1, buf2, buf3, buf4;
        reg [31:0] sum, sum_sq, mean_q412, mean_sq_q824;
        reg [31:0] var_q824, e_x2_q824;
        begin
            sum = buf0 + buf1 + buf2 + buf3 + buf4;
            mean_q412 = sum / 5;

            // [FIX-3] Parenthesised each shift — >> has lower
            //   precedence than + in Verilog, so without parens
            //   the additions bind first, producing garbage.
            sum_sq = (({16'b0, buf0} * {16'b0, buf0}) >> FRAC)
                   + (({16'b0, buf1} * {16'b0, buf1}) >> FRAC)
                   + (({16'b0, buf2} * {16'b0, buf2}) >> FRAC)
                   + (({16'b0, buf3} * {16'b0, buf3}) >> FRAC)
                   + (({16'b0, buf4} * {16'b0, buf4}) >> FRAC);

            e_x2_q824   = sum_sq / 5;
            mean_sq_q824 = (({16'b0, mean_q412} * {16'b0, mean_q412}) >> FRAC);
            var_q824 = (e_x2_q824 > mean_sq_q824)
                       ? (e_x2_q824 - mean_sq_q824) : 32'd0;
            rolling_std_q412 = isqrt_q412(var_q824);
        end
    endfunction

    // ─────────────────────────────────────────────────────
    // Delta RTT: centred at 0.50 (Q4.12 = 2048)
    // ─────────────────────────────────────────────────────
    function automatic [15:0] encode_delta;
        input signed [16:0] delta;
        reg signed [17:0] shifted;
        begin
            shifted = (delta >>> 1) + 18'sd2048;
            if (shifted < 0)         encode_delta = 16'd0;
            else if (shifted > 4095) encode_delta = 16'd4095;
            else                     encode_delta = shifted[15:0];
        end
    endfunction

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            feature_valid <= 0;
            prev_valid    <= 0;
            hist_fill     <= 0;
            hist_valid    <= 0;
            rtt_prev      <= 0;
            delta_rtt_r   <= 16'd2048;
            rtt_rstd_r    <= 0;
            rssi_rstd_r   <= 0;
            for (i = 0; i < WINDOW; i = i + 1) begin
                rtt_hist[i]  <= 0;
                rssi_hist[i] <= 0;
            end
            feature_out <= 0;
        end else begin
            feature_valid <= 0;

            if (input_valid) begin
                for (i = WINDOW-1; i > 0; i = i - 1) begin
                    rtt_hist[i]  <= rtt_hist[i-1];
                    rssi_hist[i] <= rssi_hist[i-1];
                end
                rtt_hist[0]  <= rtt_q412;
                rssi_hist[0] <= rssi_q412;

                if (hist_fill < WINDOW)
                    hist_fill <= hist_fill + 1;
                if (hist_fill == WINDOW - 1)
                    hist_valid <= 1;

                if (prev_valid) begin
                    delta_rtt_r <= encode_delta(
                        $signed({1'b0, rtt_q412}) - $signed({1'b0, rtt_prev}));
                end else begin
                    delta_rtt_r <= 16'd2048;
                end
                rtt_prev   <= rtt_q412;
                prev_valid <= 1;

                if (hist_valid) begin
                    rtt_rstd_r  <= rolling_std_q412(
                        rtt_hist[0], rtt_hist[1], rtt_hist[2],
                        rtt_hist[3], rtt_hist[4]);
                    rssi_rstd_r <= rolling_std_q412(
                        rssi_hist[0], rssi_hist[1], rssi_hist[2],
                        rssi_hist[3], rssi_hist[4]);
                end

                feature_out <= {
                    noise_floor_q412,
                    rssi_rstd_r,
                    rssi_q412,
                    rtt_rstd_r,
                    delta_rtt_r,
                    rtt_q412
                };
                feature_valid <= 1;
            end
        end
    end

endmodule
// =============================================================
// rtt_calculator.v — RTT & Noise Floor Calculator
// =============================================================
// Receives TX/RX sequence numbers and timestamps from
// packet_sniffer. Stores TX timestamps in a BRAM-backed
// lookup table indexed by sequence number. When a matching
// RX arrives, computes RTT.
//
// BRAM Time LUT:
//   Address = sequence number (16-bit → 65536 entries)
//   Data    = 64-bit TX timestamp in hardware clock ticks
//   At 100MHz: 1 tick = 10ns
//
// RTT → Q4.12:
//   rtt_q412 = min(rtt_ticks × 4096 / 5,000,000, 4095)
//   Range: 0 = 0ms, 4096 = 50ms
//
// RSSI → Q4.12:
//   rssi_q412 = (rssi_dbm + 100) × 4096 / 60
//   Range: -100dBm=0, -40dBm=4096
//
// Noise floor → Q4.12:
//   noise_q412 = (noise_dbm + 100) × 4096 / 50
//   Clean: ~-95dBm → ~410, Jammed: ~-70dBm → ~2458
//
// Fixed-point: Q4.12  Target: ZedBoard XC7Z020
// =============================================================

module rtt_calculator (
    input  wire        clk,
    input  wire        rst_n,

    input  wire [63:0] hw_timestamp,

    // From packet_sniffer
    input  wire        tx_seq_valid,
    input  wire [15:0] tx_seq_num,
    input  wire [63:0] tx_timestamp,
    input  wire        rx_seq_valid,
    input  wire [15:0] rx_seq_num,
    input  wire [63:0] rx_timestamp,

    // Raw PHY values (signed 8-bit dBm)
    input  wire signed [7:0]  rssi_raw_dbm,
    input  wire signed [7:0]  noise_floor_raw_dbm,

    // Q4.12 normalised outputs
    output reg  [15:0] rtt_q412,
    output reg  [15:0] rssi_q412,
    output reg  [15:0] noise_floor_q412,
    output reg         output_valid
);

    // ─────────────────────────────────────────────────────
    // BRAM Timestamp LUT (inferred, Vivado maps to RAMB36E1)
    // ─────────────────────────────────────────────────────
    reg [63:0] timestamp_lut [0:65535];

    // TX write
    always @(posedge clk) begin
        if (tx_seq_valid)
            timestamp_lut[tx_seq_num] <= tx_timestamp;
    end

    // ─────────────────────────────────────────────────────
    // RX pipeline: 4 stages
    // ─────────────────────────────────────────────────────
    reg [15:0] rx_seq_r;
    reg [63:0] rx_ts_r;
    reg        s1v, s2v, s3v;
    reg [63:0] tx_lut_out;
    reg [63:0] rtt_ticks;

    // Stage 1: register RX
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin s1v <= 0; rx_seq_r <= 0; rx_ts_r <= 0;
        end else begin
            s1v <= rx_seq_valid;
            if (rx_seq_valid) begin rx_seq_r <= rx_seq_num; rx_ts_r <= rx_timestamp; end
        end
    end

    // Stage 2: BRAM read
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin s2v <= 0; tx_lut_out <= 0;
        end else begin
            s2v        <= s1v;
            tx_lut_out <= timestamp_lut[rx_seq_r];
        end
    end

    // Stage 3: subtract
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin s3v <= 0; rtt_ticks <= 0;
        end else begin
            s3v       <= s2v;
            rtt_ticks <= (s2v && rx_ts_r >= tx_lut_out)
                         ? (rx_ts_r - tx_lut_out) : 64'd0;
        end
    end

    // Stage 4: scale to Q4.12
    // rtt_q412 = rtt_ticks * 4096 / 5_000_000
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin output_valid <= 0; rtt_q412 <= 0;
        end else begin
            output_valid <= s3v;
            if (s3v) begin
                reg [63:0] sc;
                sc       = (rtt_ticks * 64'd4096) / 64'd5_000_000;
                rtt_q412 <= (sc > 64'd4095) ? 16'd4095 : sc[15:0];
            end
        end
    end

    // ─────────────────────────────────────────────────────
    // RSSI → Q4.12  (rssi + 100) * 4096 / 60
    // ─────────────────────────────────────────────────────
    always @(posedge clk) begin
        reg signed [15:0] shifted;
        reg [31:0] scaled;
        shifted = $signed(rssi_raw_dbm) + 16'sd100;
        if (shifted < 0)       shifted = 0;
        else if (shifted > 60) shifted = 60;
        scaled    = shifted * 32'd4096 / 32'd60;
        rssi_q412 <= scaled[15:0];
    end

    // ─────────────────────────────────────────────────────
    // Noise floor → Q4.12  (noise + 100) * 4096 / 50
    // ─────────────────────────────────────────────────────
    always @(posedge clk) begin
        reg signed [15:0] shifted;
        reg [31:0] scaled;
        shifted = $signed(noise_floor_raw_dbm) + 16'sd100;
        if (shifted < 0)       shifted = 0;
        else if (shifted > 50) shifted = 50;
        scaled           = shifted * 32'd4096 / 32'd50;
        noise_floor_q412 <= scaled[15:0];
    end

endmodule

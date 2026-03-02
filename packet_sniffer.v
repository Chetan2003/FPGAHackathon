// =============================================================
// packet_sniffer.v — UDP Packet Sniffer FSM
// =============================================================
// Sits on the AXI Stream bus between the drone's WiFi radio
// and the antenna. Passively taps every packet — the drone's
// software is completely unmodified.
//
// Parses Ethernet/IP/UDP headers to extract:
//   - Sequence number from payload (bytes 0-1 of UDP payload)
//   - Direction: TX (dst port 8888) or RX (dst port 8889)
//   - Timestamp: captured from hardware clock at first byte
//
// Packet structure (byte offsets from Ethernet frame start):
//   [0..5]   Dst MAC
//   [6..11]  Src MAC
//   [12..13] EtherType (0x0800 = IPv4)
//   [14]     IP version+IHL (0x45 = IPv4, no options)
//   [15]     DSCP/ECN
//   [16..17] Total length
//   [18..19] Identification
//   [20..21] Flags+Fragment offset
//   [22]     TTL
//   [23]     Protocol (0x11 = UDP)
//   [24..25] Header checksum
//   [26..29] Src IP
//   [30..33] Dst IP
//   [34..35] UDP Src port
//   [36..37] UDP Dst port  ← 8888=PING_PORT(tx), 8889=ECHO_PORT(rx)
//   [38..39] UDP Length
//   [40..41] UDP Checksum
//   [42..43] Payload[0:1] = Sequence number (uint16, big-endian)
//   [44..51] Payload[2:9] = Tx timestamp in ns (uint64, big-endian)
//
// AXI Stream input: 8-bit TDATA, TVALID, TREADY, TLAST
// (byte-by-byte stream — one byte per clock when TVALID=1)
//
// Outputs:
//   tx_seq_valid   : pulses when TX sequence number extracted
//   tx_seq_num     : extracted TX sequence number
//   tx_timestamp   : hardware clock value at TX packet arrival
//   rx_seq_valid   : pulses when RX sequence number extracted
//   rx_seq_num     : extracted RX sequence number
//   rx_timestamp   : hardware clock value at RX packet arrival
//
// Target: Xilinx ZedBoard (XC7Z020)
// =============================================================

module packet_sniffer (
    input  wire        clk,
    input  wire        rst_n,

    // AXI Stream slave — from WiFi PHY
    input  wire [7:0]  s_axis_tdata,
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,   // always ready — passive sniffer
    input  wire        s_axis_tlast,    // end of frame

    // 64-bit hardware clock (1 tick = 10ns at 100MHz)
    input  wire [63:0] hw_timestamp,

    // TX packet outputs (dst port = PING_PORT = 8888)
    output reg         tx_seq_valid,
    output reg [15:0]  tx_seq_num,
    output reg [63:0]  tx_timestamp,

    // RX packet outputs (dst port = ECHO_PORT = 8889)
    output reg         rx_seq_valid,
    output reg [15:0]  rx_seq_num,
    output reg [63:0]  rx_timestamp
);

    // ─────────────────────────────────────────────────────
    // Constants
    // ─────────────────────────────────────────────────────
    localparam PING_PORT_HI = 8'h22;  // 8888 = 0x22B8
    localparam PING_PORT_LO = 8'hB8;
    localparam ECHO_PORT_HI = 8'h22;  // 8889 = 0x22B9
    localparam ECHO_PORT_LO = 8'hB9;
    localparam UDP_PROTO    = 8'h11;  // IP protocol = UDP
    localparam IPV4_ETYPE_HI = 8'h08;
    localparam IPV4_ETYPE_LO = 8'h00;

    // Byte offset FSM state — tracks which byte we're currently at
    // in the Ethernet frame
    localparam S_IDLE      = 4'd0;
    localparam S_ETH_HDR   = 4'd1;   // bytes 0..13 (Ethernet header)
    localparam S_IP_HDR    = 4'd2;   // bytes 14..33 (IP header)
    localparam S_UDP_SP    = 4'd3;   // bytes 34..35 (UDP src port)
    localparam S_UDP_DP_HI = 4'd4;   // byte  36     (UDP dst port high)
    localparam S_UDP_DP_LO = 4'd5;   // byte  37     (UDP dst port low)
    localparam S_UDP_REST  = 4'd6;   // bytes 38..41 (len + checksum, skip)
    localparam S_SEQ_HI    = 4'd7;   // byte  42     (seq num high)
    localparam S_SEQ_LO    = 4'd8;   // byte  43     (seq num low)
    localparam S_SKIP      = 4'd9;   // skip rest of frame
    localparam S_WAIT_END  = 4'd10;  // wait for TLAST

    reg [3:0]  state;
    reg [7:0]  byte_cnt;       // byte counter within current header section
    reg        is_tx_pkt;      // 1=TX (ping), 0=RX (echo)
    reg        is_udp;         // 1 if IP protocol = UDP
    reg        is_ipv4;        // 1 if EtherType = 0x0800
    reg [7:0]  dst_port_hi;    // saved dst port high byte
    reg [63:0] pkt_timestamp;  // timestamp latched at frame start

    // Passthrough: sniffer never blocks the stream
    assign s_axis_tready = 1'b1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            byte_cnt     <= 0;
            is_tx_pkt    <= 0;
            is_udp       <= 0;
            is_ipv4      <= 0;
            tx_seq_valid <= 0;
            rx_seq_valid <= 0;
            tx_seq_num   <= 0;
            rx_seq_num   <= 0;
            tx_timestamp <= 0;
            rx_timestamp <= 0;
        end else begin
            // Default: deassert one-cycle pulse outputs
            tx_seq_valid <= 0;
            rx_seq_valid <= 0;

            if (s_axis_tvalid) begin
                case (state)

                    // ── New frame begins ──────────────────────
                    S_IDLE: begin
                        // Latch hardware timestamp at first byte of frame
                        pkt_timestamp <= hw_timestamp;
                        is_ipv4       <= 0;
                        is_udp        <= 0;
                        byte_cnt      <= 1;
                        state         <= S_ETH_HDR;
                    end

                    // ── Ethernet header (14 bytes) ─────────────
                    // Bytes 0..11: MAC addresses (skip)
                    // Bytes 12..13: EtherType
                    S_ETH_HDR: begin
                        if (byte_cnt == 12 &&
                            s_axis_tdata == IPV4_ETYPE_HI)
                            is_ipv4 <= 1;  // tentative
                        else if (byte_cnt == 13) begin
                            if (!is_ipv4 || s_axis_tdata != IPV4_ETYPE_LO)
                                is_ipv4 <= 0;  // not IPv4, skip frame
                        end

                        if (byte_cnt == 13) begin
                            byte_cnt <= 14;
                            state    <= is_ipv4 ? S_IP_HDR : S_WAIT_END;
                        end else
                            byte_cnt <= byte_cnt + 1;
                    end

                    // ── IP header (20 bytes, bytes 14..33) ────
                    // Byte 23: Protocol field
                    S_IP_HDR: begin
                        if (byte_cnt == 23)
                            is_udp <= (s_axis_tdata == UDP_PROTO);

                        if (byte_cnt == 33) begin
                            byte_cnt <= 34;
                            state    <= is_udp ? S_UDP_SP : S_WAIT_END;
                        end else
                            byte_cnt <= byte_cnt + 1;
                    end

                    // ── UDP source port (bytes 34..35, skip) ──
                    S_UDP_SP: begin
                        if (byte_cnt == 35) begin
                            byte_cnt <= 36;
                            state    <= S_UDP_DP_HI;
                        end else
                            byte_cnt <= byte_cnt + 1;
                    end

                    // ── UDP dst port high byte (byte 36) ──────
                    S_UDP_DP_HI: begin
                        dst_port_hi <= s_axis_tdata;
                        byte_cnt    <= 37;
                        state       <= S_UDP_DP_LO;
                    end

                    // ── UDP dst port low byte (byte 37) ───────
                    // Determine if this is a TX (8888) or RX (8889) packet
                    S_UDP_DP_LO: begin
                        if (dst_port_hi == PING_PORT_HI &&
                            s_axis_tdata == PING_PORT_LO)
                            is_tx_pkt <= 1;   // PING_PORT=8888 → TX packet
                        else if (dst_port_hi == ECHO_PORT_HI &&
                                 s_axis_tdata == ECHO_PORT_LO)
                            is_tx_pkt <= 0;   // ECHO_PORT=8889 → RX packet
                        else begin
                            state <= S_WAIT_END;  // not our port, skip
                            byte_cnt <= 38;
                        end

                        if (state != S_WAIT_END) begin
                            byte_cnt <= 38;
                            state    <= S_UDP_REST;
                        end
                    end

                    // ── Skip UDP length + checksum (bytes 38..41) ──
                    S_UDP_REST: begin
                        if (byte_cnt == 41) begin
                            byte_cnt <= 42;
                            state    <= S_SEQ_HI;
                        end else
                            byte_cnt <= byte_cnt + 1;
                    end

                    // ── Payload seq num high byte (byte 42) ───
                    S_SEQ_HI: begin
                        // Re-use dst_port_hi register to save seq high
                        dst_port_hi <= s_axis_tdata;
                        byte_cnt    <= 43;
                        state       <= S_SEQ_LO;
                    end

                    // ── Payload seq num low byte (byte 43) ────
                    // Output extracted sequence number
                    S_SEQ_LO: begin
                        if (is_tx_pkt) begin
                            tx_seq_num   <= {dst_port_hi, s_axis_tdata};
                            tx_timestamp <= pkt_timestamp;
                            tx_seq_valid <= 1'b1;
                        end else begin
                            rx_seq_num   <= {dst_port_hi, s_axis_tdata};
                            rx_timestamp <= pkt_timestamp;
                            rx_seq_valid <= 1'b1;
                        end
                        state <= S_WAIT_END;
                    end

                    // ── Wait for TLAST (end of frame) ─────────
                    S_WAIT_END: begin
                        // Nothing — just waiting for frame end
                    end

                    // ── Should never reach here ────────────────
                    S_SKIP: state <= S_WAIT_END;

                    default: state <= S_IDLE;
                endcase

                // Frame end: reset for next packet
                if (s_axis_tlast)
                    state <= S_IDLE;
            end
        end
    end

endmodule

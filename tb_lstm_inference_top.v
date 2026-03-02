// =============================================================
// tb_lstm_inference_top.v — Testbench
// =============================================================
// Tests the hardware-accelerated LSTM inference engine.
//
// Test plan:
//   TEST 1 — mac_unit isolation
//            Known dot product, verify adder tree result
//   TEST 2 — Reset behaviour
//            Assert rst_n, verify outputs clear
//   TEST 3 — Buffer fill (first 20 samples)
//            No inference before WINDOW_SIZE samples collected
//   TEST 4 — Clean scenario (all features = nominal clean values)
//            Expect P(jammed) < THRESHOLD → no channel_switch
//   TEST 5 — Jammed scenario (features = jamming values)
//            Expect P(jammed) > THRESHOLD → channel_switch asserts
//   TEST 6 — Rolling window filter
//            Single jammed sample → filtered_switch should NOT fire
//            Three consecutive jammed → filtered_switch fires
//   TEST 7 — Inference latency measurement
//            Count cycles from new_sample_valid → inference_done
//            Must be < 40,000 cycles (400µs @ 100MHz)
//   TEST 8 — Back-to-back inferences (no gap between samples)
//            Second inference starts immediately, no deadlock
//
// Q4.12 encoding reminder:
//   float → Q4.12: int16 = round(float * 4096)
//   e.g.  0.5  → 0x0800 = 2048
//         0.1  → 0x0199 = 409
//         0.9  → 0x0E66 = 3686
//         0.0  → 0x0000
//         1.0  → 0x1000 = 4096
//
// Feature order (normalised 0→1):
//   [0] rtt_ms           clean ≈ 0.02,  jammed ≈ 0.60
//   [1] delta_rtt        clean ≈ 0.50,  jammed ≈ 0.90  (centred at 0.5)
//   [2] rolling_std      clean ≈ 0.02,  jammed ≈ 0.50
//   [3] rssi_dbm         clean ≈ 0.80,  jammed ≈ 0.30
//   [4] rssi_rolling_std clean ≈ 0.02,  jammed ≈ 0.40
//   [5] noise_floor_dbm  clean ≈ 0.05,  jammed ≈ 0.80
//
// Run in Vivado Simulator:
//   Set tb_lstm_inference_top as top simulation source.
//   All .hex files must be in the simulation working directory.
//   Run for at least 5,000,000 ns.
// =============================================================

`timescale 1ns/1ps

module tb_lstm_inference_top;

    // ─────────────────────────────────────────────────────
    // Parameters
    // ─────────────────────────────────────────────────────
    localparam CLK_PERIOD   = 10;     // 100 MHz → 10ns period
    localparam DATA_WIDTH   = 16;
    localparam NUM_FEATURES = 6;
    localparam WINDOW_SIZE  = 20;
    localparam FRAC_BITS    = 12;
    localparam SCALE        = 4096;   // 2^FRAC_BITS

    // Q4.12 encoding helper — use in initial blocks
    // (not a real function, just for readability in comments)
    // q412(x) = round(x * 4096) as signed 16-bit

    // ─────────────────────────────────────────────────────
    // DUT signals
    // ─────────────────────────────────────────────────────
    reg                                     clk;
    reg                                     rst_n;
    reg  [NUM_FEATURES*DATA_WIDTH-1:0]      feature_in;
    reg                                     new_sample_valid;

    wire                                    channel_switch;
    wire                                    filtered_switch;
    wire [DATA_WIDTH-1:0]                   prob_jammed;
    wire                                    inference_done;

    // ─────────────────────────────────────────────────────
    // Instantiate DUT
    // ─────────────────────────────────────────────────────
    lstm_inference_top #(
        .NUM_FEATURES (6),
        .WINDOW_SIZE  (20),
        .HIDDEN1      (32),
        .HIDDEN2      (16),
        .DATA_WIDTH   (16),
        .FRAC_BITS    (12),
        .THRESHOLD    (16'h0666),   // 0.40 in Q4.12
        .ROLL_WINDOW  (3)
    ) dut (
        .clk              (clk),
        .rst_n            (rst_n),
        .feature_in       (feature_in),
        .new_sample_valid (new_sample_valid),
        .channel_switch   (channel_switch),
        .filtered_switch  (filtered_switch),
        .prob_jammed      (prob_jammed),
        .inference_done   (inference_done)
    );

    // ─────────────────────────────────────────────────────
    // Clock generator: 100 MHz
    // ─────────────────────────────────────────────────────
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ─────────────────────────────────────────────────────
    // Test infrastructure
    // ─────────────────────────────────────────────────────
    integer test_num;
    integer pass_count;
    integer fail_count;
    integer cycle_count;
    integer latency_cycles;
    integer i;

    // Timeout watchdog: if no inference_done within this many cycles → fail
    localparam TIMEOUT_CYCLES = 100_000;

    // ─────────────────────────────────────────────────────
    // Q4.12 helper functions
    // ─────────────────────────────────────────────────────
    // Convert float to Q4.12 — use integer literals below
    // float 0.02  → 82  (0x0052)
    // float 0.05  → 205 (0x00CD)
    // float 0.10  → 410 (0x019A)
    // float 0.30  → 1229 (0x04CD)
    // float 0.40  → 1638 (0x0666)
    // float 0.50  → 2048 (0x0800)
    // float 0.60  → 2458 (0x099A)
    // float 0.80  → 3277 (0x0CCD)
    // float 0.90  → 3686 (0x0E66)
    // float 1.00  → 4096 (0x1000)

    // Pre-built feature vectors (packed, LSB = feature[0])
    // Pack order: feature[5] at MSB end, feature[0] at LSB
    // feature_in[15:0]   = feature[0] = rtt_ms
    // feature_in[31:16]  = feature[1] = delta_rtt
    // feature_in[47:32]  = feature[2] = rolling_std
    // feature_in[63:48]  = feature[3] = rssi_dbm
    // feature_in[79:64]  = feature[4] = rssi_rolling_std
    // feature_in[95:80]  = feature[5] = noise_floor_dbm

    // CLEAN sample: rtt=0.02, delta=0.50, std=0.02, rssi=0.80, rstd=0.02, noise=0.05
    localparam [95:0] FEAT_CLEAN = {
        16'd205,    // noise_floor_dbm = 0.05
        16'd82,     // rssi_rolling_std = 0.02
        16'd3277,   // rssi_dbm = 0.80
        16'd82,     // rolling_std = 0.02
        16'd2048,   // delta_rtt = 0.50 (centred)
        16'd82      // rtt_ms = 0.02
    };

    // JAMMED sample: rtt=0.60, delta=0.90, std=0.50, rssi=0.30, rstd=0.40, noise=0.80
    localparam [95:0] FEAT_JAMMED = {
        16'd3277,   // noise_floor_dbm = 0.80  ← key indicator
        16'd1638,   // rssi_rolling_std = 0.40
        16'd1229,   // rssi_dbm = 0.30
        16'd2048,   // rolling_std = 0.50
        16'd3686,   // delta_rtt = 0.90
        16'd2458    // rtt_ms = 0.60
    };

    // NEUTRAL sample: all mid-range (0.50)
    localparam [95:0] FEAT_NEUTRAL = {
        16'd2048, 16'd2048, 16'd2048,
        16'd2048, 16'd2048, 16'd2048
    };

    // ─────────────────────────────────────────────────────
    // Task: send one sample and wait one cycle
    // ─────────────────────────────────────────────────────
    task send_sample;
        input [NUM_FEATURES*DATA_WIDTH-1:0] feat;
        begin
            @(posedge clk);
            feature_in       <= feat;
            new_sample_valid <= 1'b1;
            @(posedge clk);
            new_sample_valid <= 1'b0;
        end
    endtask

    // ─────────────────────────────────────────────────────
    // Task: fill buffer with N copies of a feature vector
    // ─────────────────────────────────────────────────────
    task fill_buffer;
        input [NUM_FEATURES*DATA_WIDTH-1:0] feat;
        input integer n;
        integer j;
        begin
            for (j = 0; j < n; j = j + 1) begin
                send_sample(feat);
                repeat(5) @(posedge clk);  // small gap between samples
            end
        end
    endtask

    // ─────────────────────────────────────────────────────
    // Task: wait for inference_done with timeout
    // Returns latency in latency_cycles
    // ─────────────────────────────────────────────────────
    task wait_for_inference;
        output integer lat;
        integer timeout;
        begin
            lat     = 0;
            timeout = 0;
            while (!inference_done && timeout < TIMEOUT_CYCLES) begin
                @(posedge clk);
                lat     = lat + 1;
                timeout = timeout + 1;
            end
            if (timeout >= TIMEOUT_CYCLES) begin
                $display("  TIMEOUT: inference_done never asserted!");
                lat = -1;
            end
        end
    endtask

    // ─────────────────────────────────────────────────────
    // Task: check and report pass/fail
    // ─────────────────────────────────────────────────────
    task check;
        input       condition;
        input [255:0] description;
        begin
            if (condition) begin
                $display("  ✅ PASS: %s", description);
                pass_count = pass_count + 1;
            end else begin
                $display("  ❌ FAIL: %s", description);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // ─────────────────────────────────────────────────────
    // mac_unit standalone test signals
    // ─────────────────────────────────────────────────────
    // ─────────────────────────────────────────────────────
    // [FIX-6] mac_unit standalone test — uses ACTUAL interface
    //   (original TB used a non-existent vector dot-product interface)
    // ─────────────────────────────────────────────────────
    	reg  signed [15:0] mac_weight, mac_data;
    	reg                mac_en;
   	 wire signed [31:0] mac_product;

  	  mac_unit u_mac_test (
        	.clk     (clk),
       	 .rst_n   (rst_n),
      	 .en      (mac_en),
       	 .weight  (mac_weight),
        	.data_in (mac_data),
        	.product (mac_product)
   	 );

    // ─────────────────────────────────────────────────────
    // Signed decode helper (for $display)
    // ─────────────────────────────────────────────────────
    function real q412_to_float;
        input [15:0] q;
        reg signed [15:0] sq;
        begin
            sq = q;
            q412_to_float = $itor(sq) / 4096.0;
        end
    endfunction

    // ─────────────────────────────────────────────────────
    // MAIN TEST SEQUENCE
    // ─────────────────────────────────────────────────────
    initial begin
        // Waveform dump for Vivado/GTKWave
        $dumpfile("tb_lstm_inference_top.vcd");
        $dumpvars(0, tb_lstm_inference_top);

        pass_count       = 0;
        fail_count       = 0;
        new_sample_valid = 0;
        feature_in       = 0;
        mac_valid_in     = 0;
        mac_a_vec        = 0;
        mac_b_vec        = 0;

        $display("");
        $display("=======================================================");
        $display("  LSTM Inference Engine Testbench");
        $display("  Target: Xilinx ZedBoard XC7Z020 @ 100 MHz");
        $display("  Fixed-point: Q4.12 (scale = 4096)");
        $display("=======================================================");

        // ─────────────────────────────────────────────────
        	// TEST 1 — mac_unit single multiply verification
        	// ─────────────────────────────────────────────────
        	test_num = 1;
        	$display("\n--- TEST %0d: mac_unit single multiply ---", test_num);
        	$display("  weight = 0.5 (Q4.12 = 2048 = 0x0800)");
        	$display("  data   = 0.25 (Q4.12 = 1024 = 0x0400)");
        	$display("  Expected: 0.5 * 0.25 = 0.125");
        	$display("  Expected Q8.24: 0.125 * 2^24 = 2097152 = 0x200000");

        	rst_n = 0;
        	repeat(4) @(posedge clk);
        	rst_n = 1;
        	repeat(2) @(posedge clk);

        	mac_weight = 16'sh0800;  // 0.5 in Q4.12
        	mac_data   = 16'sh0400;  // 0.25 in Q4.12
        	mac_en     = 1'b1;

        	@(posedge clk);  // cycle 0: inputs presented
        	@(posedge clk);  // cycle 1: AREG/BREG latch
        	@(posedge clk);  // cycle 2: MREG latches product
        	@(posedge clk);  // cycle 3: read stable output

        	mac_en = 1'b0;

        	$display("  mac_product (Q8.24) = 0x%08X = %0d", mac_product, $signed(mac_product));
        	$display("  decoded = %f (expected 0.125)",
                    $itor($signed(mac_product)) / (4096.0 * 4096.0));

        // 0.125 in Q8.24 = 2097152 = 0x200000. Allow ±1 LSB.
        check($signed(mac_product) > 32'sh1FF000 &&
              $signed(mac_product) < 32'sh201000,
              "mac_unit: 0.5 * 0.25 = 0.125 within tolerance");

        // Test negative: -1.0 * 0.5 = -0.5
        $display("  Second test: -1.0 * 0.5 = -0.5");
        mac_weight = 16'shF000;  // -1.0 in Q4.12
        mac_data   = 16'sh0800;  // 0.5 in Q4.12
        mac_en     = 1'b1;

        repeat(4) @(posedge clk);
        mac_en = 1'b0;

        $display("  mac_product (Q8.24) = 0x%08X", mac_product);
        $display("  decoded = %f (expected -0.5)",
                 $itor($signed(mac_product)) / (4096.0 * 4096.0));
        // -0.5 in Q8.24 = -8388608 = 0xFF800000
        check($signed(mac_product) < 0 &&
              $signed(mac_product) > -32'sh900000,
              "mac_unit: -1.0 * 0.5 = -0.5 within tolerance");

        // ─────────────────────────────────────────────────
        // TEST 2 — Reset behaviour
        // ─────────────────────────────────────────────────
        test_num = 2;
        $display("\n--- TEST %0d: Reset behaviour ---", test_num);

        rst_n = 0;
        repeat(10) @(posedge clk);
        check(channel_switch  === 1'b0, "channel_switch = 0 during reset");
        check(filtered_switch === 1'b0, "filtered_switch = 0 during reset");
        check(inference_done  === 1'b0, "inference_done = 0 during reset");
        check(prob_jammed     === 16'h0000, "prob_jammed = 0 during reset");

        rst_n = 1;
        repeat(4) @(posedge clk);
        check(channel_switch  === 1'b0, "channel_switch = 0 after reset release");
        $display("  Reset behaviour: OK");

        // ─────────────────────────────────────────────────
        // TEST 3 — Buffer fill: no inference before WINDOW_SIZE samples
        // ─────────────────────────────────────────────────
        test_num = 3;
        $display("\n--- TEST %0d: No inference before buffer full ---", test_num);

        // Send 19 samples (one short of WINDOW_SIZE=20)
        for (i = 0; i < WINDOW_SIZE - 1; i = i + 1) begin
            send_sample(FEAT_NEUTRAL);
            repeat(3) @(posedge clk);
        end
        check(inference_done === 1'b0,
              "No inference_done after 19 samples (buffer not yet full)");
        $display("  Sent %0d samples, inference_done = %b (expect 0)",
                 WINDOW_SIZE-1, inference_done);

        // ─────────────────────────────────────────────────
        // TEST 4 — Clean scenario: P(jammed) should stay below threshold
        // ─────────────────────────────────────────────────
        test_num = 4;
        $display("\n--- TEST %0d: Clean scenario ---", test_num);
        $display("  Features: rtt=0.02, noise_floor=0.05 (clean/nominal)");

        // Reset and fill buffer with clean samples
        rst_n = 0; repeat(4) @(posedge clk); rst_n = 1; repeat(4) @(posedge clk);
        fill_buffer(FEAT_CLEAN, WINDOW_SIZE);

        // Trigger one more sample to start inference
        send_sample(FEAT_CLEAN);

        // Wait for result
        wait_for_inference(latency_cycles);

        if (latency_cycles >= 0) begin
            $display("  Inference done after %0d cycles (%0.1f µs @ 100MHz)",
                     latency_cycles, latency_cycles * 0.01);
            $display("  P(jammed) raw = 0x%04X = %f",
                     prob_jammed, q412_to_float(prob_jammed));
            check(latency_cycles > 0,  "Inference completed");
            check($signed(prob_jammed) < $signed(16'h0666),
                  "Clean: P(jammed) < 0.40 threshold");
            check(channel_switch === 1'b0,
                  "Clean: channel_switch not asserted");
        end else begin
            $display("  TIMEOUT on clean scenario");
            fail_count = fail_count + 1;
        end

        // ─────────────────────────────────────────────────
        // TEST 5 — Jammed scenario: P(jammed) should exceed threshold
        // ─────────────────────────────────────────────────
        test_num = 5;
        $display("\n--- TEST %0d: Jammed scenario ---", test_num);
        $display("  Features: rtt=0.60, noise_floor=0.80 (jamming)");

        // Reset and fill buffer with jammed samples
        rst_n = 0; repeat(4) @(posedge clk); rst_n = 1; repeat(4) @(posedge clk);
        fill_buffer(FEAT_JAMMED, WINDOW_SIZE);

        send_sample(FEAT_JAMMED);
        wait_for_inference(latency_cycles);

        if (latency_cycles >= 0) begin
            $display("  Inference done after %0d cycles (%0.1f µs @ 100MHz)",
                     latency_cycles, latency_cycles * 0.01);
            $display("  P(jammed) raw = 0x%04X = %f",
                     prob_jammed, q412_to_float(prob_jammed));
            check($signed(prob_jammed) > $signed(16'h0666),
                  "Jammed: P(jammed) > 0.40 threshold");
            check(channel_switch === 1'b1,
                  "Jammed: channel_switch asserted");
        end else begin
            $display("  TIMEOUT on jammed scenario");
            fail_count = fail_count + 1;
        end

        // ─────────────────────────────────────────────────
        // TEST 6 — Rolling window filter
        // ─────────────────────────────────────────────────
        test_num = 6;
        $display("\n--- TEST %0d: Rolling window filter ---", test_num);

        // Reset, fill buffer with clean samples first
        rst_n = 0; repeat(4) @(posedge clk); rst_n = 1; repeat(4) @(posedge clk);
        fill_buffer(FEAT_CLEAN, WINDOW_SIZE);

        // Sub-test 6a: Single jammed inference — filtered_switch must NOT fire
        $display("  6a: Single jammed sample after clean history");
        fill_buffer(FEAT_CLEAN, WINDOW_SIZE - 1);   // 19 clean
        send_sample(FEAT_JAMMED);                    // 1 jammed
        wait_for_inference(latency_cycles);
        $display("  channel_switch=%b  filtered_switch=%b (expect 1,0)",
                 channel_switch, filtered_switch);
        // filtered_switch needs 3 consecutive — should be 0 here
        check(filtered_switch === 1'b0,
              "Single jammed: filtered_switch NOT asserted (rolling window)");

        // Sub-test 6b: Three consecutive jammed inferences → filtered_switch fires
        $display("  6b: Three consecutive jammed inferences");
        rst_n = 0; repeat(4) @(posedge clk); rst_n = 1; repeat(4) @(posedge clk);
        fill_buffer(FEAT_JAMMED, WINDOW_SIZE);

        repeat(3) begin
            send_sample(FEAT_JAMMED);
            wait_for_inference(latency_cycles);
            repeat(10) @(posedge clk);
        end
        $display("  channel_switch=%b  filtered_switch=%b (expect 1,1)",
                 channel_switch, filtered_switch);
        check(filtered_switch === 1'b1,
              "Three consecutive jammed: filtered_switch asserted");

        // Sub-test 6c: After 3 clean, filtered_switch should clear
        $display("  6c: Three consecutive clean inferences — filtered_switch clears");
        fill_buffer(FEAT_CLEAN, WINDOW_SIZE);
        repeat(3) begin
            send_sample(FEAT_CLEAN);
            wait_for_inference(latency_cycles);
            repeat(10) @(posedge clk);
        end
        $display("  channel_switch=%b  filtered_switch=%b (expect 0,0)",
                 channel_switch, filtered_switch);
        check(filtered_switch === 1'b0,
              "Three clean after jammed: filtered_switch cleared");

        // ─────────────────────────────────────────────────
        // TEST 7 — Inference latency measurement
        // ─────────────────────────────────────────────────
        test_num = 7;
        $display("\n--- TEST %0d: Inference latency ---", test_num);
        $display("  Target: < 40,000 cycles (400 µs @ 100 MHz)");
        $display("  Design goal: ~38,900 cycles (389 µs)");

        rst_n = 0; repeat(4) @(posedge clk); rst_n = 1; repeat(4) @(posedge clk);
        fill_buffer(FEAT_CLEAN, WINDOW_SIZE);

        // Measure exactly
        @(posedge clk);
        feature_in       <= FEAT_CLEAN;
        new_sample_valid <= 1'b1;
        @(posedge clk);
        new_sample_valid <= 1'b0;

        cycle_count = 0;
        while (!inference_done && cycle_count < TIMEOUT_CYCLES) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end

        if (cycle_count < TIMEOUT_CYCLES) begin
            $display("  Measured latency: %0d cycles = %0.2f µs",
                     cycle_count, cycle_count * 0.01);
            check(cycle_count < 40000,
                  "Latency < 40,000 cycles (400 µs budget)");
            check(cycle_count < 100000,
                  "Latency < 100,000 cycles (hard timeout)");
            // Verify it's faster than serial implementation
            // Serial would be ~162,000 cycles
            check(cycle_count < 162000,
                  "Parallel is faster than serial reference (162,000 cycles)");
        end else begin
            $display("  TIMEOUT: inference never completed");
            fail_count = fail_count + 1;
        end

        // ─────────────────────────────────────────────────
        // TEST 8 — Back-to-back inferences (no deadlock)
        // ─────────────────────────────────────────────────
        test_num = 8;
        $display("\n--- TEST %0d: Back-to-back inferences ---", test_num);

        rst_n = 0; repeat(4) @(posedge clk); rst_n = 1; repeat(4) @(posedge clk);
        fill_buffer(FEAT_CLEAN, WINDOW_SIZE);

        begin : back_to_back
            integer done_count;
            integer bcount;
            done_count = 0;
            bcount = 0;

            // Send 5 samples as fast as possible (one per inference interval)
            fork
                // Thread 1: send samples
                begin
                    repeat(5) begin
                        send_sample(FEAT_NEUTRAL);
                        // Wait for inference to complete before sending next
                        // (in real system samples arrive every 100ms = 10,000,000 cycles)
                        // Here we wait just long enough for one inference
                        repeat(50000) @(posedge clk);
                    end
                end
                // Thread 2: count completions
                begin
                    repeat(5) begin
                        @(posedge inference_done);
                        done_count = done_count + 1;
                        $display("  Inference %0d complete, P(jammed)=0x%04X (%f)",
                                 done_count, prob_jammed, q412_to_float(prob_jammed));
                    end
                end
            join
            check(done_count == 5, "All 5 back-to-back inferences completed");
        end

        // ─────────────────────────────────────────────────
        // SUMMARY
        // ─────────────────────────────────────────────────
        repeat(20) @(posedge clk);

        $display("");
        $display("=======================================================");
        $display("  TESTBENCH COMPLETE");
        $display("  PASSED: %0d", pass_count);
        $display("  FAILED: %0d", fail_count);
        if (fail_count == 0)
            $display("  STATUS: ✅ ALL TESTS PASSED");
        else
            $display("  STATUS: ❌ %0d TEST(S) FAILED — review above", fail_count);
        $display("=======================================================");
        $display("");

        $finish;
    end

    // ─────────────────────────────────────────────────────
    // Global watchdog: kill simulation if hung
    // ─────────────────────────────────────────────────────
    initial begin
        #50_000_000;  // 50ms wall time maximum
        $display("WATCHDOG: Simulation exceeded 50ms — forcing stop");
        $finish;
    end

    // ─────────────────────────────────────────────────────
    // Signal monitor: print every inference result
    // ─────────────────────────────────────────────────────
    always @(posedge inference_done) begin
        $display("  [%0t ns] inference_done: P(jammed)=0x%04X (%f)  switch=%b  filtered=%b",
                 $time, prob_jammed, q412_to_float(prob_jammed),
                 channel_switch, filtered_switch);
    end

endmodule

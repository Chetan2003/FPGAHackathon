# LSTM Inference Engine — Verilog RTL
## Drone Anti-Jamming System — FPGA Implementation

Target: Xilinx ZedBoard (XC7Z020)
Fixed-point: Q4.12 (16-bit signed, 4 integer + 12 fractional bits)

---

## File Structure

```
verilog/
├── weight_extractor.py     ← Run first — converts .npy weights to .hex
├── sigmoid_lut.v           ← Sigmoid activation (256-entry ROM)
├── tanh_lut.v              ← Tanh activation (256-entry ROM)
├── lstm_cell.v             ← Parameterized LSTM cell (reused for L1 and L2)
├── fc_layer.v              ← Fully connected layer (16→1 + sigmoid)
└── lstm_inference_top.v    ← Top-level: connects everything + rolling window
```

---

## Step 1 — Extract weights

Run this AFTER lstm_train.py has produced lstm_weights.npy:

```bash
cp lstm_weights.npy verilog/
cd verilog/
python weight_extractor.py
```

This generates 10 hex files:

| File               | Contents                         | Size        |
|--------------------|----------------------------------|-------------|
| lstm1_w_ih.hex     | LSTM1 input weights  (128×6)     | 768 values  |
| lstm1_w_hh.hex     | LSTM1 hidden weights (128×32)    | 4096 values |
| lstm1_bias.hex     | LSTM1 combined biases            | 128 values  |
| lstm2_w_ih.hex     | LSTM2 input weights  (64×32)     | 2048 values |
| lstm2_w_hh.hex     | LSTM2 hidden weights (64×16)     | 1024 values |
| lstm2_bias.hex     | LSTM2 combined biases            | 64 values   |
| fc_weight.hex      | FC weights                       | 16 values   |
| fc_bias.hex        | FC bias                          | 1 value     |
| sigmoid_lut.hex    | Sigmoid LUT                      | 256 values  |
| tanh_lut.hex       | Tanh LUT                         | 256 values  |

---

## Step 2 — Simulate in Vivado

Create a new Vivado project targeting XC7Z020:
1. Add all .v files as design sources
2. Add all .hex files as simulation sources (for $readmemh)
3. Set lstm_inference_top.v as top module

Testbench inputs:
- feature_in : 6 × 16-bit Q4.12 normalised features
- new_sample_valid : pulse every 100ms (simulated clock cycles)

---

## Architecture

```
new_sample_valid ──→ Input Buffer (20 × 6 × 16-bit shift register)
                              │
                              ▼ (when buffer full)
                     LSTM Cell 1 (6→32, 20 timesteps)
                       ├─ Serial MAC: W_ih (768) + W_hh (4096)
                       ├─ Sigmoid LUT (i, f, o gates)
                       ├─ Tanh LUT (g gate)
                       └─ Cell/hidden state update
                              │
                              ▼
                     LSTM Cell 2 (32→16, 20 timesteps)
                       └─ Same structure, smaller dimensions
                              │
                              ▼
                     FC Layer (16→1)
                       └─ Dot product + sigmoid → P(jammed)
                              │
                              ▼
                     Threshold (0.4) → channel_switch (raw)
                              │
                              ▼
                     3-step Rolling Window → filtered_switch
```

---

## Fixed-point Q4.12 Reference

| Operation            | Verilog                              |
|----------------------|--------------------------------------|
| Multiply A × B       | product = $signed(A) * $signed(B)    |
| Convert to Q4.12     | result = product >>> 12              |
| Float → Q4.12        | round(float * 4096) & 0xFFFF         |
| Q4.12 → Float        | int16_value / 4096.0                 |
| LUT address from acc | acc[14:7] after >> 12                |

---

## Decision threshold

THRESHOLD = 0x0666 = 1638 = 1638/4096 = 0.3999 ≈ 0.40 in Q4.12

From the trained model:
- Clean predictions cluster at P ≈ 0.05–0.15
- Jammed predictions cluster at P ≈ 0.75–0.80
- Gap between clusters → 0.40 is optimal threshold

filtered_switch requires 3 consecutive raw switch decisions = 1
This reduces false alarms from 40% to under 10% in practice.

---

## Resource estimates (XC7Z020)

| Resource   | Estimated usage     | XC7Z020 available |
|------------|---------------------|-------------------|
| DSP48E1    | 2–4 (serial MAC)    | 220               |
| BRAM       | 4–6 (weight ROMs)   | 140               |
| LUT        | ~800–1200           | 53,200            |
| FF         | ~400–600            | 106,400           |

Serial MAC design uses minimal DSP blocks — well within budget.

---

## Timing

At 100MHz system clock, worst-case inference latency:

| Stage      | MACs              | Clock cycles |
|------------|-------------------|--------------|
| LSTM1 × 20 | 20 × 4992         | ~99,840      |
| LSTM2 × 20 | 20 × 3136         | ~62,720      |
| FC         | 16 + overhead     | ~50          |
| **Total**  |                   | **~162,610** |

162,610 cycles @ 100MHz = **1.63ms latency**
New sample every 100ms → **1.63% utilization**
Inference completes well before next sample arrives.

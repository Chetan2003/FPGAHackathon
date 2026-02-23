# =============================================================
# Weight Extractor for LSTM FPGA Implementation
# =============================================================
# Converts trained PyTorch weights (lstm_weights.npy) to
# Q4.12 fixed-point hex files for Verilog $readmemh ROMs.
#
# Also generates sigmoid and tanh LUT hex files.
#
# Q4.12 format:
#   16-bit signed, 4 integer bits + 12 fractional bits
#   Range    : -8.0 to +7.999755859375
#   Precision: 2^-12 = 0.000244
#   Convert  : int16 = round(float * 4096) & 0xFFFF
#
# Run AFTER lstm_train.py has produced lstm_weights.npy
#
# Usage:
#   python weight_extractor.py
#
# Outputs (place in same folder as Verilog files):
#   lstm1_w_ih.hex   (768  values: 128 x 6)
#   lstm1_w_hh.hex   (4096 values: 128 x 32)
#   lstm1_bias.hex   (128  values)
#   lstm2_w_ih.hex   (2048 values: 64 x 32)
#   lstm2_w_hh.hex   (1024 values: 64 x 16)
#   lstm2_bias.hex   (64   values)
#   fc_weight.hex    (16   values)
#   fc_bias.hex      (1    value)
#   sigmoid_lut.hex  (256  values)
#   tanh_lut.hex     (256  values)
# =============================================================

import numpy as np

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
WEIGHTS_FILE = "lstm_weights.npy"
FRAC_BITS    = 12
SCALE        = 2 ** FRAC_BITS   # 4096
Q_MAX        =  8.0 - (1.0 / SCALE)   # +7.999755...
Q_MIN        = -8.0

# ─────────────────────────────────────────────
# Fixed-point conversion
# ─────────────────────────────────────────────
def to_q4_12(x):
    """Convert float scalar to Q4.12 unsigned hex (2's complement)."""
    x_clipped = float(np.clip(x, Q_MIN, Q_MAX))
    int_val   = int(np.round(x_clipped * SCALE))
    # Mask to 16 bits (handles negative via 2's complement)
    return int_val & 0xFFFF

def write_hex(filename, data, description=""):
    """Write 1D array of floats as Q4.12 hex, one value per line."""
    flat = np.array(data).flatten()
    with open(filename, 'w') as f:
        for val in flat:
            f.write(f'{to_q4_12(val):04x}\n')
    print(f"  {filename:<25} {len(flat):>6} values  {description}")

# ─────────────────────────────────────────────
# Load weights
# ─────────────────────────────────────────────
print("=" * 55)
print("Loading weights from", WEIGHTS_FILE)
print("=" * 55)

weights = np.load(WEIGHTS_FILE, allow_pickle=True).item()

print("\nWeight shapes:")
for k, v in weights.items():
    print(f"  {k:<35} {str(v.shape):<15} "
          f"range [{v.min():.4f}, {v.max():.4f}]")

# ─────────────────────────────────────────────
# Sanity check: are weights within Q4.12 range?
# ─────────────────────────────────────────────
all_vals = np.concatenate([v.flatten() for v in weights.values()])
clipped  = np.sum(np.abs(all_vals) > 7.5)
if clipped > 0:
    print(f"\nWARNING: {clipped} weights outside safe Q4.12 range "
          f"(will be clipped to ±7.999)")
else:
    print(f"\n✓ All weights within Q4.12 range")

# ─────────────────────────────────────────────
# LSTM1 weights
# PyTorch gate order: input(i), forget(f), cell(g), output(o)
# weight_ih shape: (4*H, INPUT_SIZE) = (128, 6)
# weight_hh shape: (4*H, H)          = (128, 32)
# bias combined  : bias_ih + bias_hh  = (128,)
# ─────────────────────────────────────────────
print("\nWriting LSTM1 weight files:")
write_hex("lstm1_w_ih.hex",
          weights['lstm1.weight_ih_l0'],
          "(128x6  input→gate weights)")

write_hex("lstm1_w_hh.hex",
          weights['lstm1.weight_hh_l0'],
          "(128x32 hidden→gate weights)")

# PyTorch adds both biases — combine them
lstm1_bias = (weights['lstm1.bias_ih_l0'] +
              weights['lstm1.bias_hh_l0'])
write_hex("lstm1_bias.hex", lstm1_bias,
          "(128    gate biases)")

# ─────────────────────────────────────────────
# LSTM2 weights
# weight_ih shape: (4*H2, H1) = (64, 32)
# weight_hh shape: (4*H2, H2) = (64, 16)
# bias combined  :              (64,)
# ─────────────────────────────────────────────
print("\nWriting LSTM2 weight files:")
write_hex("lstm2_w_ih.hex",
          weights['lstm2.weight_ih_l0'],
          "(64x32  input→gate weights)")

write_hex("lstm2_w_hh.hex",
          weights['lstm2.weight_hh_l0'],
          "(64x16  hidden→gate weights)")

lstm2_bias = (weights['lstm2.bias_ih_l0'] +
              weights['lstm2.bias_hh_l0'])
write_hex("lstm2_bias.hex", lstm2_bias,
          "(64     gate biases)")

# ─────────────────────────────────────────────
# FC layer
# weight shape: (1, 16)
# bias shape  : (1,)
# ─────────────────────────────────────────────
print("\nWriting FC layer files:")
write_hex("fc_weight.hex", weights['fc.weight'],
          "(16     output weights)")
write_hex("fc_bias.hex",   weights['fc.bias'],
          "(1      output bias)")

# ─────────────────────────────────────────────
# Activation LUTs
#
# 256 entries, index maps to input range [-8, +7.9375]
# index 0   → input = -8.0
# index 128 → input =  0.0
# index 255 → input = +7.9375
#
# Input mapping: val = (index - 128) / 16.0
# This covers sigmoid/tanh saturation regions well
# ─────────────────────────────────────────────
print("\nGenerating activation LUTs:")

indices   = np.arange(256)
lut_input = (indices - 128) / 16.0   # maps index to float input

sigmoid_vals = 1.0 / (1.0 + np.exp(-lut_input))
tanh_vals    = np.tanh(lut_input)

write_hex("sigmoid_lut.hex", sigmoid_vals,
          "(256 entries, input range [-8, +8))")
write_hex("tanh_lut.hex",    tanh_vals,
          "(256 entries, input range [-8, +8))")

# ─────────────────────────────────────────────
# Verification: decode a few LUT entries
# ─────────────────────────────────────────────
print("\nLUT verification:")
for idx in [0, 64, 128, 192, 255]:
    inp = (idx - 128) / 16.0
    s   = sigmoid_vals[idx]
    t   = tanh_vals[idx]
    print(f"  index={idx:3d}  input={inp:+.4f}  "
          f"sigmoid={s:.4f}→0x{to_q4_12(s):04x}  "
          f"tanh={t:+.4f}→0x{to_q4_12(t):04x}")

# ─────────────────────────────────────────────
# Q4.12 arithmetic verification
# ─────────────────────────────────────────────
print("\nQ4.12 round-trip verification:")
test_vals = [-7.5, -1.0, -0.5, 0.0, 0.5, 1.0, 3.14, 7.5]
for v in test_vals:
    encoded  = to_q4_12(v)
    # Decode: treat as signed 16-bit
    signed   = encoded if encoded < 32768 else encoded - 65536
    decoded  = signed / SCALE
    print(f"  {v:+.4f} → 0x{encoded:04x} → {decoded:+.4f}  "
          f"(error={abs(v-decoded):.6f})")

print("\n" + "=" * 55)
print("DONE. Files ready for Verilog $readmemh:")
print("  Place all .hex files in same directory as .v files")
print("=" * 55)

# =============================================================
# COLUMN-MAJOR HEX FILES for lstm_cell_parallel.v
# =============================================================
# Each address in the ROM = one column of the weight matrix
# Word width = N_NEURONS × 16 bits
# This allows reading all N_NEURONS weights for feature k
# in a single clock cycle (broadcast architecture)
# =============================================================

print("\n" + "=" * 55)
print("Generating column-major ROMs for parallel architecture")
print("=" * 55)

def write_colmajor_hex(filename, matrix, description=""):
    """
    Write weight matrix in column-major format.
    matrix shape: (N_NEURONS, N_COLS) — standard row-major PyTorch layout
    Output: N_COLS addresses, each = N_NEURONS packed 16-bit values
    For each column k: write [w[0][k], w[1][k], ..., w[N-1][k]] packed
    Each line in hex file = one 16-bit value (same as row-major)
    but values are ordered column by column instead of row by row.
    """
    n_rows, n_cols = matrix.shape
    with open(filename, 'w') as f:
        for col in range(n_cols):
            for row in range(n_rows):
                f.write(f'{to_q4_12(matrix[row, col]):04x}\n')
    total = n_rows * n_cols
    print(f"  {filename:<30} {n_cols} columns × {n_rows} neurons "
          f"= {total} values  {description}")

# LSTM1 column-major
print("\nLSTM1 (N_NEURONS=128):")
w_ih1 = weights['lstm1.weight_ih_l0']  # shape (128, 6)
w_hh1 = weights['lstm1.weight_hh_l0']  # shape (128, 32)
write_colmajor_hex("lstm1_w_ih_col.hex", w_ih1,
                   "(128 neurons × 6 features — broadcast arch)")
write_colmajor_hex("lstm1_w_hh_col.hex", w_hh1,
                   "(128 neurons × 32 hidden — broadcast arch)")

# LSTM2 column-major
print("\nLSTM2 (N_NEURONS=64):")
w_ih2 = weights['lstm2.weight_ih_l0']  # shape (64, 32)
w_hh2 = weights['lstm2.weight_hh_l0']  # shape (64, 16)
write_colmajor_hex("lstm2_w_ih_col.hex", w_ih2,
                   "(64 neurons × 32 features — broadcast arch)")
write_colmajor_hex("lstm2_w_hh_col.hex", w_hh2,
                   "(64 neurons × 16 hidden — broadcast arch)")

print("\n" + "=" * 55)
print("All hex files generated.")
print("\nFor lstm_cell_parallel.v, use the _col.hex files.")
print("For lstm_cell.v (serial), use the original .hex files.")
print("=" * 55)

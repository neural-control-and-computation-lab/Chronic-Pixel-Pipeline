"""Extract and plot IMU data from an IMU-enabled recording.

Usage (from project root):
    python examples/plot_imu.py Data/testgyro_segment1.dat
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import chronic_pixel_pipeline as cpp

# ── Parse args ──────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python examples/plot_imu.py <dat_file> [output_dir]")
    sys.exit(1)

dat_file = Path(sys.argv[1])
output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)

# ── Extract IMU ─────────────────────────────────────────────────────────────
print(f"Extracting IMU from {dat_file}...")
imu = cpp.load_imu(dat_file)
fs = imu["sampling_rate"]

print(f"  Sampling rate: {fs:.0f} Hz")
for key in ("accel", "gyro", "mag"):
    d = imu[key]
    print(f"  {key:5s}: {d.shape[0]} samples, range [{d.min()}, {d.max()}]")

# ── Plot ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
sensors = [("accel", "Accelerometer"), ("gyro", "Gyroscope"), ("mag", "Magnetometer")]
colors = ["#e74c3c", "#2ecc71", "#3498db"]

for ax, (key, title) in zip(axes, sensors):
    data = imu[key]
    t = np.arange(data.shape[0]) / fs
    for i, (axis_label, c) in enumerate(zip(["X", "Y", "Z"], colors)):
        ax.plot(t, data[:, i], color=c, alpha=0.8, linewidth=0.5, label=axis_label)
    ax.set_ylabel(title)
    ax.legend(loc="upper right", ncol=3)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (s)")
axes[0].set_title(f"IMU Data \u2014 {dat_file.name}  ({fs:.0f} Hz, {imu['accel'].shape[0]} samples)")
plt.tight_layout()

out_path = output_dir / f"{dat_file.stem}_imu.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")

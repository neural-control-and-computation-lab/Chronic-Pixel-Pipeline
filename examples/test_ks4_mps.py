"""Test Kilosort 4 on MPS (Apple Silicon GPU).

Usage (from project root):
    python examples/test_ks4_mps.py
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
import numpy as np
from pathlib import Path

# Check MPS availability
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
    print("MPS not available, exiting.")
    exit(1)

device = torch.device('mps')
print(f"Using device: {device}")

# Quick MPS sanity checks
print("\n--- MPS Sanity Checks ---")
try:
    x = torch.randn(10, 10, device=device)
    y = torch.randn(10, 10, device=device)
    z = x @ y
    print(f"Matrix multiply: OK ({z.shape})")
except Exception as e:
    print(f"Matrix multiply FAILED: {e}")

try:
    A = torch.randn(5, 5, device=device)
    inv = torch.linalg.inv(A)
    print(f"Matrix inverse: OK ({inv.shape})")
except Exception as e:
    print(f"Matrix inverse FAILED: {e}")

try:
    from torch.nn.functional import conv1d
    x = torch.randn(1, 1, 100, device=device)
    w = torch.randn(1, 1, 10, device=device)
    out = conv1d(x, w, padding=5)
    print(f"conv1d: OK ({out.shape})")
except Exception as e:
    print(f"conv1d FAILED: {e}")

try:
    x = torch.randn(10, 10, device=device)
    u, s, v = torch.svd(x)
    print(f"SVD: OK ({u.shape}, {s.shape}, {v.shape})")
except Exception as e:
    print(f"SVD FAILED: {e}")

# Now run KS4
print("\n--- Running KS4 on MPS ---")

import spikeinterface.full as si
import chronic_pixel_pipeline as cpp

data_dir = Path("Data")
rec = cpp.load_segment(sorted(data_dir.glob("*.dat"))[0])
print(f"Recording: {rec.get_num_channels()} channels, {rec.get_num_samples()} samples")

try:
    import kilosort
    print(f"Kilosort version: {kilosort.__version__}")

    sorting = si.run_sorter(
        'kilosort4',
        rec,
        folder='test_output_mps',
        remove_existing_folder=True,
        torch_device='mps',
        batch_size=30000,
        nblocks=0,
    )

    print(f"\nSorting complete!")
    print(f"Units found: {len(sorting.get_unit_ids())}")
    print(f"Total spikes: {sorting.count_num_spikes_per_unit().values()}")

except Exception as e:
    import traceback
    print(f"\nERROR: {e}")
    traceback.print_exc()

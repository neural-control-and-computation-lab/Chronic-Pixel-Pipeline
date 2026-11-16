#!/usr/bin/env python3
"""Unit tests for general electrophysiology toolkit modules (drift, waveforms, ccg, lfp, kinematics)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronic_pixel_pipeline.ccg import compute_jitter_corrected_ccgs
from chronic_pixel_pipeline.drift import extract_bank_spike_cloud, load_ks4_drift_trace
from chronic_pixel_pipeline.kinematics import (
    compute_movement_onset_peths,
    compute_neuron_dropping_curve,
    compute_neuron_kinematic_lag_tuning,
    detect_spontaneous_movement_bouts,
)
from chronic_pixel_pipeline.lfp import compute_bipolar_depth_psd_map
from chronic_pixel_pipeline.waveforms import (
    compute_spike_train_biophysics,
    compute_waveform_celltype_metrics,
    extract_2col_spatial_footprint,
)


class TestDriftAndWaveforms(unittest.TestCase):
    def test_ks4_drift_and_coordinate_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bdir = Path(tmp) / "bank03_ks4"
            sdir = bdir / "sorter_output"
            sdir.mkdir(parents=True)
            np.save(sdir / "ops.npy", {"dshift": np.linspace(-2.0, 4.0, 20)[:, None], "batch_size": 30000})
            np.save(sdir / "spike_times.npy", np.arange(100, 300100, 300, dtype=np.int64))
            np.save(sdir / "spike_clusters.npy", np.zeros(1000, dtype=np.int64))
            np.save(sdir / "amplitudes.npy", np.full(1000, 15.0, dtype=np.float32))
            # Already globally offset in bank 3 (11520..15360 um)
            pos = np.column_stack([np.full(1000, 11.0), np.linspace(11600.0, 15200.0, 1000)])
            np.save(sdir / "spike_positions.npy", pos.astype(np.float32))

            tr = load_ks4_drift_trace(bdir)
            self.assertAlmostEqual(tr["drift_ptp_um"], 6.0, places=5)
            cloud = extract_bank_spike_cloud(bdir, bank_index=3, p_neural_min=None)
            # Must NOT double-add 3 * 3840 = 11520
            self.assertTrue(11520.0 <= float(np.min(cloud["depth_um"])) < float(np.max(cloud["depth_um"])) <= 15360.0)

    def test_footprint_celltype_and_biophysics(self) -> None:
        n_samp, n_ch = 90, 20
        t_ms = (np.arange(n_samp) - 30) / 30.0
        pos = np.column_stack([np.tile([11.0, 114.0], 10), np.repeat(np.arange(10) * 20.0, 2)])

        # Narrow spike (~0.27 ms trough-to-peak) and broad spike (~0.70 ms trough-to-peak)
        wf_narrow = -80.0 * np.exp(-0.5 * (t_ms / 0.12) ** 2) + 25.0 * np.exp(-0.5 * ((t_ms - 0.27) / 0.12) ** 2)
        wf_broad = -80.0 * np.exp(-0.5 * (t_ms / 0.22) ** 2) + 25.0 * np.exp(-0.5 * ((t_ms - 0.70) / 0.24) ** 2)
        tmpls = np.zeros((2, n_samp, n_ch), dtype=np.float64)
        tmpls[0, :, 10] = wf_narrow
        tmpls[0, :, 8] = 0.4 * wf_narrow
        tmpls[1, :, 11] = wf_broad
        tmpls[1, :, 9] = 0.5 * wf_broad

        fp = extract_2col_spatial_footprint(tmpls, pos, unit_index=0, n_rows=5)
        self.assertGreaterEqual(fp["waveforms_uv"].shape[1], 8)
        self.assertGreater(fp["ptp_uv"], 90.0)

        cls_df = compute_waveform_celltype_metrics(tmpls, channel_positions_um=pos)
        self.assertEqual(cls_df.loc[0, "cell_class"], "narrow")
        self.assertEqual(cls_df.loc[1, "cell_class"], "broad")

        # Bursty spike train: pairs separated by 4 ms every 100 ms
        burst_t = []
        for s in np.arange(0.1, 10.0, 0.1):
            burst_t.extend([s, s + 0.004])
        bio = compute_spike_train_biophysics(np.asarray(burst_t) * 30000.0, fs_hz=30000.0, valid_duration_s=10.0)
        self.assertGreater(bio["burst_index_10ms"], 0.45)
        self.assertAlmostEqual(bio["firing_rate_hz"], len(burst_t) / 10.0, places=3)


class TestCCGAndLFPAndKinematics(unittest.TestCase):
    def test_jitter_corrected_ccg_detects_monosynaptic_and_rejects_slow_comodulation(self) -> None:
        rng = np.random.default_rng(7)
        dur_s = 120.0
        # Slow co-modulated background on neurons 0 and 1 (shared sinusoidal envelope) -> should NOT pass 25ms jitter
        t_grid = np.arange(0.0, dur_s, 0.001)
        rate_slow = 8.0 + 7.5 * np.sin(2.0 * np.pi * 0.8 * t_grid)
        sp0 = t_grid[rng.random(t_grid.size) < rate_slow * 0.001]
        sp1 = t_grid[rng.random(t_grid.size) < rate_slow * 0.001]
        # Neuron 2 gets causal +2 ms monosynaptic spikes driven by sp0 (with ~0.25 probability) + tonic baseline
        keep_d = sp0[rng.random(sp0.size) < 0.25]
        sp2_driven = keep_d + 0.002 + rng.normal(0.0, 0.00025, size=keep_d.size)
        sp2_bg = np.sort(rng.uniform(0.0, dur_s, size=600))
        sp2 = np.sort(np.concatenate([sp2_bg, sp2_driven[(sp2_driven > 0) & (sp2_driven < dur_s)]]))

        res = compute_jitter_corrected_ccgs([sp0, sp1, sp2], valid_duration_s=dur_s)
        df = res["pairs_df"]
        self.assertFalse(df.empty)
        pair_02 = df[(df["unit_idx_i"] == 0) & (df["unit_idx_j"] == 2)]
        self.assertEqual(len(pair_02), 1)
        self.assertEqual(pair_02.iloc[0]["connection_type"], "monosynaptic_exc")
        self.assertAlmostEqual(float(pair_02.iloc[0]["peak_lag_ms"]), 2.0, delta=1.0)
        pair_01 = df[(df["unit_idx_i"] == 0) & (df["unit_idx_j"] == 1)]
        self.assertEqual(len(pair_01), 0)

    def test_bipolar_psd_depth_map_detects_local_beta_peak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lf_path = Path(tmp) / "test.imec0.lf.bin"
            fs = 2500.0
            n_samp = int(20 * fs)
            t = np.arange(n_samp) / fs
            # 385 channels int16, 2.34375 uV/bit -> global common noise + localized 20 Hz oscillation on bottom rows
            raw = rng_arr = np.random.default_rng(11).normal(0, 4.0, size=(n_samp, 385))
            common_60 = 15.0 * np.sin(2.0 * np.pi * 60.0 * t)
            raw[:, :384] += common_60[:, None]
            # Depth gradient in 20 Hz beta on channels 40..80 even vs odd difference
            beta_sig = 25.0 * np.sin(2.0 * np.pi * 20.0 * t)
            for c in range(42, 82, 2):
                raw[:, c] += beta_sig
            arr_i16 = np.clip(np.round(raw), -512, 511).astype("<i2")
            arr_i16.tofile(lf_path)

            out = compute_bipolar_depth_psd_map(lf_path, 0, n_samp, fs_hz=fs, ptp_min_uv=0.1)
            self.assertEqual(out["psd_depth_db"].shape[0], 191)
            self.assertTrue(out["qc_pass"])

    def test_movement_bouts_peth_lag_and_dropping(self) -> None:
        dt = 0.05
        t = np.arange(0.0, 80.0, dt)
        bids = np.zeros_like(t, dtype=np.int64)
        acc = np.full_like(t, 0.01)
        onset_gt = [12.0, 28.0, 44.0, 60.0]
        for og in onset_gt:
            m = (t >= og) & (t < og + 2.5)
            acc[m] = 0.25

        bouts = detect_spontaneous_movement_bouts(t, acc, bids)
        self.assertEqual(len(bouts["onset_times_s"]), 4)

        rng = np.random.default_rng(9)
        n_units = 24
        rate_mat = rng.normal(5.0, 1.0, size=(t.size, n_units))
        # First 12 units strongly driven around movement bouts + slight lead (-150 ms)
        shifted_acc = np.roll(acc, -3)
        for u in range(12):
            rate_mat[:, u] += 25.0 * shifted_acc + rng.normal(0.0, 0.5, size=t.size)

        lag_df = compute_neuron_kinematic_lag_tuning(rate_mat, acc, bids, bin_dt_s=dt)
        self.assertGreater(float(lag_df.loc[:11, "peak_r"].mean()), 0.5)
        self.assertLess(float(lag_df.loc[:11, "best_lag_ms"].median()), 0.0)

        drop_df = compute_neuron_dropping_curve(rate_mat, acc, bids, counts=[2, 8, 24], n_draws=6)
        self.assertGreater(float(drop_df.iloc[-1]["mean_r2"]), float(drop_df.iloc[0]["mean_r2"]))

        spikes = [np.sort(np.concatenate([og + rng.uniform(0.0, 1.0, size=15) for og in onset_gt])) for _ in range(4)]
        peth = compute_movement_onset_peths(spikes, bouts["onset_times_s"])
        self.assertGreater(float(np.max(peth["peth_hz"])), 10.0)


if __name__ == "__main__":
    unittest.main()

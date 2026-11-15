#!/usr/bin/env python3
"""Unit and live-header integration tests for chronic_pixel_pipeline and nhp_chronic_np."""

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronic_pixel_pipeline.discovery import discover_surveys, discover_wireless_sessions
from chronic_pixel_pipeline.packet import (
    ACCEL_LSB_PER_G,
    HEADER_SIZE,
    MUX_ACCEL,
    MUX_GYRO,
    MUX_NONE,
    MUX_RFSYNC,
    SYNC_BYTE,
    detect_format,
    extract_imu,
    find_continuity_blocks,
    is_usable_segment,
)
from chronic_pixel_pipeline.probe_geometry import make_wireless_bank_probe
from nhp_chronic_np.config import PACKET_BYTES, PROJECT_PATHS, WIRELESS_SESSIONS
from nhp_chronic_np.geometry import (
    geometry_from_sns_geom_map,
    np1032_bank_geometry,
    parse_spikeglx_meta,
    spikegadgets_wireless_bank_geometry,
)
from nhp_chronic_np.imu_decoding import decode_imu_with_circular_null
from nhp_chronic_np.packet import extract_imu_from_packets, inspect_dat_headers_bounded
from nhp_chronic_np.tracking import match_units_hungarian


def _make_synthetic_dat(path: Path, n_packets: int = 1200, bank: int = 2, dio: int = 0x24,
                        drop_at: tuple[int, int] = (600, 15)) -> None:
    arr = np.zeros((n_packets, PACKET_BYTES), dtype=np.uint8)
    arr[:, 0] = SYNC_BYTE
    arr[:, 1] = dio
    counters = np.arange(n_packets, dtype=np.uint32)
    counters[drop_at[0]:] += np.uint32(drop_at[1])
    arr[:, 76:80] = counters.view(np.uint8).reshape(-1, 4)

    # Embed distinct non-zero LFP values in bytes 12..75 to prove IMU does not read LFP!
    lfp_junk = np.full((n_packets, 32), 7777, dtype="<i2")
    arr[:, 12:76] = lfp_junk.view(np.uint8).reshape(n_packets, 64)

    for i in range(n_packets):
        if i % 300 == 0:
            arr[i, 2] = MUX_ACCEL
            arr[i, 4:10] = np.frombuffer(struct.pack("<hhh", 0, 0, int(ACCEL_LSB_PER_G)), dtype=np.uint8)
        elif i % 300 == 150:
            arr[i, 2] = MUX_GYRO
            arr[i, 4:10] = np.frombuffer(struct.pack("<hhh", 120, -40, 15), dtype=np.uint8)
        elif i % 600 == 10:
            arr[i, 2] = MUX_RFSYNC
        else:
            arr[i, 2] = MUX_NONE
            arr[i, 3] = bank
            arr[i, 5] = 0xFF
    arr.tofile(path)


class TestPacketAndDiscovery(unittest.TestCase):
    def test_streaming_imu_vs_lfp_isolation_and_continuity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dat = Path(tmp) / "synth.dat"
            _make_synthetic_dat(dat, n_packets=1200, bank=3, dio=0x24)
            bank, n_ch, psize, has_imu = detect_format(dat)
            self.assertEqual((bank, n_ch, psize, has_imu), (3, 384, 848, True))
            self.assertFalse(is_usable_segment(dat, min_packets=5000))

            imu = extract_imu(dat, chunk_packets=500)
            self.assertEqual(imu["bank"], 3)
            self.assertEqual(imu["n_gap_events"], 1)
            self.assertEqual(imu["accel"].shape, (4, 3))
            self.assertNotEqual(int(imu["accel"][0, 0]), 7777)
            np.testing.assert_allclose(imu["accel_g"][0], [0.0, 0.0, 1.0])
            self.assertEqual(imu["gyro"][0].tolist(), [120, -40, 15])

    def test_wireless_channel_map_consistency(self) -> None:
        probe_si = make_wireless_bank_probe(384, "NP1032", bank_number=0)
        geom = spikegadgets_wireless_bank_geometry(bank=0, n_channels=384, global_z=False)
        np.testing.assert_allclose(probe_si.contact_positions, geom.positions_um)

        ref = PROJECT_PATHS.sorting_output / r"April21_wirelessank00_ks4\spikeinterface_recording.json"
        if ref.exists():
            locs = np.array(json.loads(ref.read_text(encoding="utf-8"))["properties"]["location"])
            np.testing.assert_allclose(geom.positions_um, locs)

    def test_discovery_and_live_headers(self) -> None:
        if not PROJECT_PATHS.trodes_root.exists():
            self.skipTest("Raw Trodes data drive not mounted")
        surveys = discover_surveys()
        self.assertGreaterEqual(len(surveys), 41)
        self.assertIn("20261106", surveys)
        self.assertIn("20261112", surveys)

        wls = discover_wireless_sessions()
        for sess in WIRELESS_SESSIONS:
            self.assertIn(sess, wls)
            for df in sorted(wls[sess].glob("*.dat")):
                s = inspect_dat_headers_bounded(df)
                self.assertEqual(s.rem_bytes, 0)
                self.assertTrue(s.sync_0x55_ok)
                if df.name == "April7_wireless_segment7.dat":
                    self.assertFalse(is_usable_segment(df))


class TestTrackingAndImuDecoding(unittest.TestCase):
    def test_hungarian_zero_collisions_and_cross_bank_guard(self) -> None:
        rng = np.random.default_rng(7)
        w_a = rng.normal(size=(10, 40))
        w_b = w_a + 0.01 * rng.normal(size=(10, 40))
        pos = np.column_stack([np.full(10, 11.0), np.linspace(0, 1800, 10)])
        rates = np.full(10, 5.0)
        matches = match_units_hungarian(w_a, pos, rates, w_b, pos, rates, bank_a=0, bank_b=0)
        self.assertEqual(len(matches), 10)
        self.assertEqual(len({m.target_idx for m in matches}), 10)
        with self.assertRaises(ValueError):
            match_units_hungarian(w_a, pos, rates, w_b, pos, rates, bank_a=0, bank_b=1)

    def test_imu_decoding_circular_null(self) -> None:
        rng = np.random.default_rng(11)
        n = 450
        bids = np.repeat(np.arange(3), n // 3)
        lat = np.sin(np.linspace(0, 18 * np.pi, n)) + 0.2 * rng.normal(size=n)
        x = lat[:, None] * np.linspace(0.4, 1.4, 8)[None, :] + 0.2 * rng.normal(size=(n, 8))
        y = 1.5 * lat + 0.1 * rng.normal(size=n)
        res = decode_imu_with_circular_null(x, y, bids, bin_dt_s=0.1, min_shift_s=2.0, n_permutations=40, seed=3)
        self.assertGreater(res.cv_r2, 0.7)
        self.assertLess(res.p_value, 0.05)


if __name__ == "__main__":
    unittest.main()

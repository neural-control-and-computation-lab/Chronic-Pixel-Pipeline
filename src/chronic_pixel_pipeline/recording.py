"""SpikeInterface recording extractor for SpikeGadgets Neuropixels wireless data logger."""

from pathlib import Path

import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment

from .packet import (
    DATA_OFFSET,
    DTYPE,
    HEADER_SIZE,
    SAMPLING_RATE,
    compute_packet_size,
    count_packets,
    detect_format,
)


class SpikeGadgetsNPXRecordingSegment(BaseRecordingSegment):
    """Segment that reads neural data from a single SpikeGadgets .dat file.

    Each packet in the file has an 80-byte header followed by n_channels x int16
    neural data. This segment reads on demand, stripping headers via reshape+slice.
    """

    def __init__(self, file_path, n_channels, packet_size, sampling_frequency):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)
        self._file_path = Path(file_path)
        self._n_channels = n_channels
        self._packet_size = packet_size
        self._num_samples = count_packets(self._file_path, self._packet_size)
        self._file = open(self._file_path, "rb")

    def get_num_samples(self) -> int:
        return self._num_samples

    def get_traces(self, start_frame, end_frame, channel_indices):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self._num_samples

        n_frames = end_frame - start_frame
        if n_frames == 0:
            if channel_indices is None:
                return np.zeros((0, self._n_channels), dtype=DTYPE)
            return np.zeros((0, len(channel_indices)), dtype=DTYPE)

        # Read raw bytes for the requested frame range
        self._file.seek(start_frame * self._packet_size)
        raw = self._file.read(n_frames * self._packet_size)
        raw = np.frombuffer(raw, dtype=np.uint8)

        # Reshape to (n_frames, packet_size), strip headers, view as int16
        raw_2d = raw.reshape(n_frames, self._packet_size)
        data_bytes = raw_2d[:, HEADER_SIZE:]
        traces = np.ascontiguousarray(data_bytes).view(DTYPE).reshape(n_frames, self._n_channels)

        if channel_indices is not None:
            traces = traces[:, channel_indices]

        return traces

    def __del__(self):
        if hasattr(self, "_file") and not self._file.closed:
            self._file.close()


class SpikeGadgetsNPXRecording(BaseRecording):
    """SpikeInterface recording extractor for SpikeGadgets wireless Neuropixels data.

    Reads a single .dat file containing packetized neural data from one probe bank.
    Each packet has an 80-byte header + n_channels x int16 data at 30 kHz.

    Channel IDs are set to electrode IDs on the probe: bank * 384 + channel_index,
    so they map directly to physical electrode positions.

    µV scaling
    ----------
    The Neuropixels 1.0 ASIC has a 10-bit SAR ADC with Vref = 0.6V.
    SpikeGadgets stores the 10-bit codes left-shifted by 6 in int16
    (all raw values are multiples of 64).

    Conversion: µV = raw_int16 × Vref / (2^16 × ap_gain) × 1e6
                   = raw_int16 × 600000 / (65536 × ap_gain)

    At gain 1000: µV/bit = 0.009155 ... wait, let me recalculate:
    µV = (raw / 64) × (1200000 / 1024) / gain
       = raw × 1200000 / (64 × 1024 × gain)
       = raw × 1200000 / (65536 × gain)

    At gain 1000: µV/bit = 1200000 / (65536 × 1000) = 0.018311 µV/bit

    Parameters
    ----------
    file_path : str or Path
        Path to a .dat file.
    probe_part_number : str
        Neuropixels probe part number for geometry. Default "NP1032".
    attach_probe : bool
        If True and probeinterface is installed, attach probe geometry.
    ap_gain : int
        Neuropixels AP band gain setting. Default 1000.
        Common values: 50, 125, 250, 500, 1000, 1500, 2000, 3000.
    """

    # NP1.0 ADC reference: full-scale is 1.2V (±0.6V bipolar), 10-bit → 1024 codes
    # Data is left-shifted by 6 bits in int16, so effective divisor is 64 × 1024 = 65536
    _VREF_UV = 1_200_000.0  # 1.2V in µV (full ADC range)
    _ADC_RANGE = 65536.0     # 2^16 (10-bit left-shifted by 6)

    def __init__(self, file_path, probe_part_number="NP1032", attach_probe=True,
                 ap_gain=1000):
        file_path = Path(file_path)
        bank_number, n_channels, packet_size, _has_imu = detect_format(file_path)

        # Channel IDs = electrode IDs on the probe
        channel_ids = np.arange(
            bank_number * 384,
            bank_number * 384 + n_channels,
        )

        BaseRecording.__init__(
            self,
            sampling_frequency=SAMPLING_RATE,
            channel_ids=channel_ids,
            dtype=DTYPE,
        )

        segment = SpikeGadgetsNPXRecordingSegment(
            file_path, n_channels, packet_size, SAMPLING_RATE
        )
        self.add_recording_segment(segment)

        # ── µV scaling ─────────────────────────────────────────────────
        uv_per_bit = self._VREF_UV / (self._ADC_RANGE * ap_gain)
        self.set_channel_gains(gains=np.full(n_channels, uv_per_bit))
        self.set_channel_offsets(offsets=np.zeros(n_channels))

        # Store for SpikeInterface serialization
        self._kwargs = {
            "file_path": str(file_path),
            "probe_part_number": probe_part_number,
            "attach_probe": attach_probe,
            "ap_gain": ap_gain,
        }

        self.annotate(is_filtered=False)
        self.set_property("bank_number", np.full(n_channels, bank_number))
        self.set_property("electrode_id", channel_ids)

        self._bank_number = bank_number
        self._n_channels = n_channels
        self._ap_gain = ap_gain
        self._uv_per_bit = uv_per_bit

        if attach_probe:
            try:
                from .probe_geometry import make_probe_for_bank

                probe = make_probe_for_bank(
                    bank_number, n_channels, probe_part_number=probe_part_number
                )
                self.set_probe(probe, in_place=True)
            except ImportError:
                pass

    @property
    def bank_number(self):
        return self._bank_number

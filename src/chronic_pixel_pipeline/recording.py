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

    Parameters
    ----------
    file_path : str or Path
        Path to a .dat file.
    probe_part_number : str
        Neuropixels probe part number for geometry. Default "NP1032".
    attach_probe : bool
        If True and probeinterface is installed, attach probe geometry.
    """

    def __init__(self, file_path, probe_part_number="NP1032", attach_probe=True):
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

        # Store for SpikeInterface serialization
        self._kwargs = {
            "file_path": str(file_path),
            "probe_part_number": probe_part_number,
            "attach_probe": attach_probe,
        }

        self.annotate(is_filtered=False)
        self.set_property("bank_number", np.full(n_channels, bank_number))
        self.set_property("electrode_id", channel_ids)

        self._bank_number = bank_number
        self._n_channels = n_channels

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

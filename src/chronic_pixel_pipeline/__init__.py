"""Chronic Pixel Pipeline — SpikeGadgets Neuropixels wireless & survey toolbox."""

__version__ = "0.2.0"

from .discovery import (
    AP_PRIMARY_BANKS,
    configure_temp_dir,
    discover_surveys,
    discover_wireless_sessions,
)
from .io import load_all_segments, load_concatenated, load_imu, load_segment
from .packet import (
    ACCEL_LSB_PER_G,
    GYRO_LSB_PER_DPS,
    count_packets,
    detect_format,
    extract_imu,
    find_continuity_blocks,
    is_usable_segment,
    parse_header,
)
from .probe_geometry import (
    electrode_positions,
    make_full_probe,
    make_probe_for_bank,
    make_probe_from_trodesconf,
    make_wireless_bank_probe,
    parse_trodesconf_channel_map,
)
from .recording import SpikeGadgetsNPXRecording, SpikeGadgetsNPXRecordingSegment
from .preprocessing import preprocess, remove_bad_channels
from .drift import (
    extract_bank_spike_cloud,
    fit_chronic_settling_trajectory,
    load_ks4_drift_trace,
    register_1d_profiles_across_sessions,
)
from .waveforms import (
    compute_spike_train_biophysics,
    compute_waveform_celltype_metrics,
    extract_2col_spatial_footprint,
)
from .ccg import compute_jitter_corrected_ccgs
from .lfp import BAND_DEFINITIONS_HZ, compute_bipolar_depth_psd_map
from .kinematics import (
    compute_movement_onset_peths,
    compute_neuron_dropping_curve,
    compute_neuron_kinematic_lag_tuning,
    detect_spontaneous_movement_bouts,
)
from .plotting import (
    add_scalebar,
    plot_2col_footprint,
    plot_ccg_with_flanks,
    plot_neuron_dropping,
    plot_shank_waveforms,
)

__all__ = [
    "__version__",
    "AP_PRIMARY_BANKS",
    "ACCEL_LSB_PER_G",
    "GYRO_LSB_PER_DPS",
    "BAND_DEFINITIONS_HZ",
    "configure_temp_dir",
    "discover_surveys",
    "discover_wireless_sessions",
    "load_segment",
    "load_all_segments",
    "load_concatenated",
    "load_imu",
    "SpikeGadgetsNPXRecording",
    "SpikeGadgetsNPXRecordingSegment",
    "detect_format",
    "is_usable_segment",
    "find_continuity_blocks",
    "parse_header",
    "count_packets",
    "extract_imu",
    "electrode_positions",
    "make_probe_for_bank",
    "make_wireless_bank_probe",
    "make_probe_from_trodesconf",
    "make_full_probe",
    "parse_trodesconf_channel_map",
    "remove_bad_channels",
    "preprocess",
    "load_ks4_drift_trace",
    "extract_bank_spike_cloud",
    "register_1d_profiles_across_sessions",
    "fit_chronic_settling_trajectory",
    "extract_2col_spatial_footprint",
    "compute_waveform_celltype_metrics",
    "compute_spike_train_biophysics",
    "compute_jitter_corrected_ccgs",
    "compute_bipolar_depth_psd_map",
    "detect_spontaneous_movement_bouts",
    "compute_movement_onset_peths",
    "compute_neuron_kinematic_lag_tuning",
    "compute_neuron_dropping_curve",
    "add_scalebar",
    "plot_2col_footprint",
    "plot_ccg_with_flanks",
    "plot_shank_waveforms",
    "plot_neuron_dropping",
]

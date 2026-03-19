"""Neuropixels NHP long probe geometry for probeinterface.

Supports NP1032 (linear) and NP1030 (staggered) probe variants.
Electrode positions are computed from the probe layout parameters,
matching the probeinterface convention.

The SpikeGadgets wireless logger (.dat files) stores channels in an
ADC-interleaved order defined by the trodesconf, NOT in sequential
electrode order.  The ``spikegadgets_channel_map`` function computes
the correct mapping from .dat channel index to electrode index within
a bank, so that probe geometry is correctly assigned.
"""

import numpy as np

# Probe parameters by part number
PROBE_PARAMS = {
    "NP1032": {
        "description": "Neuropixels 1.0 NHP long linear",
        "cols_per_shank": 2,
        "x_pitch": 103.0,  # um, horizontal distance between columns
        "y_pitch": 20.0,  # um, vertical distance between rows
        "even_x_offset": 11.0,  # um
        "odd_x_offset": 11.0,  # um (same = no stagger)
        "contact_width": 12.0,  # um
        "n_electrodes": 4416,
        "electrodes_per_bank": 384,
        "n_banks": 12,  # 11 full (384 ch) + 1 half (192 ch)
    },
    "NP1030": {
        "description": "Neuropixels 1.0 NHP long staggered",
        "cols_per_shank": 2,
        "x_pitch": 87.0,
        "y_pitch": 20.0,
        "even_x_offset": 27.0,
        "odd_x_offset": 11.0,  # different = stagger
        "contact_width": 12.0,
        "n_electrodes": 4416,
        "electrodes_per_bank": 384,
        "n_banks": 12,
    },
}


def spikegadgets_channel_map(n_channels=384):
    """Map SpikeGadgets .dat channel index to electrode index within a bank.

    The SpikeGadgets wireless logger stores neural data in an ADC-interleaved
    order (8 blocks of 48, each with 2 sub-groups of 12 ADC pairs).  This
    function returns the permutation that maps recording channel indices to
    electrode indices (0–383) within any single bank.

    The mapping is a firmware property and is the same for all banks.

    Parameters
    ----------
    n_channels : int
        Number of channels (384 for a full bank, 192 for the half bank).

    Returns
    -------
    electrode_indices : np.ndarray, shape (n_channels,)
        ``electrode_indices[j]`` is the within-bank electrode index for
        recording channel *j*.  Use ``bank * 384 + electrode_indices``
        for absolute electrode IDs.
    """
    j = np.arange(n_channels)
    b = j // 48
    r = j % 48
    s = r // 24
    r2 = r % 24
    g = r2 // 2
    c = r2 % 2
    # dat channel j → hardware channel (hwChan)
    hw = 383 - 4 * b - 2 * s - 32 * g - c
    # hwChan → (row, column) on the probe
    pair = hw // 2
    col = 1 - (hw % 2)  # even hwChan → right column, odd → left
    row = (pair % 16) * 12 + (pair // 16)
    return row * 2 + col


def electrode_positions(electrode_ids, probe_part_number="NP1032"):
    """Compute (x, y) positions in micrometers for given electrode IDs.

    Parameters
    ----------
    electrode_ids : array-like
        Electrode IDs (0-based, up to 4415).
    probe_part_number : str
        "NP1032" (linear) or "NP1030" (staggered).

    Returns
    -------
    positions : np.ndarray, shape (n_electrodes, 2)
        (x, y) coordinates in micrometers.
    """
    params = PROBE_PARAMS[probe_part_number]
    electrode_ids = np.asarray(electrode_ids)

    cols = params["cols_per_shank"]
    y_idx, x_idx = np.divmod(electrode_ids, cols)

    stagger = np.where(
        y_idx % 2 == 0,
        params["even_x_offset"],
        params["odd_x_offset"],
    )

    x = x_idx * params["x_pitch"] + stagger
    y = y_idx * params["y_pitch"]

    return np.column_stack([x, y])


def make_probe_for_bank(
    bank_number,
    n_channels=384,
    probe_part_number="NP1032",
    apply_channel_map=False,
):
    """Create a probeinterface Probe for a single bank of the NHP long probe.

    Parameters
    ----------
    bank_number : int
        Probe bank (0-11).
    n_channels : int
        Number of channels in this bank (384 or 192 for the last bank).
    probe_part_number : str
        "NP1032" or "NP1030".
    apply_channel_map : bool
        If True, apply the SpikeGadgets ADC-interleaved channel map
        (needed for firmware v9 / magic 0x1A55).  If False (default),
        channels are assumed to be in sequential electrode order
        (firmware v10+ / magic 0x0C55, or SpikeGLX recordings).

    Returns
    -------
    probeinterface.Probe
    """
    from probeinterface import Probe

    params = PROBE_PARAMS[probe_part_number]
    electrode_ids = np.arange(
        bank_number * params["electrodes_per_bank"],
        bank_number * params["electrodes_per_bank"] + n_channels,
    )
    positions = electrode_positions(electrode_ids, probe_part_number)

    if apply_channel_map:
        device_channel_indices = spikegadgets_channel_map(n_channels)
    else:
        device_channel_indices = np.arange(n_channels)

    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(
        positions=positions,
        shapes="square",
        shape_params={"width": params["contact_width"]},
    )
    probe.set_device_channel_indices(device_channel_indices)
    probe.create_auto_shape(probe_type="tip")
    probe.annotate(
        manufacturer="imec",
        probe_part_number=probe_part_number,
        bank_number=bank_number,
    )

    return probe


def parse_trodesconf_channel_map(trodesconf_path):
    """Parse a SpikeGadgets .trodesconf to extract the channel-to-position map.

    The trodesconf defines the ordering of data channels via SpikeNTrode
    entries, each with a ``hwChan`` and ``coord_dv`` / ``coord_ml`` fields.
    The order of entries defines the data channel order in the .rec (and .dat)
    file.

    Parameters
    ----------
    trodesconf_path : str or Path
        Path to a .trodesconf XML file.

    Returns
    -------
    positions : np.ndarray, shape (n_channels, 2)
        (x, y) coordinates in micrometers for each data channel,
        where x is derived from ``coord_ml`` and y from ``coord_dv``.
    hw_chans : np.ndarray, shape (n_channels,)
        Hardware channel numbers in data order.
    """
    import re
    from pathlib import Path

    with open(trodesconf_path) as f:
        text = f.read()

    # Extract hwChan, coord_dv, coord_ml from each SpikeChannel element.
    # Attributes may appear in any order, so we parse each element separately.
    spike_channels = re.findall(r'<SpikeChannel\s+[^>]+/>', text)
    if not spike_channels:
        raise ValueError(f"No SpikeChannel entries found in {trodesconf_path}")

    hw_list, dv_list, ml_list = [], [], []
    for elem in spike_channels:
        hw_m = re.search(r'hwChan="(\d+)"', elem)
        dv_m = re.search(r'coord_dv="([^"]+)"', elem)
        ml_m = re.search(r'coord_ml="([^"]+)"', elem)
        if hw_m and dv_m and ml_m:
            hw_list.append(int(hw_m.group(1)))
            dv_list.append(float(dv_m.group(1)))
            ml_list.append(float(ml_m.group(1)))

    if not hw_list:
        raise ValueError(f"No SpikeChannel entries with hwChan/coord_dv/coord_ml in {trodesconf_path}")

    hw_chans = np.array(hw_list)
    coords_dv = np.array(dv_list)
    coords_ml = np.array(ml_list)

    # Convert to probe coordinates: ml -> x, dv -> y (shifted so min=0)
    x = np.where(coords_ml < 0, 11.0, 114.0)
    y = coords_dv - coords_dv.min()

    return np.column_stack([x, y]), hw_chans


def make_probe_from_trodesconf(trodesconf_path, data_channel_to_hwchan=None,
                               n_channels=384):
    """Create a probeinterface Probe using positions from a .trodesconf file.

    This is the preferred method for SpikeGadgets recordings (both wired
    ``.rec`` and wireless ``.dat``) since the trodesconf defines the
    ground-truth physical positions for each hardware channel.

    The key insight is that different recording formats deliver data in
    different channel orders, so the caller must specify how data channels
    map to hardware channels:

    - ``.rec`` files (Neo): data columns follow chip-interleaved byte order
      (``chanPerChip=2`` → evens first, then odds: ``[0, 2, 4, …, 382,
      1, 3, …, 383]``).  Pass Neo's ``signal_channels['id']`` as
      ``data_channel_to_hwchan``.
    - ``.dat`` files (v10+/v11 firmware): identity mapping — data channel
      ``i`` = hwChan ``i``.  Pass ``None`` or ``np.arange(n_channels)``.
    - ``.dat`` files (v9 firmware): ADC-interleaved order.  Pass the result
      of ``spikegadgets_channel_map(n_channels)``.

    Parameters
    ----------
    trodesconf_path : str or Path
        Path to a ``.trodesconf`` XML file.
    data_channel_to_hwchan : array-like or None
        Mapping from data channel index to hardware channel ID.
        ``data_channel_to_hwchan[i]`` is the hwChan for data column ``i``.
        If None, identity mapping is assumed (data channel i = hwChan i).
    n_channels : int
        Number of neural channels (384 for a full bank, 192 for half).

    Returns
    -------
    probeinterface.Probe
    """
    from probeinterface import Probe

    # Parse trodesconf for hwChan → position lookup
    tc_positions, tc_hw_chans = parse_trodesconf_channel_map(trodesconf_path)
    hw_to_pos = {int(h): pos for h, pos in zip(tc_hw_chans, tc_positions)}

    # Default: identity mapping (data channel i = hwChan i)
    if data_channel_to_hwchan is None:
        data_channel_to_hwchan = np.arange(n_channels)
    else:
        data_channel_to_hwchan = np.asarray(data_channel_to_hwchan)

    # Build ordered positions: position[i] = trodesconf position for data column i
    ordered_positions = np.array(
        [hw_to_pos[int(data_channel_to_hwchan[i])] for i in range(n_channels)]
    )

    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(
        positions=ordered_positions,
        shapes="square",
        shape_params={"width": 12.0},
    )
    probe.set_device_channel_indices(np.arange(n_channels))
    probe.create_auto_shape(probe_type="tip")
    probe.annotate(
        manufacturer="imec",
        probe_part_number="NP1032",
    )

    return probe


def make_full_probe(probe_part_number="NP1032"):
    """Create a probeinterface Probe for all 4,416 electrodes.

    Parameters
    ----------
    probe_part_number : str
        "NP1032" or "NP1030".

    Returns
    -------
    probeinterface.Probe
    """
    from probeinterface import Probe

    params = PROBE_PARAMS[probe_part_number]
    n_electrodes = params["n_electrodes"]
    electrode_ids = np.arange(n_electrodes)
    positions = electrode_positions(electrode_ids, probe_part_number)

    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(
        positions=positions,
        shapes="square",
        shape_params={"width": params["contact_width"]},
    )
    probe.set_device_channel_indices(np.arange(n_electrodes))
    probe.create_auto_shape(probe_type="tip")
    probe.annotate(
        manufacturer="imec",
        probe_part_number=probe_part_number,
    )

    return probe

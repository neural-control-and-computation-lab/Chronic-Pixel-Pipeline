"""Neuropixels NHP long probe geometry for probeinterface.

Supports NP1032 (linear) and NP1030 (staggered) probe variants.
Electrode positions are computed from the probe layout parameters,
matching the probeinterface convention.
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


def make_probe_for_bank(bank_number, n_channels=384, probe_part_number="NP1032"):
    """Create a probeinterface Probe for a single bank of the NHP long probe.

    Parameters
    ----------
    bank_number : int
        Probe bank (0-11).
    n_channels : int
        Number of channels in this bank (384 or 192 for the last bank).
    probe_part_number : str
        "NP1032" or "NP1030".

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

    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(
        positions=positions,
        shapes="square",
        shape_params={"width": params["contact_width"]},
    )
    probe.set_device_channel_indices(np.arange(n_channels))
    probe.create_auto_shape(probe_type="tip")
    probe.annotate(
        manufacturer="imec",
        probe_part_number=probe_part_number,
        bank_number=bank_number,
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

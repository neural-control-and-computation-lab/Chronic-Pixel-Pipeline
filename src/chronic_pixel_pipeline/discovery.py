"""Automatic dataset discovery and temporary directory routing."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

AP_PRIMARY_BANKS = tuple(range(7))
SURVEY_DIR_RE = re.compile(r"^SvyPrb_(\d{4})-(\d{2})-(\d{2})T")
WIRELESS_DIR_RE = re.compile(r".*_wireless$", re.IGNORECASE)


def configure_temp_dir(temp_root: str | Path | None = None) -> Path:
    """Direct TEMP/TMP/TMPDIR off system drive onto high-capacity sorting storage."""
    if temp_root is None:
        data_root = Path(os.environ.get("DATA_ROOT", "E:/"))
        temp_root = data_root / "sorting_output" / "_tmp_ks4"
    tmp_path = Path(temp_root)
    tmp_path.mkdir(parents=True, exist_ok=True)
    for var in ("TEMP", "TMP", "TMPDIR"):
        os.environ[var] = str(tmp_path)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return tmp_path


def default_search_roots() -> list[Path]:
    """Return candidate raw data directories from DATA_ROOT and TRODES_ROOT."""
    roots: list[Path] = []
    for env_key, fallback in (
        ("DATA_ROOT", "E:/"),
        ("TRODES_ROOT", "E:/Dropbox/Lab Files/TRODES test"),
    ):
        val = os.environ.get(env_key, fallback)
        if val:
            p = Path(val)
            if p.exists() and p not in roots:
                roots.append(p)
    return roots


def discover_surveys(
    roots: Iterable[str | Path] | None = None,
    exclude_keys: Iterable[str] = ("20260226",),
) -> dict[str, Path]:
    """Auto-discover SpikeGLX `SvyPrb_*` survey folders sorted chronologically."""
    search_roots = [Path(r) for r in roots] if roots is not None else default_search_roots()
    excl = set(exclude_keys)
    found: dict[str, Path] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            m = SURVEY_DIR_RE.match(entry.name)
            if not m:
                continue
            key = f"{m.group(1)}{m.group(2)}{m.group(3)}"
            if key in excl:
                continue
            # Prefer direct DATA_ROOT entries if mirrored; verify .ap.bin exists
            if any(entry.glob("*.ap.bin")) and key not in found:
                found[key] = entry
    return dict(sorted(found.items()))


def discover_wireless_sessions(roots: Iterable[str | Path] | None = None) -> dict[str, Path]:
    """Auto-discover `*wireless` session directories containing `.dat` segments."""
    search_roots = [Path(r) for r in roots] if roots is not None else default_search_roots()
    found: dict[str, Path] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and WIRELESS_DIR_RE.match(entry.name):
                if any(entry.glob("*.dat")) and entry.name not in found:
                    found[entry.name] = entry
    return dict(sorted(found.items()))

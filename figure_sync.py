"""
Sync figure outputs from the local figure tree to the Dropbox-shared tree.

Local:   E:/figures/sorting/<subdir>/
Dropbox: E:/Dropbox/Lab Files/TRODES test/figures/sorting/<subdir>/

Only PNG and SVG files are copied; files are copied unconditionally
(overwriting any existing Dropbox copy with the same name).

Usage as a module:
    from figure_sync import sync_to_dropbox
    sync_to_dropbox('tracking')
    sync_to_dropbox('drift_concat')

Or from the CLI:
    python figure_sync.py tracking drift_concat
"""
import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_ROOT = Path(os.environ.get('DATA_ROOT', 'E:/'))
LOCAL_FIG_ROOT = DATA_ROOT / 'figures' / 'sorting'
DROPBOX_FIG_ROOT = Path(os.environ.get(
    'DROPBOX_FIG_ROOT',
    'E:/Dropbox/Lab Files/TRODES test/figures/sorting'))


def sync_to_dropbox(subdir, patterns=('*.png', '*.svg'), recursive=False):
    """Copy figures under figures/sorting/<subdir>/ to the Dropbox mirror.

    Parameters
    ----------
    subdir : str
        Subfolder under figures/sorting/ (e.g. 'tracking', 'drift_concat').
    patterns : tuple of str
        Glob patterns to copy. Default is PNG + SVG.
    recursive : bool
        If True, mirror subdirectories beneath <subdir> (e.g. per-survey
        unit figures). Defaults False — only files directly in <subdir>.

    Returns
    -------
    list[Path]
        Paths of files copied (Dropbox-side).
    """
    src_dir = LOCAL_FIG_ROOT / subdir
    dst_dir = DROPBOX_FIG_ROOT / subdir
    if not src_dir.exists():
        logger.warning('Source dir missing, skipping sync: %s', src_dir)
        return []
    if not DROPBOX_FIG_ROOT.exists():
        logger.warning('Dropbox figure root missing — is Dropbox synced? %s',
                       DROPBOX_FIG_ROOT)
        return []
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    if recursive:
        files = []
        for pat in patterns:
            files.extend(src_dir.rglob(pat))
    else:
        files = []
        for pat in patterns:
            files.extend(src_dir.glob(pat))
    for src in files:
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
    logger.info('Synced %d files from %s -> %s',
                 len(copied), src_dir, dst_dir)
    return copied


def main():
    logging.basicConfig(level=logging.INFO,
                         format='%(asctime)s [%(levelname)s] %(message)s',
                         datefmt='%H:%M:%S')
    if len(sys.argv) < 2:
        print('Usage: python figure_sync.py <subdir> [<subdir> ...]')
        print('Example: python figure_sync.py tracking drift_concat')
        sys.exit(1)
    for sub in sys.argv[1:]:
        sync_to_dropbox(sub, recursive='--recursive' in sys.argv)


if __name__ == '__main__':
    main()

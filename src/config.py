"""Central configuration: path and environment (Colab/local) handling.

Data and checkpoints live outside the source tree. Locations are auto-detected
and can be overridden with the environment variables CRYPT_DATA_DIR,
CRYPT_CKPT_DIR and CRYPT_LOG_DIR.
"""

import os
from pathlib import Path


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


IN_COLAB = _in_colab()

# Repository root (this file is src/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Storage root for data and checkpoints: Drive on Colab, repo root locally.
# Override with CRYPT_DRIVE_DIR if the Drive folder has a different name.
if IN_COLAB:
    _drive = Path(os.environ.get(
        "CRYPT_DRIVE_DIR", "/content/drive/MyDrive/cryptanalysis-ai"))
    STORAGE_ROOT = _drive if _drive.exists() else PROJECT_ROOT
else:
    STORAGE_ROOT = PROJECT_ROOT

DATA_DIR       = Path(os.environ.get("CRYPT_DATA_DIR", STORAGE_ROOT / "data"))
CHECKPOINT_DIR = Path(os.environ.get("CRYPT_CKPT_DIR", STORAGE_ROOT / "checkpoints"))
TB_LOG_DIR     = Path(os.environ.get("CRYPT_LOG_DIR",  STORAGE_ROOT / "tb_logs"))

# Turkish alphabet (29 letters, no space). Special tokens are added in the generators.
ALPHABET = "abc\u00e7defg\u011fh\u0131ijklmno\u00f6prs\u015ftu\u00fcvyz"


def data_file(name: str = "final_dataset_shuffled.txt") -> Path:
    return DATA_DIR / name


def model_paths(model_name: str):
    """Return (checkpoint_dir, tb_log_dir) for a model, creating them if needed."""
    ckpt = CHECKPOINT_DIR / model_name
    logs = TB_LOG_DIR / model_name
    ckpt.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    return ckpt, logs


def find_checkpoint(model_name: str, prefer: str = "best"):
    """Pick a .ckpt from a model's checkpoint dir (prefer 'best' or 'last')."""
    ckpt_dir = CHECKPOINT_DIR / model_name
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        return None
    for tag in ([prefer, "last"] if prefer != "last" else ["last", "best"]):
        for p in ckpts:
            if tag in p.name:
                return p
    return ckpts[0]


def describe() -> str:
    return (
        f"IN_COLAB     = {IN_COLAB}\n"
        f"PROJECT_ROOT = {PROJECT_ROOT}\n"
        f"STORAGE_ROOT = {STORAGE_ROOT}\n"
        f"DATA_DIR     = {DATA_DIR}\n"
        f"CHECKPOINT   = {CHECKPOINT_DIR}\n"
        f"TB_LOG_DIR   = {TB_LOG_DIR}"
    )

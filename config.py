"""
config.py
=========
Central configuration for the Spectrum-SLM pipeline.
All paths, hyperparameters, and toggles live here — zero hardcoding elsewhere.

Authors : Anjani, Ashish Joshi, Mayank
Guide   : Dr. Abhinandan S.P.
Dated   : April 2026
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Root paths  (all other paths derived from these two)
# ─────────────────────────────────────────────────────────────────────────────

# Directory that contains this config.py file  (SDR_Data/)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Dataset paths
# ─────────────────────────────────────────────────────────────────────────────

# ── Phase 1 / original dataset ───────────────────────────────────────────────
PHASE1_DATA_DIR = os.path.join(ROOT_DIR, "Secondary_User")

# ── Phase 2 / NEW dataset  (files-20260411T185728Z-3-001) ────────────────────
PHASE2_DATA_DIR = os.path.join(ROOT_DIR, "files-20260411T185728Z-3-001", "files")

# Symbol sub-directories inside PHASE2_DATA_DIR
PHASE2_SYMBOL_DIRS = ["Symbol1", "Symbol2", "Symbol3"]

# Modulation folders present in each Symbol dir
PHASE2_MODULATIONS = ["bpsk", "qpsk", "8psk", "16qam", "dqpsk"]

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint directories
# ─────────────────────────────────────────────────────────────────────────────

CKPT_ROOT   = os.path.join(ROOT_DIR, "checkpoints")
CKPT_PHASE1 = os.path.join(CKPT_ROOT, "phase1")
CKPT_PHASE2 = os.path.join(CKPT_ROOT, "phase2")   # ← New dataset model
CKPT_PHASE3 = os.path.join(CKPT_ROOT, "phase3")

# Legacy checkpoint directory (existing slm_checkpoints/)
LEGACY_CKPT_DIR = os.path.join(ROOT_DIR, "slm_checkpoints")

# ─────────────────────────────────────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────────────────────────────────────

N_BINS          = 176
PATCH_SIZE      = 8
N_PATCHES       = N_BINS // PATCH_SIZE   # 22
D_MODEL         = 128
N_HEAD          = 4
NUM_LAYERS      = 4
DIM_FEEDFORWARD = 512
DROPOUT         = 0.1

# ── Modulation classes ────────────────────────────────────────────────────────

# Phase 1 (original dataset) — 4 classes
MOD_MAP_V1 = {"BPSK": 0, "QPSK": 1, "8PSK": 2, "16QAM": 3}
N_MOD_CLASSES_V1 = 4

# Phase 2 (new dataset) — 5 classes including DQPSK
MOD_MAP_V2 = {"BPSK": 0, "QPSK": 1, "8PSK": 2, "16QAM": 3, "DQPSK": 4}
MOD_MAP_V2_INV = {v: k for k, v in MOD_MAP_V2.items()}
N_MOD_CLASSES_V2 = 5                     # ← Used for Phase 2 new dataset model
MOD_NAMES_V2 = ["BPSK", "QPSK", "8PSK", "16QAM", "DQPSK"]
MOD_COLORS_V2 = ["#58a6ff", "#3fb950", "#f78166", "#ffa657", "#d2a8ff"]

# ─────────────────────────────────────────────────────────────────────────────
# Training hyperparameters  (Phase 2 — new dataset)
# ─────────────────────────────────────────────────────────────────────────────

PHASE2_BATCH_SIZE   = 64
PHASE2_LR           = 1e-4
PHASE2_EPOCHS       = 50
PHASE2_PATIENCE     = 8
PHASE2_VAL_RATIO    = 0.15
PHASE2_TEST_RATIO   = 0.15
PHASE2_MASK_RATIO   = 0.20
PHASE2_RANDOM_STATE = 42
PHASE2_NUM_WORKERS  = 0         # 0 = main process (safe for Windows + Kaggle)
PHASE2_AUGMENT      = True
PHASE2_LEARN_WEIGHTS = True     # Kendall uncertainty weighting

# Multi-task loss weights
LOSS_ALPHA = 1.0    # PU detection (Focal)
LOSS_BETA  = 0.5    # Modulation (CE)
LOSS_GAMMA = 0.3    # SNR (MSE)

# ─────────────────────────────────────────────────────────────────────────────
# Output artefact filenames  (inside CKPT_PHASE2/)
# ─────────────────────────────────────────────────────────────────────────────

CKPT_PHASE2_BEST       = "slm_phase2_new_best.pt"
CKPT_PHASE2_LAST       = "slm_phase2_new_last.pt"
NORMALIZER_FILE        = "normalizer_phase2.pkl"
METRICS_FILE           = "metrics_phase2.json"
PREDICTIONS_FILE       = "predictions_phase2.csv"
HISTORY_FILE           = "training_history_phase2.json"

# ─────────────────────────────────────────────────────────────────────────────
# SNR bins for per-bin evaluation
# ─────────────────────────────────────────────────────────────────────────────

SNR_BINS = [4, 6, 8, 10, 12, 14, 16, 18, 20]


# ─────────────────────────────────────────────────────────────────────────────
# Utility: ensure all checkpoint directories exist
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs():
    """Create checkpoint directories if they don't exist."""
    for d in [CKPT_ROOT, CKPT_PHASE1, CKPT_PHASE2, CKPT_PHASE3]:
        os.makedirs(d, exist_ok=True)


def get_phase2_ckpt_path(filename: str) -> str:
    """Return full path for a Phase 2 checkpoint artefact."""
    ensure_dirs()
    return os.path.join(CKPT_PHASE2, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle-specific overrides
# When running on Kaggle the dataset is mounted at /kaggle/input/<dataset-name>
# ─────────────────────────────────────────────────────────────────────────────

def kaggle_override(kaggle_dataset_path: str):
    """
    Call this at the top of a Kaggle notebook to override data paths.

    Args:
        kaggle_dataset_path: path to the mounted Kaggle dataset,
            e.g. '/kaggle/input/spectrum-slm-new-dataset'
    """
    global PHASE2_DATA_DIR, CKPT_PHASE2
    PHASE2_DATA_DIR = kaggle_dataset_path
    CKPT_PHASE2     = "/kaggle/working/checkpoints/phase2"
    os.makedirs(CKPT_PHASE2, exist_ok=True)
    print(f"[CONFIG] Kaggle mode — data: {PHASE2_DATA_DIR}")
    print(f"[CONFIG] Kaggle mode — ckpt: {CKPT_PHASE2}")


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ensure_dirs()
    print("=== Spectrum-SLM Config ===")
    print(f"  ROOT_DIR        : {ROOT_DIR}")
    print(f"  PHASE1_DATA_DIR : {PHASE1_DATA_DIR}")
    print(f"  PHASE2_DATA_DIR : {PHASE2_DATA_DIR}  (exists={os.path.isdir(PHASE2_DATA_DIR)})")
    print(f"  CKPT_PHASE2     : {CKPT_PHASE2}")
    print(f"  N_MOD_CLASSES_V2: {N_MOD_CLASSES_V2}  {MOD_NAMES_V2}")
    print(f"  N_BINS          : {N_BINS}  PATCH_SIZE={PATCH_SIZE}  N_PATCHES={N_PATCHES}")
    print("  Checkpoint dirs created ✓")

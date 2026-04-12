"""
training/phase2_trainer.py
==========================
Modular Phase 2 trainer for the new SDR dataset.

Wraps finetune_supervised() from spectrum_slm_train.py with:
  - Config-driven paths (config.py)
  - n_mod_classes = 5 (BPSK/QPSK/8PSK/16QAM/DQPSK)
  - Saves normalizer.pkl alongside checkpoint
  - Saves metrics_phase2.json + predictions_phase2.csv
  - Saves full training history as training_history_phase2.json

CLI:
    python training/phase2_trainer.py
    python training/phase2_trainer.py --dry-run
    python training/phase2_trainer.py --epochs 30 --batch-size 128

Authors : Anjani, Ashish Joshi, Mayank
Guide   : Dr. Abhinandan S.P.
Dated   : April 2026
"""

import os
import sys
import json
import argparse
import pickle
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ── Path setup — allow running from any working directory ────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)   # SDR_Data/
sys.path.insert(0, _ROOT)

from config import (
    N_BINS, N_MOD_CLASSES_V2, MOD_NAMES_V2, MOD_MAP_V2,
    PHASE2_BATCH_SIZE, PHASE2_LR, PHASE2_EPOCHS, PHASE2_PATIENCE,
    PHASE2_DATA_DIR, CKPT_PHASE2,
    CKPT_PHASE2_BEST, CKPT_PHASE2_LAST, NORMALIZER_FILE,
    METRICS_FILE, PREDICTIONS_FILE, HISTORY_FILE,
    LOSS_ALPHA, LOSS_BETA, LOSS_GAMMA,
    PHASE2_LEARN_WEIGHTS, PHASE2_NUM_WORKERS,
    ensure_dirs, get_phase2_ckpt_path,
)
from spectrum_slm_model   import SpectrumSLM
from spectrum_slm_train   import (
    get_device, save_checkpoint, load_checkpoint,
    finetune_supervised, evaluate_model,
)
from spectrum_slm_dataset_v2 import (
    build_dataloaders_v2, save_normalizer, load_normalizer,
)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction export
# ─────────────────────────────────────────────────────────────────────────────

def export_predictions(
    model:       SpectrumSLM,
    test_loader: DataLoader,
    save_path:   str,
    device:      torch.device,
) -> None:
    """Run inference on test set and save predictions as CSV."""
    model.eval().to(device)
    rows = []

    with torch.no_grad():
        for batch in test_loader:
            psd, pu_lab, mod_lab, snr_lab = batch
            psd = psd.to(device)
            out = model(psd)

            pu_probs  = torch.softmax(out["pu_logits"],  dim=1).cpu().numpy()
            mod_probs = torch.softmax(out["mod_logits"], dim=1).cpu().numpy()
            snr_pred  = out["snr_pred"].cpu().numpy()

            B = psd.size(0)
            for i in range(B):
                row = {
                    "true_pu":     int(pu_lab[i]),
                    "pred_pu":     int(pu_probs[i, 1] > 0.5),
                    "pu_conf":     round(float(pu_probs[i, 1]), 4),
                    "true_mod":    MOD_NAMES_V2[int(mod_lab[i])] if int(mod_lab[i]) >= 0 else "UNK",
                    "pred_mod":    MOD_NAMES_V2[int(np.argmax(mod_probs[i]))],
                    "mod_conf":    round(float(np.max(mod_probs[i])), 4),
                    "true_snr_db": round(float(snr_lab[i]), 2),
                    "pred_snr_db": round(float(snr_pred[i]), 2),
                }
                # Also store per-class modulation probabilities
                for j, mname in enumerate(MOD_NAMES_V2):
                    row[f"prob_{mname}"] = round(float(mod_probs[i, j]), 4)
                rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"  ✓ Predictions saved → {save_path}  ({len(df):,} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Main trainer
# ─────────────────────────────────────────────────────────────────────────────

def train_phase2(
    data_dir:       str   = PHASE2_DATA_DIR,
    save_dir:       str   = CKPT_PHASE2,
    epochs:         int   = PHASE2_EPOCHS,
    batch_size:     int   = PHASE2_BATCH_SIZE,
    lr:             float = PHASE2_LR,
    patience:       int   = PHASE2_PATIENCE,
    learn_weights:  bool  = PHASE2_LEARN_WEIGHTS,
    resume:         bool  = True,
    dry_run:        bool  = False,
    symbol_dirs:    list  = None,
) -> dict:
    """
    End-to-end Phase 2 training on the new SDR dataset.

    Args:
        data_dir      : path to the new dataset root
        save_dir      : directory where checkpoints + artefacts are saved
        epochs        : maximum training epochs
        batch_size    : mini-batch size
        lr            : peak learning rate
        patience      : early-stopping patience
        learn_weights : Kendall uncertainty weighting for multi-task loss
        resume        : if True, resume from existing best checkpoint
        dry_run       : if True, only run one batch to verify the pipeline
        symbol_dirs   : which Symbol subdirs to load (None = all)

    Returns:
        metrics dict (also saved to metrics_phase2.json)
    """
    device   = get_device()
    ensure_dirs()
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Spectrum-SLM — Phase 2 Trainer (New Dataset)")
    print(f"  Data dir   : {data_dir}")
    print(f"  Save dir   : {save_dir}")
    print(f"  Device     : {device}")
    print(f"  Mod classes: {N_MOD_CLASSES_V2}  {MOD_NAMES_V2}")
    print(f"  Dry run    : {dry_run}")
    print(f"{'='*60}")

    # ── 1. DataLoaders ────────────────────────────────────────────────────────
    norm_path = os.path.join(save_dir, NORMALIZER_FILE)
    train_loader, val_loader, test_loader, normalizer, meta = build_dataloaders_v2(
        data_dir            = data_dir,
        batch_size          = batch_size,
        num_workers         = PHASE2_NUM_WORKERS,
        normalizer_save_path = norm_path,
        symbol_dirs         = symbol_dirs,
    )

    if dry_run:
        print("\n  [DRY RUN] Loading one batch to verify pipeline...")
        batch = next(iter(train_loader))
        psd_b, pu_b, mod_b, snr_b = batch
        print(f"  Batch — PSD:{psd_b.shape}  PU:{pu_b.shape}  "
              f"Mod:{mod_b.shape}  SNR:{snr_b.shape}")
        print(f"  Mod IDs seen : {mod_b.unique().tolist()}")
        print("  ✓ Dry run passed — pipeline OK.\n")
        return {}

    # ── 2. Model ──────────────────────────────────────────────────────────────
    model = SpectrumSLM(
        n_bins          = N_BINS,
        patch_size      = 8,
        d_model         = 128,
        nhead           = 4,
        num_layers      = 4,
        dim_feedforward = 512,
        dropout         = 0.1,
        n_mod_classes   = N_MOD_CLASSES_V2,   # ← 5 classes
    )
    print(f"\n  Model parameters: {model.count_parameters():,}")

    # ── 3. Resume from checkpoint ─────────────────────────────────────────────
    best_ckpt = os.path.join(save_dir, CKPT_PHASE2_BEST)
    if resume and os.path.exists(best_ckpt):
        print(f"\n  [RESUME] Loading checkpoint: {best_ckpt}")
        load_checkpoint(model, best_ckpt, device=device)

    # ── 4. Train ──────────────────────────────────────────────────────────────
    history = finetune_supervised(
        model            = model,
        train_loader     = train_loader,
        val_loader       = val_loader,
        pu_class_weight  = meta["pu_weights"],
        n_epochs         = epochs,
        lr               = lr,
        device           = device,
        save_dir         = save_dir,
        patience         = patience,
        alpha            = LOSS_ALPHA,
        beta             = LOSS_BETA,
        gamma            = LOSS_GAMMA,
        learn_weights    = learn_weights,
    )

    # Rename saved checkpoint to Phase-2-new naming convention
    legacy_best = os.path.join(save_dir, "slm_phase2_best.pt")
    if os.path.exists(legacy_best) and not os.path.exists(best_ckpt):
        os.rename(legacy_best, best_ckpt)
        print(f"  Checkpoint renamed → {best_ckpt}")
    elif os.path.exists(legacy_best):
        # Copy the data across
        import shutil
        shutil.copy2(legacy_best, best_ckpt)

    # Save last epoch too
    last_ckpt = os.path.join(save_dir, CKPT_PHASE2_LAST)
    torch.save(model.state_dict(), last_ckpt)

    # Save training history
    history_path = os.path.join(save_dir, HISTORY_FILE)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  ✓ Training history saved → {history_path}")

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    if os.path.exists(best_ckpt):
        load_checkpoint(model, best_ckpt, device=device)

    metrics = evaluate_model(model, test_loader, device=device)

    # Annotate with dataset metadata
    metrics["dataset"]       = "phase2_new"
    metrics["n_mod_classes"] = N_MOD_CLASSES_V2
    metrics["mod_names"]     = MOD_NAMES_V2
    metrics["n_epochs_run"]  = len(history)
    metrics["n_train"]       = meta["n_train"]
    metrics["n_val"]         = meta["n_val"]
    metrics["n_test"]        = meta["n_test"]

    metrics_path = os.path.join(save_dir, METRICS_FILE)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Metrics saved → {metrics_path}")

    # ── 6. Save predictions ───────────────────────────────────────────────────
    pred_path = os.path.join(save_dir, PREDICTIONS_FILE)
    export_predictions(model, test_loader, pred_path, device)

    print(f"\n{'='*60}")
    print("  Phase 2 Training Complete!")
    print(f"  Best checkpoint : {best_ckpt}")
    print(f"  Normalizer      : {norm_path}")
    print(f"  Metrics JSON    : {metrics_path}")
    print(f"  Predictions CSV : {pred_path}")
    print(f"{'='*60}\n")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spectrum-SLM Phase 2 Trainer — New SDR Dataset (5-class)"
    )
    parser.add_argument("--data-dir",      type=str,   default=PHASE2_DATA_DIR,
                        help="Path to new dataset root directory")
    parser.add_argument("--save-dir",      type=str,   default=CKPT_PHASE2,
                        help="Directory for checkpoints and artefacts")
    parser.add_argument("--epochs",        type=int,   default=PHASE2_EPOCHS)
    parser.add_argument("--batch-size",    type=int,   default=PHASE2_BATCH_SIZE)
    parser.add_argument("--lr",            type=float, default=PHASE2_LR)
    parser.add_argument("--patience",      type=int,   default=PHASE2_PATIENCE)
    parser.add_argument("--no-resume",     action="store_true",
                        help="Start fresh (ignore existing checkpoint)")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Only run one batch — verify pipeline")
    parser.add_argument("--symbol-dirs",   nargs="+",  default=None,
                        help="Symbol subdirs to include (default: all)")

    args = parser.parse_args()

    train_phase2(
        data_dir      = args.data_dir,
        save_dir      = args.save_dir,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        lr            = args.lr,
        patience      = args.patience,
        resume        = not args.no_resume,
        dry_run       = args.dry_run,
        symbol_dirs   = args.symbol_dirs,
    )

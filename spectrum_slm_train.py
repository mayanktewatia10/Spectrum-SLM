"""
spectrum_slm_train.py
=====================
Three-phase training pipeline for Spectrum-SLM.

Phase 1  — Self-supervised Pre-training (Masked Spectrum Modelling)
Phase 2  — Supervised Multi-task Fine-tuning  (PU + Mod + SNR)
Phase 3  — Generative / Autoregressive Next-PSD Prediction

Also includes:
  • evaluate_model()  — full metrics (accuracy, F1, MAE, AUC, per-SNR-bin)
  • export_onnx()     — ONNX export for edge deployment

Authors : Anjani, Ashish Joshi, Mayank
Guide   : Dr. Abhinandan S.P.
Dated   : March 2026
"""

import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, mean_absolute_error
)
from typing import Optional, Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

from spectrum_slm_model  import SpectrumSLM, MultiTaskLoss, MSMLoss, FocalLoss
from spectrum_slm_dataset import (
    build_dataloaders, generate_synthetic_psd,
    SpectrumNormalizer, SpectrumDataset, SpectrumAugmenter,
    N_BINS, N_PATCHES, PATCH_SIZE, SNR_BINS
)


# ─────────────────────────────────────────────────────────────────────────────
# Utility — device / checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    try:
        if torch.backends.mps.is_available():
            return torch.device('mps')
    except AttributeError:
        pass
    return torch.device('cpu')


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        'epoch'     : epoch,
        'model'     : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'val_loss'  : val_loss,
    }, path)
    print(f"  ✓ Checkpoint saved → {path}")


def load_checkpoint(model, path, optimizer=None, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    print(f"  ✓ Loaded checkpoint from {path}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt.get('epoch', 0)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Pre-training (Masked Spectrum Modelling)
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_msm(
    model:      SpectrumSLM,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    n_epochs:   int = 30,
    lr:         float = 3e-4,
    device:     torch.device = None,
    save_dir:   str = '.',
    patience:   int = 5,
) -> List[dict]:
    """
    Phase 1: Masked Spectrum Modelling (self-supervised).

    Randomly masks 15–30% of spectral patches and trains the model to
    reconstruct them. No labels required — uses ALL available data.
    """
    if device is None:
        device = get_device()
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    criterion = MSMLoss()

    best_val_loss = float('inf')
    patience_left = patience
    history = []

    print(f"\n{'='*60}")
    print(f"  PHASE 1 — Masked Spectrum Modelling  ({n_epochs} epochs)")
    print(f"  Device: {device}  |  LR: {lr}")
    print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):
        # ── Training ────────────────────────────────────────────────────────
        model.train()
        t0 = time.time()
        train_losses = []

        for batch in train_loader:
            psd, mask = batch                      # (B, 176), (B, 176) bool
            psd, mask = psd.to(device), mask.to(device)

            # Ground truth patches before masking
            true_patches = psd.view(-1, N_PATCHES, PATCH_SIZE)   # (B, 176, 1)

            optimizer.zero_grad()
            out = model(psd, mask=mask, return_msm=True)
            loss = criterion(out['msm_pred'], true_patches, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                psd, mask = batch
                psd, mask = psd.to(device), mask.to(device)
                true_patches = psd.view(-1, N_PATCHES, PATCH_SIZE)   # (B, 176, 1)
                out = model(psd, mask=mask, return_msm=True)
                loss = criterion(out['msm_pred'], true_patches, mask)
                val_losses.append(loss.item())

        scheduler.step()

        train_l = np.mean(train_losses)
        val_l   = np.mean(val_losses)
        elapsed = time.time() - t0

        history.append({'epoch': epoch, 'train_msm': train_l, 'val_msm': val_l})
        print(f"  Epoch {epoch:3d}/{n_epochs}  "
              f"Train Loss: {train_l:.4f}  Val Loss: {val_l:.4f}  "
              f"({elapsed:.1f}s)")

        if val_l < best_val_loss:
            best_val_loss = val_l
            patience_left = patience
            save_checkpoint(model, optimizer, epoch, val_l,
                            os.path.join(save_dir, 'slm_phase1_best.pt'))
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\n  ✓ Phase 1 complete.  Best Val MSM Loss: {best_val_loss:.4f}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Supervised Multi-task Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def finetune_supervised(
    model:           SpectrumSLM,
    train_loader:    DataLoader,
    val_loader:      DataLoader,
    pu_class_weight: Optional[torch.Tensor] = None,
    n_epochs:        int = 50,
    lr:              float = 1e-4,
    device:          torch.device = None,
    save_dir:        str = '.',
    patience:        int = 8,
    alpha:           float = 1.0,   # PU weight
    beta:            float = 0.5,   # Mod weight
    gamma:           float = 0.3,   # SNR weight
    learn_weights:   bool = True,
) -> List[dict]:
    """
    Phase 2: Supervised Multi-task Fine-tuning.

    Loss = α·FocalLoss(PU) + β·CE(Modulation) + γ·MSE(SNR)
    """
    if device is None:
        device = get_device()
    model = model.to(device)

    if pu_class_weight is not None:
        pu_class_weight = pu_class_weight.to(device)

    criterion = MultiTaskLoss(
        alpha=alpha, beta=beta, gamma=gamma,
        pu_class_weight=pu_class_weight,
        focal_gamma=2.0,
        learn_weights=learn_weights,
    ).to(device)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = OneCycleLR(
        optimizer, max_lr=lr * 10,
        epochs=n_epochs, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='cos',
    )

    best_val_loss = float('inf')
    patience_left = patience
    history = []

    print(f"\n{'='*60}")
    print(f"  PHASE 2 — Supervised Multi-task Fine-tuning  ({n_epochs} epochs)")
    print(f"  Device: {device}  |  LR: {lr}  |  Learn weights: {learn_weights}")
    print(f"  Loss weights (α, β, γ) = ({alpha}, {beta}, {gamma})")
    print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        t0 = time.time()
        tr_total, tr_pu, tr_mod, tr_snr = [], [], [], []

        for psd, pu_lab, mod_lab, snr_lab in train_loader:
            psd, pu_lab = psd.to(device), pu_lab.to(device)
            mod_lab, snr_lab = mod_lab.to(device), snr_lab.to(device)

            # Skip samples with unknown modulation
            valid = mod_lab >= 0
            if valid.sum() == 0:
                continue

            optimizer.zero_grad()
            out = model(psd)

            total_loss, breakdown = criterion(
                out['pu_logits'][valid], pu_lab[valid],
                out['mod_logits'][valid], mod_lab[valid],
                out['snr_pred'][valid], snr_lab[valid],
            )
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            tr_total.append(breakdown['total'])
            tr_pu.append(breakdown['pu'])
            tr_mod.append(breakdown['mod'])
            tr_snr.append(breakdown['snr'])

        # Validation
        model.eval()
        vl_total = []
        with torch.no_grad():
            for psd, pu_lab, mod_lab, snr_lab in val_loader:
                psd, pu_lab = psd.to(device), pu_lab.to(device)
                mod_lab, snr_lab = mod_lab.to(device), snr_lab.to(device)
                valid = mod_lab >= 0
                if valid.sum() == 0:
                    continue
                out = model(psd)
                loss, _ = criterion(
                    out['pu_logits'][valid], pu_lab[valid],
                    out['mod_logits'][valid], mod_lab[valid],
                    out['snr_pred'][valid],  snr_lab[valid],
                )
                vl_total.append(loss.item())

        elapsed = time.time() - t0
        vl_mean = np.mean(vl_total) if vl_total else float('inf')
        history.append({
            'epoch':    epoch,
            'train_total': np.mean(tr_total),
            'train_pu':    np.mean(tr_pu),
            'train_mod':   np.mean(tr_mod),
            'train_snr':   np.mean(tr_snr),
            'val_total':   vl_mean,
        })
        print(f"  Epoch {epoch:3d}/{n_epochs}  "
              f"Train: {np.mean(tr_total):.4f} "
              f"(PU={np.mean(tr_pu):.3f} Mod={np.mean(tr_mod):.3f} SNR={np.mean(tr_snr):.3f})  "
              f"Val: {vl_mean:.4f}  ({elapsed:.1f}s)")

        if vl_mean < best_val_loss:
            best_val_loss = vl_mean
            patience_left = patience
            save_checkpoint(model, optimizer, epoch, vl_mean,
                            os.path.join(save_dir, 'slm_phase2_best.pt'))
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\n  ✓ Phase 2 complete.  Best Val Loss: {best_val_loss:.4f}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Autoregressive Generative Training
# ─────────────────────────────────────────────────────────────────────────────

def train_generative(
    model:        SpectrumSLM,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    n_epochs:     int = 20,
    lr:           float = 5e-5,
    device:       torch.device = None,
    save_dir:     str = '.',
    patience:     int = 5,
) -> List[dict]:
    """
    Phase 3: Autoregressive Next-PSD Prediction.

    Given a sequence of past PSD snapshots, predict the next one.
    Uses the generative head on the CLS token.
    """
    if device is None:
        device = get_device()
    model = model.to(device)

    optimizer = optim.AdamW(
        list(model.gen_head.parameters()) + list(model.encoder.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_left = patience
    history = []

    print(f"\n{'='*60}")
    print(f"  PHASE 3 — Autoregressive Generation  ({n_epochs} epochs)")
    print(f"  Device: {device}  |  LR: {lr}")
    print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        t0 = time.time()
        tr_losses = []

        for seq, target in train_loader:
            # seq: (B, L, 176)  target: (B, 176)
            seq, target = seq.to(device), target.to(device)
            B, L, _ = seq.shape

            # Use last PSD in the sequence as input to the generative head
            last_psd = seq[:, -1, :]           # (B, 176)

            optimizer.zero_grad()
            out = model(last_psd)
            loss = criterion(out['gen_pred'], target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        vl_losses = []
        with torch.no_grad():
            for seq, target in val_loader:
                seq, target = seq.to(device), target.to(device)
                last_psd = seq[:, -1, :]
                out = model(last_psd)
                loss = criterion(out['gen_pred'], target)
                vl_losses.append(loss.item())

        scheduler.step()
        elapsed = time.time() - t0
        vl_mean = np.mean(vl_losses) if vl_losses else float('inf')

        history.append({'epoch': epoch, 'train_gen': np.mean(tr_losses),
                        'val_gen': vl_mean})
        print(f"  Epoch {epoch:3d}/{n_epochs}  "
              f"Train MSE: {np.mean(tr_losses):.4f}  "
              f"Val MSE: {vl_mean:.4f}  ({elapsed:.1f}s)")

        if vl_mean < best_val_loss:
            best_val_loss = vl_mean
            patience_left = patience
            save_checkpoint(model, optimizer, epoch, vl_mean,
                            os.path.join(save_dir, 'slm_phase3_best.pt'))
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\n  ✓ Phase 3 complete.  Best Val Gen MSE: {best_val_loss:.4f}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model:       SpectrumSLM,
    test_loader: DataLoader,
    device:      torch.device = None,
    snr_bins:    Optional[List[float]] = None,
    snr_labels_all: Optional[np.ndarray] = None,
) -> dict:
    """
    Full evaluation: accuracy, F1, AUC, MAE + per-SNR-bin breakdown.

    Returns a metric dict compatible with JSON serialisation.
    """
    if device is None:
        device = get_device()
    model.eval().to(device)
    if snr_bins is None:
        snr_bins = SNR_BINS

    all_pu_pred, all_pu_true  = [], []
    all_mod_pred, all_mod_true = [], []
    all_snr_pred, all_snr_true = [], []
    all_snr_label_float        = []

    with torch.no_grad():
        for batch in test_loader:
            psd, pu_lab, mod_lab, snr_lab = batch
            psd = psd.to(device)

            out = model(psd)

            pu_pred  = out['pu_logits'].argmax(dim=1).cpu().numpy()
            mod_pred = out['mod_logits'].argmax(dim=1).cpu().numpy()
            snr_pred = out['snr_pred'].cpu().numpy()

            all_pu_pred.extend(pu_pred)
            all_pu_true.extend(pu_lab.numpy())
            all_mod_pred.extend(mod_pred)
            all_mod_true.extend(mod_lab.numpy())
            all_snr_pred.extend(snr_pred)
            all_snr_true.extend(snr_lab.numpy())
            all_snr_label_float.extend(snr_lab.numpy())

    all_pu_pred  = np.array(all_pu_pred)
    all_pu_true  = np.array(all_pu_true)
    all_mod_pred = np.array(all_mod_pred)
    all_mod_true = np.array(all_mod_true)
    all_snr_pred = np.array(all_snr_pred)
    all_snr_true = np.array(all_snr_true)
    all_snr_labels = np.array(all_snr_label_float)

    # ── PU Detection ─────────────────────────────────────────────────────
    pu_acc  = accuracy_score(all_pu_true, all_pu_pred)
    pu_f1   = f1_score(all_pu_true, all_pu_pred, average='binary', zero_division=0)
    try:
        pu_auc = roc_auc_score(all_pu_true, all_pu_pred)
    except ValueError:
        pu_auc = float('nan')
    pu_cm = confusion_matrix(all_pu_true, all_pu_pred).tolist()

    # ── Low-SNR PU (<8 dB) ───────────────────────────────────────────────
    low_snr_mask = all_snr_labels < 8.0
    if low_snr_mask.sum() > 0:
        low_snr_acc = accuracy_score(all_pu_true[low_snr_mask],
                                     all_pu_pred[low_snr_mask])
        low_snr_f1  = f1_score(all_pu_true[low_snr_mask],
                                all_pu_pred[low_snr_mask],
                                average='binary', zero_division=0)
    else:
        low_snr_acc, low_snr_f1 = float('nan'), float('nan')

    # ── Modulation Classification ─────────────────────────────────────────
    # Only evaluate on samples with known modulation  (-1 = unknown)
    valid_mod = all_mod_true >= 0
    if valid_mod.sum() > 0:
        mod_acc = accuracy_score(all_mod_true[valid_mod], all_mod_pred[valid_mod])
        mod_f1  = f1_score(all_mod_true[valid_mod], all_mod_pred[valid_mod],
                           average='macro', zero_division=0)
        mod_report = classification_report(
            all_mod_true[valid_mod], all_mod_pred[valid_mod],
            target_names=['BPSK', 'QPSK', '8PSK', '16QAM'],
            output_dict=True, zero_division=0,
        )
    else:
        mod_acc, mod_f1, mod_report = float('nan'), float('nan'), {}

    # ── SNR Estimation ────────────────────────────────────────────────────
    snr_mae  = mean_absolute_error(all_snr_true, all_snr_pred)
    snr_rmse = float(np.sqrt(np.mean((all_snr_pred - all_snr_true) ** 2)))

    # ── Per-SNR-bin PU accuracy ───────────────────────────────────────────
    per_snr_acc = {}
    for sbin in snr_bins:
        mask = (all_snr_labels >= sbin - 1) & (all_snr_labels < sbin + 1)
        if mask.sum() > 5:
            per_snr_acc[str(sbin)] = float(
                accuracy_score(all_pu_true[mask], all_pu_pred[mask])
            )

    metrics = {
        'pu_accuracy'    : float(pu_acc),
        'pu_f1'          : float(pu_f1),
        'pu_auc'         : float(pu_auc),
        'pu_confusion'   : pu_cm,
        'low_snr_pu_acc' : float(low_snr_acc),
        'low_snr_pu_f1'  : float(low_snr_f1),
        'mod_accuracy'   : float(mod_acc),
        'mod_f1_macro'   : float(mod_f1),
        'mod_report'     : mod_report,
        'snr_mae_db'     : float(snr_mae),
        'snr_rmse_db'    : float(snr_rmse),
        'per_snr_bin_pu_acc': per_snr_acc,
        'n_samples'      : len(all_pu_true),
    }

    # ── Pretty print ─────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("  EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  PU Detection  — Acc: {pu_acc*100:.2f}%  "
          f"F1: {pu_f1:.4f}  AUC: {pu_auc:.4f}")
    print(f"  Low-SNR (<8dB)— Acc: {low_snr_acc*100:.2f}%  F1: {low_snr_f1:.4f}")
    print(f"  Modulation    — Acc: {mod_acc*100:.2f}%  F1-macro: {mod_f1:.4f}")
    print(f"  SNR MAE       — {snr_mae:.3f} dB  |  RMSE: {snr_rmse:.3f} dB")
    print(f"  Per-SNR PU accuracy: {per_snr_acc}")
    print(f"{'='*50}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# ONNX Export
# ─────────────────────────────────────────────────────────────────────────────

def export_onnx(
    model:     SpectrumSLM,
    save_path: str = 'spectrum_slm.onnx',
    device:    torch.device = None,
    opset:     int = 17,
) -> None:
    """
    Export the trained Spectrum-SLM to ONNX format for edge deployment.

    Compatible with ONNX Runtime, TensorRT, and CoreML.
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("  [WARN] onnx / onnxruntime not installed. "
              "Run: pip install onnx onnxruntime")
        return

    if device is None:
        device = torch.device('cpu')
    model = model.eval().to(device)

    dummy_psd = torch.randn(1, N_BINS).to(device)

    print(f"\nExporting to ONNX → {save_path}")
    torch.onnx.export(
        model,
        (dummy_psd,),
        save_path,
        opset_version   = opset,
        input_names     = ['psd_input'],
        output_names    = ['pu_logits', 'mod_logits', 'snr_pred',
                           'gen_pred', 'cls_feat'],
        dynamic_axes    = {
            'psd_input' : {0: 'batch_size'},
            'pu_logits' : {0: 'batch_size'},
            'mod_logits': {0: 'batch_size'},
            'snr_pred'  : {0: 'batch_size'},
            'gen_pred'  : {0: 'batch_size'},
            'cls_feat'  : {0: 'batch_size'},
        },
    )

    # Verify the exported model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)

    # Quick inference test
    ort_session = ort.InferenceSession(save_path)
    ort_out = ort_session.run(None, {'psd_input': dummy_psd.numpy()})
    print(f"  ✓ ONNX export verified. Output shapes: "
          f"{[o.shape for o in ort_out]}")

    # Latency benchmark
    import timeit
    n_iters = 100
    elapsed = timeit.timeit(
        lambda: ort_session.run(None, {'psd_input': dummy_psd.numpy()}),
        number=n_iters
    )
    print(f"  ✓ ONNX inference latency: "
          f"{1000 * elapsed / n_iters:.2f} ms / sample")


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper (single PSD vector → predictions)
# ─────────────────────────────────────────────────────────────────────────────

MOD_NAMES = {0: 'BPSK', 1: 'QPSK', 2: '8PSK', 3: '16QAM'}

def predict_single(
    model:      SpectrumSLM,
    psd_vector: np.ndarray,
    normalizer: SpectrumNormalizer,
    device:     torch.device = None,
) -> dict:
    """
    Run inference on a single 176-bin PSD vector.

    Returns a human-readable dict of predictions.
    """
    if device is None:
        device = get_device()
    model = model.eval().to(device)

    psd_norm = normalizer.transform(psd_vector.reshape(1, -1))
    psd_t    = torch.tensor(psd_norm, dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(psd_t)

    pu_prob  = torch.softmax(out['pu_logits'], dim=1)[0, 1].item()
    pu_pred  = int(pu_prob > 0.5)
    mod_prob = torch.softmax(out['mod_logits'], dim=1)[0].cpu().numpy()
    mod_pred = int(np.argmax(mod_prob))
    snr_pred = out['snr_pred'][0].item()

    return {
        'pu_present'       : bool(pu_pred),
        'pu_confidence'    : float(pu_prob),
        'modulation'       : MOD_NAMES.get(mod_pred, 'Unknown'),
        'mod_confidence'   : float(mod_prob[mod_pred]),
        'mod_probabilities': {MOD_NAMES[i]: float(mod_prob[i]) for i in range(4)},
        'snr_estimated_db' : float(snr_pred),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device   = get_device()
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Spectrum-SLM Training Pipeline")
    print(f"  Data dir : {args.data_dir}")
    print(f"  Device   : {device}")
    print(f"  Save dir : {save_dir}")
    print(f"{'='*60}")

    # ── Data ─────────────────────────────────────────────────────────────────
    if args.synthetic:
        print("\n[SYNTHETIC MODE] Generating 10,000 synthetic samples...")
        psds, pu, mod, snr = generate_synthetic_psd(n_samples=10000)
        normalizer = SpectrumNormalizer()
        psds_norm  = normalizer.fit_transform(psds)
        from sklearn.model_selection import train_test_split
        n = len(psds_norm)
        idx = np.arange(n)
        idx_tr, idx_tmp = train_test_split(idx, test_size=0.3, stratify=pu)
        idx_v,  idx_te  = train_test_split(idx_tmp, test_size=0.5, stratify=pu[idx_tmp])

        aug = SpectrumAugmenter()
        tr_ds = SpectrumDataset(psds_norm[idx_tr], pu[idx_tr], mod[idx_tr], snr[idx_tr],
                                phase=1, augmenter=aug)
        vl_ds = SpectrumDataset(psds_norm[idx_v], pu[idx_v], mod[idx_v], snr[idx_v],
                                phase=1)
        te_ds = SpectrumDataset(psds_norm[idx_te], pu[idx_te], mod[idx_te], snr[idx_te],
                                phase=2)

        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
        vl_loader = DataLoader(vl_ds, batch_size=args.batch_size)
        te_loader = DataLoader(te_ds, batch_size=args.batch_size)

        pu_counts  = np.bincount(pu[idx_tr])
        pu_weights = torch.tensor(len(idx_tr) / (2 * np.maximum(pu_counts, 1)),
                                  dtype=torch.float32)
    else:
        tr_loader, vl_loader, te_loader, normalizer, meta = build_dataloaders(
            data_dir   = args.data_dir,
            phase      = 1,         # Phase 1 dataloaders
            batch_size = args.batch_size,
        )
        pu_weights = meta['pu_weights']

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SpectrumSLM(
        n_bins          = N_BINS,
        patch_size      = 1,
        d_model         = 128,
        nhead           = 4,
        num_layers      = 4,
        dim_feedforward = 512,
        dropout         = 0.1,
    )
    print(f"\n  Model parameters: {model.count_parameters():,}")

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if args.phase >= 1:
        ckpt_p1 = os.path.join(save_dir, 'slm_phase1_best.pt')
        if os.path.exists(ckpt_p1):
            print("\n  [INFO] Resuming Phase 1 from existing checkpoint...")
            load_checkpoint(model, ckpt_p1, device=device)
            
        history1 = pretrain_msm(
            model, tr_loader, vl_loader,
            n_epochs=args.epochs_p1, lr=args.lr, device=device,
            save_dir=save_dir, patience=args.patience,
        )

    # ── Phase 2 — rebuild loaders with phase=2 ────────────────────────────────
    if args.phase >= 2:
        # Load best phase-1 weights if available
        ckpt_p1 = os.path.join(save_dir, 'slm_phase1_best.pt')
        if os.path.exists(ckpt_p1):
            load_checkpoint(model, ckpt_p1, device=device)

        if not args.synthetic:
            tr_loader, vl_loader, te_loader, normalizer, meta = build_dataloaders(
                data_dir   = args.data_dir,
                phase      = 2,
                batch_size = args.batch_size,
            )
        else:
            # Rebuild synthetic loaders for phase 2
            for ds in [tr_ds, vl_ds, te_ds]:
                ds.phase = 2

        if args.epochs_p2 > 0:
            history2 = finetune_supervised(
                model, tr_loader, vl_loader,
                pu_class_weight = pu_weights,
                n_epochs        = args.epochs_p2,
                lr              = args.lr / 3,
                device          = device,
                save_dir        = save_dir,
                patience        = args.patience,
            )

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    if args.phase >= 3:
        ckpt_p2 = os.path.join(save_dir, 'slm_phase2_best.pt')
        if os.path.exists(ckpt_p2):
            load_checkpoint(model, ckpt_p2, device=device)

        if not args.synthetic:
            tr_loader_g, vl_loader_g, _, _, _ = build_dataloaders(
                data_dir   = args.data_dir,
                phase      = 3,
                batch_size = args.batch_size,
            )
        else:
            for ds in [tr_ds, vl_ds]:
                ds.phase = 3
            tr_loader_g, vl_loader_g = tr_loader, vl_loader

        history3 = train_generative(
            model, tr_loader_g, vl_loader_g,
            n_epochs=args.epochs_p3, lr=args.lr / 10,
            device=device, save_dir=save_dir, patience=args.patience,
        )

    # ── Evaluation ────────────────────────────────────────────────────────────
    if args.phase >= 2:
        ckpt_best = os.path.join(save_dir, 'slm_phase2_best.pt')
        if os.path.exists(ckpt_best):
            load_checkpoint(model, ckpt_best, device=device)
        if not args.synthetic:
            te_ds2 = te_loader.dataset
            te_loader_eval = DataLoader(te_ds2, batch_size=args.batch_size)
        else:
            te_ds.phase = 2
            te_loader_eval = DataLoader(te_ds, batch_size=args.batch_size)

        metrics = evaluate_model(model, te_loader_eval, device=device)
        metrics_path = os.path.join(save_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Metrics saved → {metrics_path}")

    # ── ONNX Export ───────────────────────────────────────────────────────────
    if args.export_onnx:
        load_checkpoint(model, os.path.join(save_dir, 'slm_phase2_best.pt'),
                        device=torch.device('cpu'))
        export_onnx(model, os.path.join(save_dir, 'spectrum_slm.onnx'))


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spectrum-SLM Training Pipeline')
    parser.add_argument('--data_dir',    type=str,   default='.',
                        help='Path to SDR_Data directory')
    parser.add_argument('--save_dir',    type=str,   default='./slm_checkpoints',
                        help='Directory to save checkpoints and metrics')
    parser.add_argument('--phase',       type=int,   default=2,
                        choices=[1, 2, 3], help='Training phase to run up to')
    parser.add_argument('--epochs_p1',   type=int,   default=30)
    parser.add_argument('--epochs_p2',   type=int,   default=50)
    parser.add_argument('--epochs_p3',   type=int,   default=20)
    parser.add_argument('--batch_size',  type=int,   default=64)
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--patience',    type=int,   default=8)
    parser.add_argument('--synthetic',   action='store_true',
                        help='Use synthetic data (no SDR files needed)')
    parser.add_argument('--export_onnx', action='store_true',
                        help='Export final model to ONNX')

    args = parser.parse_args()
    main(args)

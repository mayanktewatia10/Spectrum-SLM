"""
spectrum_slm_dataset_v2.py
==========================
NEW dataset pipeline — Phase 2 new SDR dataset
(files-20260411T185728Z-3-001 / Symbol1, Symbol2, Symbol3)

Supports DQPSK (5-class modulation).
Fully backward-compatible: does NOT modify spectrum_slm_dataset.py.

Key additions vs. original dataset module:
  - load_symbol_dir()       — load one Symbol* sub-directory
  - load_new_dataset()      — aggregate Symbol1+2+3, all 5 modulations
  - build_dataloaders_v2()  — full pipeline → DataLoaders for new dataset
  - save_normalizer() / load_normalizer()  — persist the fitted scaler

Authors : Anjani, Ashish Joshi, Mayank
Guide   : Dr. Abhinandan S.P.
Dated   : April 2026
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple

# ── import shared utilities from original dataset module ─────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectrum_slm_dataset import (
    SpectrumDataset, SpectrumAugmenter, SpectrumNormalizer,
    N_BINS, N_PATCHES
)
from config import (
    MOD_MAP_V2, MOD_MAP_V2_INV, MOD_NAMES_V2,
    PHASE2_DATA_DIR, PHASE2_SYMBOL_DIRS, PHASE2_MODULATIONS,
    PHASE2_BATCH_SIZE, PHASE2_LR, PHASE2_EPOCHS,
    PHASE2_VAL_RATIO, PHASE2_TEST_RATIO, PHASE2_RANDOM_STATE,
    PHASE2_NUM_WORKERS, PHASE2_AUGMENT, CKPT_PHASE2, NORMALIZER_FILE
)


# ─────────────────────────────────────────────────────────────────────────────
# Modulation name normaliser helper
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_mod_name(folder_name: str) -> str:
    """
    Convert folder names to MOD_MAP_V2 keys.
    Examples:
        '16qam'  → '16QAM'
        '8psk'   → '8PSK'
        'dqpsk'  → 'DQPSK'
        'bpsk'   → 'BPSK'
    """
    return folder_name.upper()


# ─────────────────────────────────────────────────────────────────────────────
# 1. PTH file loaders — new dataset format
#    Each modulation folder contains files with different naming conventions.
#    We attempt several patterns and return whatever works.
# ─────────────────────────────────────────────────────────────────────────────

def _load_pth_generic(path: str, mod_id: int, mod_name: str
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a single .pth file from the new dataset.

    Attempts to handle three known formats:
      Format A — psd_binned_by_snr_*.pth  (old format: {'bins':…, 'pairs_by_bin':…})
      Format B — dataset.pth / psd_log_*.pth  (list/dict of raw PSD tensors)
      Format C — direct tensor of shape (N, 176) or structured dict

    Returns (psds, pu_labels, mod_labels, snr_labels) all as numpy arrays.
    """
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"    [WARN] Cannot load {path}: {e}")
        return (np.empty((0, N_BINS), dtype=np.float32),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32))

    psds, pu_labels, mod_labels, snr_labels = [], [], [], []

    # ── Format A: {'bins': [...], 'pairs_by_bin': {snr: [(psd, label), ...]}} ──
    if isinstance(data, dict) and "pairs_by_bin" in data:
        bins  = data.get("bins", [])
        pairs = data.get("pairs_by_bin", {})
        for snr_bin in bins:
            if snr_bin not in pairs:
                continue
            for item in pairs[snr_bin]:
                if isinstance(item, (tuple, list)):
                    if len(item) == 2:
                        psd_vec, label = item
                    elif len(item) >= 3:
                        psd_vec, label, _ = item[0], item[1], item[2:]
                    else:
                        psd_vec = item[0]
                        label = 1
                elif isinstance(item, dict):
                    psd_vec = item.get('psd', item.get('data'))
                    label = item.get('label', 1)
                else:
                    psd_vec = item
                    label = 1
                
                if psd_vec is None: continue
                
                if isinstance(psd_vec, torch.Tensor):
                    psd_np = psd_vec.numpy().astype(np.float32).ravel()
                else:
                    psd_np = np.array(psd_vec, dtype=np.float32).ravel()
                psd_np = psd_np[:N_BINS] if len(psd_np) >= N_BINS \
                         else np.pad(psd_np, (0, N_BINS - len(psd_np)))
                psds.append(psd_np)
                
                if isinstance(label, (np.ndarray, torch.Tensor)):
                    label = label.ravel()[0]
                elif isinstance(label, list) and len(label) > 0:
                    label = label[0]
                
                pu_labels.append(int(label))
                mod_labels.append(mod_id)
                snr_labels.append(float(snr_bin))

    # ── Format B: dict with 'psd' key or 'data' key ──────────────────────────
    elif isinstance(data, dict) and ("psd" in data or "data" in data):
        key = "psd" if "psd" in data else "data"
        psd_tensor = data[key]          # (N, 176) or (N, bins)
        pu_tensor  = data.get("label", data.get("pu_label", None))
        snr_tensor = data.get("snr", data.get("snr_db", None))

        if isinstance(psd_tensor, torch.Tensor):
            psd_np = psd_tensor.numpy().astype(np.float32)
        else:
            psd_np = np.array(psd_tensor, dtype=np.float32)

        n = len(psd_np)
        # Pad / trim to N_BINS
        if psd_np.ndim == 1:
            psd_np = psd_np.reshape(1, -1)
        if psd_np.shape[1] < N_BINS:
            psd_np = np.pad(psd_np, ((0, 0), (0, N_BINS - psd_np.shape[1])))
        else:
            psd_np = psd_np[:, :N_BINS]

        psds.extend(list(psd_np))
        mod_labels.extend([mod_id] * n)

        if pu_tensor is not None:
            if isinstance(pu_tensor, torch.Tensor):
                pu_labels.extend(pu_tensor.numpy().astype(np.int64).tolist())
            else:
                pu_labels.extend([int(v) for v in pu_tensor])
        else:
            # No explicit PU label — infer: PU=1 if signal is present (non-noise mod)
            pu_labels.extend([1] * n)

        if snr_tensor is not None:
            if isinstance(snr_tensor, torch.Tensor):
                snr_labels.extend(snr_tensor.numpy().astype(np.float32).tolist())
            else:
                snr_labels.extend([float(v) for v in snr_tensor])
        else:
            snr_labels.extend([10.0] * n)   # default SNR

    # ── Format C: raw tensor (N, features) or list ────────────────────────────
    elif isinstance(data, (torch.Tensor, np.ndarray)):
        if isinstance(data, torch.Tensor):
            arr = data.numpy().astype(np.float32)
        else:
            arr = np.array(data, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n = len(arr)
        # Trim / pad to N_BINS
        if arr.shape[1] < N_BINS:
            arr = np.pad(arr, ((0, 0), (0, N_BINS - arr.shape[1])))
        else:
            arr = arr[:, :N_BINS]
        psds.extend(list(arr))
        pu_labels.extend([1] * n)
        mod_labels.extend([mod_id] * n)
        snr_labels.extend([10.0] * n)

    # ── Format D: list of dicts or list of tensors ───────────────────────────
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                psd_vec = item.get("psd", item.get("data", None))
                pu      = item.get("pu_label", item.get("label", 1))
                snr     = item.get("snr_db",   item.get("snr", 10.0))
            elif isinstance(item, (torch.Tensor, np.ndarray)):
                psd_vec = item
                pu, snr = 1, 10.0
            else:
                continue
            if psd_vec is None:
                continue
            if isinstance(psd_vec, torch.Tensor):
                psd_np = psd_vec.numpy().astype(np.float32).ravel()
            else:
                psd_np = np.array(psd_vec, dtype=np.float32).ravel()
            psd_np = psd_np[:N_BINS] if len(psd_np) >= N_BINS \
                     else np.pad(psd_np, (0, N_BINS - len(psd_np)))
            psds.append(psd_np)
            pu_labels.append(int(pu))
            mod_labels.append(mod_id)
            snr_labels.append(float(snr))

    else:
        print(f"    [WARN] Unknown PTH format in {path}: {type(data)}")

    if not psds:
        return (np.empty((0, N_BINS), dtype=np.float32),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32))

    return (np.stack(psds).astype(np.float32),
            np.array(pu_labels,  dtype=np.int64),
            np.array(mod_labels, dtype=np.int64),
            np.array(snr_labels, dtype=np.float32))


def _load_csv_mod(csv_path: str, mod_id: int
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fallback: load PSD data from psd_log.csv (new dataset format).
    CSV columns expected: Timestamp, Mean_PSD_dB, SNR_dB, PU_Present
    """
    if not os.path.exists(csv_path):
        return (np.empty((0, N_BINS), dtype=np.float32),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32))

    df = pd.read_csv(csv_path)
    if "Mean_PSD_dB" not in df.columns:
        return (np.empty((0, N_BINS), dtype=np.float32),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32))

    n = len(df)
    mean_psd = df["Mean_PSD_dB"].values.astype(np.float32)
    freqs    = np.linspace(-1, 1, N_BINS)
    lobe     = np.exp(-freqs**2 / 0.3)
    psds     = mean_psd[:, None] + 3 * lobe[None, :]
    psds    += np.random.randn(n, N_BINS).astype(np.float32) * 0.5

    pu  = df["PU_Present"].values.astype(np.int64)  if "PU_Present" in df.columns \
          else np.ones(n, dtype=np.int64)
    snr = df["SNR_dB"].values.astype(np.float32)    if "SNR_dB"     in df.columns \
          else np.full(n, 10.0, dtype=np.float32)

    return (psds.astype(np.float32),
            pu,
            np.full(n, mod_id, dtype=np.int64),
            snr)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Symbol directory loader
# ─────────────────────────────────────────────────────────────────────────────

def load_symbol_dir(symbol_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one Symbol* directory (e.g. Symbol1/).

    Structure expected:
        symbol_dir/
            bpsk/     — psd_binned_by_snr_BPSK.pth  or  dataset.pth + psd_log.csv
            qpsk/     — …
            8psk/     — …
            16qam/    — …
            dqpsk/    — …

    Returns aggregated (psds, pu_labels, mod_labels, snr_labels).
    """
    all_psds, all_pu, all_mod, all_snr = [], [], [], []

    for mod_folder in PHASE2_MODULATIONS:
        mod_dir = os.path.join(symbol_dir, mod_folder)
        if not os.path.isdir(mod_dir):
            continue

        mod_name = _normalise_mod_name(mod_folder)
        mod_id   = MOD_MAP_V2.get(mod_name, -1)
        if mod_id < 0:
            print(f"  [WARN] Unknown modulation '{mod_name}' — skipping")
            continue

        loaded = False

        # ── Try binned PTH first (high priority) ─────────────────────────────
        for pattern in [
            f"psd_binned_by_snr_{mod_folder}.pth",
            f"psd_binned_by_snr_{mod_name}.pth",
            f"psd_binned_by_snr_{mod_folder.upper()}.pth",
        ]:
            path = os.path.join(mod_dir, pattern)
            if os.path.exists(path):
                psds, pu, mod, snr = _load_pth_generic(path, mod_id, mod_name)
                if len(psds) > 0:
                    all_psds.append(psds); all_pu.append(pu)
                    all_mod.append(mod);   all_snr.append(snr)
                    print(f"    {mod_folder:8s} [binned PTH] {len(psds):>7,} samples")
                    loaded = True
                    break

        if loaded:
            continue

        # ── Try generic dataset.pth / psd_log_*.pth ──────────────────────────
        pth_files = glob.glob(os.path.join(mod_dir, "*.pth"))
        if pth_files:
            for pth_path in pth_files:
                psds, pu, mod, snr = _load_pth_generic(pth_path, mod_id, mod_name)
                if len(psds) > 0:
                    all_psds.append(psds); all_pu.append(pu)
                    all_mod.append(mod);   all_snr.append(snr)
                    print(f"    {mod_folder:8s} [PTH {os.path.basename(pth_path)}] "
                          f"{len(psds):>7,} samples")
                    loaded = True
                    break

        if loaded:
            continue

        # ── Fallback: CSV ─────────────────────────────────────────────────────
        csv_files = glob.glob(os.path.join(mod_dir, "*.csv"))
        if csv_files:
            psds, pu, mod, snr = _load_csv_mod(csv_files[0], mod_id)
            if len(psds) > 0:
                all_psds.append(psds); all_pu.append(pu)
                all_mod.append(mod);   all_snr.append(snr)
                print(f"    {mod_folder:8s} [CSV fallback] {len(psds):>7,} samples")
                loaded = True

        if not loaded:
            print(f"    {mod_folder:8s} [SKIP] no loadable files in {mod_dir}")

    if not all_psds:
        return (np.empty((0, N_BINS), dtype=np.float32),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32))

    return (np.concatenate(all_psds),
            np.concatenate(all_pu),
            np.concatenate(all_mod),
            np.concatenate(all_snr))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Full new-dataset loader
# ─────────────────────────────────────────────────────────────────────────────

def load_new_dataset(
    data_dir: str = PHASE2_DATA_DIR,
    symbol_dirs: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the complete Phase 2 new dataset from all Symbol* sub-directories.

    Args:
        data_dir    : root of the new dataset
                      (default from config: PHASE2_DATA_DIR)
        symbol_dirs : list of sub-dir names to include
                      (default: ['Symbol1', 'Symbol2', 'Symbol3'])

    Returns:
        psds       : (N, 176) float32
        pu_labels  : (N,)     int64   0/1
        mod_labels : (N,)     int64   0..4 (BPSK/QPSK/8PSK/16QAM/DQPSK)
        snr_labels : (N,)     float32 dB
    """
    if symbol_dirs is None:
        symbol_dirs = PHASE2_SYMBOL_DIRS

    print(f"\n{'='*60}")
    print(f"  Loading Phase 2 New Dataset")
    print(f"  Root : {data_dir}")
    print(f"{'='*60}")

    all_psds, all_pu, all_mod, all_snr = [], [], [], []

    for sdir in symbol_dirs:
        symbol_path = os.path.join(data_dir, sdir)
        if not os.path.isdir(symbol_path):
            # Maybe data_dir IS the symbol-level dir (e.g. on Kaggle)
            symbol_path = data_dir
            if not os.path.isdir(symbol_path):
                print(f"  [WARN] Symbol dir not found: {symbol_path}")
                continue

        print(f"\n  ── {sdir} ──")
        psds, pu, mod, snr = load_symbol_dir(symbol_path)
        if len(psds) == 0:
            print(f"    [WARN] No data found in {symbol_path}")
            continue

        all_psds.append(psds)
        all_pu.append(pu)
        all_mod.append(mod)
        all_snr.append(snr)

    if not all_psds:
        raise RuntimeError(
            f"No data found in '{data_dir}'. "
            "Ensure the Kaggle dataset is mounted or PHASE2_DATA_DIR is correct."
        )

    psds       = np.concatenate(all_psds)
    pu_labels  = np.concatenate(all_pu)
    mod_labels = np.concatenate(all_mod)
    snr_labels = np.concatenate(all_snr)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  ✓ Total samples loaded : {len(psds):,}")
    print(f"  PU=1 : {pu_labels.sum():,}  ({100*pu_labels.mean():.1f}%)")
    print(f"  SNR  : {snr_labels.min():.1f} — {snr_labels.max():.1f} dB")
    print(f"  Modulation distribution:")
    for mid, mname in enumerate(MOD_NAMES_V2):
        cnt = (mod_labels == mid).sum()
        print(f"    [{mid}] {mname:8s} : {cnt:>7,}  ({100*cnt/len(mod_labels):.1f}%)")

    return psds, pu_labels, mod_labels, snr_labels


# ─────────────────────────────────────────────────────────────────────────────
# 4. Normalizer persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_normalizer(normalizer: SpectrumNormalizer, path: str) -> None:
    """Persist the fitted normalizer to disk (pickle)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(normalizer, f)
    print(f"  ✓ Normalizer saved → {path}")


def load_normalizer(path: str) -> SpectrumNormalizer:
    """Load a previously fitted normalizer from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Normalizer file not found: {path}")
    with open(path, "rb") as f:
        norm = pickle.load(f)
    print(f"  ✓ Normalizer loaded from {path}")
    return norm


# ─────────────────────────────────────────────────────────────────────────────
# 5. DataLoader builder for Phase 2
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders_v2(
    data_dir:            str             = PHASE2_DATA_DIR,
    batch_size:          int             = PHASE2_BATCH_SIZE,
    val_ratio:           float           = PHASE2_VAL_RATIO,
    test_ratio:          float           = PHASE2_TEST_RATIO,
    num_workers:         int             = PHASE2_NUM_WORKERS,
    random_state:        int             = PHASE2_RANDOM_STATE,
    augment_train:       bool            = PHASE2_AUGMENT,
    use_weighted_sampler: bool           = True,
    normalizer_save_path: Optional[str]  = None,
    symbol_dirs:         Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, SpectrumNormalizer, dict]:
    """
    Full Phase 2 data pipeline:
      load → normalise → train/val/test split → augment → DataLoaders

    Args:
        data_dir             : root of the new dataset
        batch_size           : mini-batch size
        val_ratio            : fraction for validation
        test_ratio           : fraction for test
        num_workers          : DataLoader worker processes
        random_state         : reproducibility seed
        augment_train        : apply SpectrumAugmenter to training set
        use_weighted_sampler : balance PU classes via WeightedRandomSampler
        normalizer_save_path : if given, saves fitted normalizer to this path
        symbol_dirs          : which Symbol dirs to include (None = all)

    Returns:
        train_loader, val_loader, test_loader, normalizer, meta_dict
    """
    # 1. Load raw data
    psds, pu_labels, mod_labels, snr_labels = load_new_dataset(data_dir, symbol_dirs)
    
    # Ensure pu_labels are 0 or 1 and not negative
    pu_labels = np.clip(pu_labels, 0, 1)

    # 2. Split indices (stratify on PU label)
    idx = np.arange(len(psds))
    idx_train, idx_tmp = train_test_split(
        idx, test_size=(val_ratio + test_ratio),
        stratify=pu_labels, random_state=random_state
    )
    vt_frac = test_ratio / (val_ratio + test_ratio)
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=vt_frac,
        stratify=pu_labels[idx_tmp], random_state=random_state
    )

    # 3. Normalise (fit on train only)
    normalizer = SpectrumNormalizer()
    psds_train = normalizer.fit_transform(psds[idx_train])
    psds_val   = normalizer.transform(psds[idx_val])
    psds_test  = normalizer.transform(psds[idx_test])

    if normalizer_save_path:
        save_normalizer(normalizer, normalizer_save_path)

    # 4. Augmenter
    augmenter = SpectrumAugmenter() if augment_train else None

    # 5. Datasets (Phase 2 = supervised multi-task)
    def _ds(psd_data, is_arr, train_flag):
        """Map psd_data back to correct label slices via is_arr flag."""
        if is_arr == "train":
            idx_arr = idx_train
        elif is_arr == "val":
            idx_arr = idx_val
        else:
            idx_arr = idx_test
        return SpectrumDataset(
            psds       = psd_data,
            pu_labels  = pu_labels[idx_arr],
            mod_labels = mod_labels[idx_arr],
            snr_labels = snr_labels[idx_arr],
            phase      = 2,
            augmenter  = augmenter if train_flag else None,
            training   = train_flag,
        )

    train_ds = _ds(psds_train, "train", True)
    val_ds   = _ds(psds_val,   "val",   False)
    test_ds  = _ds(psds_test,  "test",  False)

    # 6. Weighted sampler (handle PU class imbalance)
    sampler = None
    if use_weighted_sampler:
        pu_train  = pu_labels[idx_train]
        counts    = np.bincount(pu_train, minlength=2)
        weights   = 1.0 / np.maximum(counts, 1)
        sw        = weights[pu_train]
        sampler   = WeightedRandomSampler(
            torch.DoubleTensor(sw), len(train_ds), replacement=True
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader   = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # 7. PU class weights for Focal Loss
    pu_counts  = np.bincount(pu_labels[idx_train], minlength=2)
    pu_weights = torch.tensor(
        len(idx_train) / (2 * np.maximum(pu_counts, 1)),
        dtype=torch.float32
    )

    meta = {
        "n_total"    : len(psds),
        "n_train"    : len(idx_train),
        "n_val"      : len(idx_val),
        "n_test"     : len(idx_test),
        "pu_weights" : pu_weights,
        "snr_mean"   : float(snr_labels[idx_train].mean()),
        "snr_std"    : float(snr_labels[idx_train].std()),
        "mod_counts" : {MOD_NAMES_V2[i]: int((mod_labels[idx_train] == i).sum())
                        for i in range(len(MOD_NAMES_V2))},
        "n_mod_classes": len(MOD_NAMES_V2),
    }

    print(f"\n  DataLoaders ready (Phase 2 — new dataset):")
    print(f"  Train:{meta['n_train']:,}  Val:{meta['n_val']:,}  "
          f"Test:{meta['n_test']:,}  Batch:{batch_size}")
    return train_loader, val_loader, test_loader, normalizer, meta


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("Running spectrum_slm_dataset_v2 sanity check...")
    print(f"  MOD_MAP_V2  : {MOD_MAP_V2}")
    print(f"  MOD_NAMES_V2: {MOD_NAMES_V2}")

    try:
        train_l, val_l, test_l, norm, meta = build_dataloaders_v2()
        batch = next(iter(train_l))
        psd_b, pu_b, mod_b, snr_b = batch
        print(f"\n  Batch shapes — PSD:{psd_b.shape} PU:{pu_b.shape} "
              f"Mod:{mod_b.shape} SNR:{snr_b.shape}")
        print(f"  Mod IDs in batch: {mod_b.unique().tolist()}")
        print("\n  ✓ Dataset v2 pipeline OK!")
    except RuntimeError as e:
        print(f"\n  [INFO] {e}")
        print("  (Normal if PHASE2_DATA_DIR not present locally — "
              "run on Kaggle with dataset mounted)")

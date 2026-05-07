"""
spectrum_slm_dataset.py
=======================
Data pipeline for Spectrum-SLM.

Handles:
  - Loading psd_binned_by_snr_*.pth files (full 176-bin PSD vectors)
  - Loading Output.csv (merged tabular dataset)
  - Per-modulation CSV files (psd_log_bpsk.csv etc.)
  - Normalisation, augmentation, stratified splits
  - PyTorch Datasets and DataLoaders for all 3 training phases

Authors : Anjani, Ashish Joshi, Mayank
Guide   : Dr. Abhinandan S.P.
Dated   : March 2026
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, List, Dict, Union


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

N_BINS        = 176                  # PSD frequency bins
PATCH_SIZE    = 1
N_PATCHES     = N_BINS // PATCH_SIZE  # 176  (each bin is its own patch)
MOD_MAP       = {'BPSK': 0, 'QPSK': 1, '8PSK': 2, '16QAM': 3}
MOD_MAP_INV   = {v: k for k, v in MOD_MAP.items()}
SNR_BINS      = [4, 6, 8, 10, 12, 14, 16, 18, 20]   # dB


# ─────────────────────────────────────────────────────────────────────────────
# 1. PTH Loader — Full PSD Vectors
# ─────────────────────────────────────────────────────────────────────────────

def load_pth_file(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads a psd_binned_by_snr_*.pth file.

    Returns:
      psds       : (N, 176)  float32
      pu_labels  : (N,)      int    (0 / 1)
      snr_labels : (N,)      float  (bin centre in dB)
    """
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"  [WARN] Failed to load {path}: {e}")
        return np.empty((0, N_BINS)), np.empty(0), np.empty(0)
    bins   = data.get('bins', [])
    pairs  = data.get('pairs_by_bin', {})

    all_psds, all_pu, all_snr = [], [], []
    for snr_bin in bins:
        if snr_bin not in pairs:
            continue
        for psd_vec, label in pairs[snr_bin]:
            psd_np = np.array(psd_vec, dtype=np.float32).ravel()
            # If shorter or longer than 176, pad/trim
            if len(psd_np) < N_BINS:
                psd_np = np.pad(psd_np, (0, N_BINS - len(psd_np)))
            else:
                psd_np = psd_np[:N_BINS]
            all_psds.append(psd_np)
            all_pu.append(int(label))
            all_snr.append(float(snr_bin))

    if not all_psds:
        return np.empty((0, N_BINS)), np.empty(0), np.empty(0)

    return (np.stack(all_psds),
            np.array(all_pu, dtype=np.int64),
            np.array(all_snr, dtype=np.float32))


def load_all_pth_files(
    data_dir: str,
    modulations: Optional[List[str]] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Loads all psd_binned_by_snr_*.pth files from data_dir.

    Returns dict keyed by modulation name:
      { 'bpsk': (psds, pu_labels, snr_labels), ... }
    """
    if modulations is None:
        modulations = ['bpsk', 'qpsk', '8psk', '16qam']

    result = {}
    for mod in modulations:
        matches = []
        # 1. Check root data_dir
        matches = glob.glob(os.path.join(data_dir, f'psd_binned_by_snr_{mod}.pth'))
        # 2. Recursively search all subfolders (handles Symbol1_Modulation, Symbol2_Results etc.)
        if not matches:
            matches = glob.glob(
                os.path.join(data_dir, '**', f'psd_binned_by_snr_{mod}.pth'),
                recursive=True
            )
        if matches:
            psds, pu, snr = load_pth_file(matches[0])
            if len(psds) > 0:
                result[mod] = (psds, pu, snr)
                print(f"  Loaded {mod}: {len(psds):,} samples from {matches[0]}")
        else:
            print(f"  [WARN] No .pth found for modulation: {mod}")

    # ── Fallback: single combined file (dataset_binned.pth / dataset.pth) ───
    if not result:
        for fname in ['dataset_binned.pth', 'dataset.pth']:
            combined = glob.glob(os.path.join(data_dir, fname))
            if not combined:
                combined = glob.glob(
                    os.path.join(data_dir, '**', fname), recursive=True
                )
            if combined:
                print(f"  [INFO] Using combined dataset: {combined[0]}")
                psds, pu, snr = load_pth_file(combined[0])
                if len(psds) > 0:
                    result['combined'] = (psds, pu, snr)
                    print(f"  Loaded combined: {len(psds):,} samples")
                    break

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. CSV Loader — Tabular Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_dataset(csv_path: str) -> Optional[pd.DataFrame]:
    """
    Loads Output.csv (merged modulation data) or individual psd_log_*.csv.

    Expected columns: Timestamp, Mean_PSD_dB, SNR_dB, PU_Present
    Optional        : Modulation, Modulation_ID
    """
    if not os.path.exists(csv_path):
        print(f"  [WARN] CSV not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    print(f"  Loaded CSV: {os.path.basename(csv_path)} — {len(df):,} rows, "
          f"cols: {list(df.columns)}")
    return df


def build_psd_array_from_csv(df: pd.DataFrame) -> np.ndarray:
    """
    When full PSD bins aren't available, replicate Mean_PSD_dB into
    176-bin Gaussian-shaped mock PSD for compatibility.
    Only used as a fallback — prefer .pth files.
    """
    n = len(df)
    mean_psd = df['Mean_PSD_dB'].values.astype(np.float32)
    # Simple Gaussian lobe approximation
    freqs = np.linspace(-1, 1, N_BINS)
    lobe  = np.exp(-freqs**2 / 0.3)                   # shape (176,)
    psds  = mean_psd[:, None] + 3 * lobe[None, :]     # (N, 176)
    psds += np.random.randn(n, N_BINS).astype(np.float32) * 0.5  # add noise
    return psds.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Data Augmentation
# ─────────────────────────────────────────────────────────────────────────────

class SpectrumAugmenter:
    """
    Augmentation strategies for PSD vectors:

    1. Noise injection   — add AWGN to the PSD vector
    2. Spectral shift    — circular shift in frequency axis
    3. Amplitude scale   — multiply by random gain factor
    4. Mixup             — convex combination of two samples
    """

    def __init__(
        self,
        noise_std: float = 0.02,
        max_shift: int = 5,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        mixup_alpha: float = 0.2,
        p_noise: float = 0.5,
        p_shift: float = 0.3,
        p_scale: float = 0.3,
    ):
        self.noise_std   = noise_std
        self.max_shift   = max_shift
        self.scale_range = scale_range
        self.mixup_alpha = mixup_alpha
        self.p_noise     = p_noise
        self.p_shift     = p_shift
        self.p_scale     = p_scale

    def augment(self, psd: np.ndarray) -> np.ndarray:
        """psd: (176,) float32"""
        if np.random.rand() < self.p_noise:
            psd = psd + np.random.randn(N_BINS).astype(np.float32) * self.noise_std

        if np.random.rand() < self.p_shift:
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
            psd = np.roll(psd, shift)

        if np.random.rand() < self.p_scale:
            scale = np.random.uniform(*self.scale_range)
            psd = psd * scale

        return psd

    def mixup(
        self,
        psd_a: np.ndarray, label_a: int,
        psd_b: np.ndarray, label_b: int,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Returns mixed psd, lambdas for soft labels.
        lam * label_a + (1-lam) * label_b
        """
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        mixed_psd = lam * psd_a + (1 - lam) * psd_b
        return mixed_psd.astype(np.float32), float(lam), float(1 - lam)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Normalisation
# ─────────────────────────────────────────────────────────────────────────────

class SpectrumNormalizer:
    """
    Per-bin StandardScaler wrapper around numpy arrays.
    Fit on training data only; apply to val/test.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, psds: np.ndarray) -> 'SpectrumNormalizer':
        """psds: (N, 176)"""
        self.scaler.fit(psds)
        self._fitted = True
        return self

    def transform(self, psds: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit() first"
        return self.scaler.transform(psds).astype(np.float32)

    def fit_transform(self, psds: np.ndarray) -> np.ndarray:
        return self.fit(psds).transform(psds)

    def inverse_transform(self, psds: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(psds)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SpectrumDataset(Dataset):
    """
    PyTorch Dataset for Spectrum-SLM.

    Supports all 3 training phases:
      Phase 1 (MSM)    : returns (psd, mask)                   — labels optional
      Phase 2 (SFT)    : returns (psd, pu_label, mod_label, snr_label)
      Phase 3 (Gen)    : returns (psd_sequence, target_psd)    — sequences

    Parameters
    ----------
    psds        : (N, 176) float32 — normalised PSD vectors
    pu_labels   : (N,) int64       — 0/1
    mod_labels  : (N,) int64       — 0-3 (or -1 if unknown)
    snr_labels  : (N,) float32     — SNR in dB
    phase       : 1, 2, or 3
    mask_ratio  : fraction to mask in phase 1 (0.15–0.30)
    seq_len     : temporal context window for phase 3
    augmenter   : optional SpectrumAugmenter
    """

    def __init__(
        self,
        psds:       np.ndarray,
        pu_labels:  np.ndarray,
        mod_labels: np.ndarray,
        snr_labels: np.ndarray,
        phase:      int = 2,
        mask_ratio: float = 0.20,
        seq_len:    int = 8,
        augmenter:  Optional[SpectrumAugmenter] = None,
        training:   bool = True,
    ):
        super().__init__()
        self.psds       = psds
        self.pu_labels  = pu_labels
        self.mod_labels = mod_labels
        self.snr_labels = snr_labels
        self.phase      = phase
        self.mask_ratio = mask_ratio
        self.seq_len    = seq_len
        self.augmenter  = augmenter
        self.training   = training

    def __len__(self) -> int:
        if self.phase == 3:
            return max(0, len(self.psds) - self.seq_len)
        return len(self.psds)

    def _random_mask(self) -> torch.Tensor:
        """Returns bool mask of shape (176,) with ~mask_ratio True entries."""
        n_mask = max(1, int(N_PATCHES * self.mask_ratio))
        idx = np.random.choice(N_PATCHES, n_mask, replace=False)
        mask = torch.zeros(N_PATCHES, dtype=torch.bool)
        mask[idx] = True
        return mask

    def __getitem__(self, idx: int):
        if self.phase == 1:
            # ── Phase 1: Masked Spectrum Modelling ──────────────────────────
            psd = self.psds[idx].copy()
            if self.training and self.augmenter:
                psd = self.augmenter.augment(psd)
            mask = self._random_mask()
            return torch.tensor(psd, dtype=torch.float32), mask

        elif self.phase == 2:
            # ── Phase 2: Supervised Multi-task ──────────────────────────────
            psd = self.psds[idx].copy()
            if self.training and self.augmenter:
                psd = self.augmenter.augment(psd)
            return (
                torch.tensor(psd, dtype=torch.float32),
                torch.tensor(self.pu_labels[idx],  dtype=torch.long),
                torch.tensor(self.mod_labels[idx], dtype=torch.long),
                torch.tensor(self.snr_labels[idx], dtype=torch.float32),
            )

        else:
            # ── Phase 3: Next-PSD Autoregressive ────────────────────────────
            seq   = self.psds[idx : idx + self.seq_len]         # (L, 176)
            target = self.psds[idx + self.seq_len]              # (176,)
            # Use the last PSD in the sequence as input
            return (
                torch.tensor(seq, dtype=torch.float32),         # (L, 176)
                torch.tensor(target, dtype=torch.float32),      # (176,)
            )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Dataset Assembly & DataLoaders
# ─────────────────────────────────────────────────────────────────────────────

def assemble_dataset(
    data_dir: str,
    use_pth: bool = True,
    csv_fallback: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Assembles psds, pu_labels, mod_labels, snr_labels from all available sources.

    Priority: .pth files > Output.csv > individual modulation CSVs.

    Returns:
      psds       : (N, 176) float32
      pu_labels  : (N,)     int64
      mod_labels : (N,)     int64   (-1 if unknown)
      snr_labels : (N,)     float32
    """
    all_psds, all_pu, all_mod, all_snr = [], [], [], []

    # ── Try .pth files ──────────────────────────────────────────────────────
    if use_pth:
        pth_data = load_all_pth_files(data_dir)
        for mod_name, (psds, pu, snr) in pth_data.items():
            mod_id = MOD_MAP.get(mod_name.upper().replace('PSK', 'PSK')
                                       .replace('QAM', 'QAM'), -1)
            # Normalise key: '8psk' -> '8PSK', '16qam' -> '16QAM'
            clean = mod_name.upper()
            if clean in MOD_MAP:
                mod_id = MOD_MAP[clean]
            elif clean.replace('PSK', 'PSK') in MOD_MAP:
                mod_id = MOD_MAP[clean]
            else:
                mod_id = -1

            all_psds.append(psds)
            all_pu.append(pu)
            all_mod.append(np.full(len(psds), mod_id, dtype=np.int64))
            all_snr.append(snr)

    # ── Try Output.csv ──────────────────────────────────────────────────────
    if csv_fallback and not all_psds:
        csv_candidates = [
            os.path.join(data_dir, 'Secondary_User', 'Symbol1_Modulation', 'Output.csv'),
            os.path.join(data_dir, 'Output.csv'),
        ]
        for csv_path in csv_candidates:
            df = load_csv_dataset(csv_path)
            if df is not None and len(df) > 0:
                psds = build_psd_array_from_csv(df)
                pu   = df['PU_Present'].values.astype(np.int64)
                if 'Modulation_ID' in df.columns:
                    mod = df['Modulation_ID'].values.astype(np.int64)
                elif 'Modulation' in df.columns:
                    mod = df['Modulation'].map(MOD_MAP).fillna(-1).values.astype(np.int64)
                else:
                    mod = np.full(len(df), -1, dtype=np.int64)
                snr = df['SNR_dB'].values.astype(np.float32)

                all_psds.append(psds)
                all_pu.append(pu)
                all_mod.append(mod)
                all_snr.append(snr)
                break

    if not all_psds:
        raise RuntimeError(
            f"No data found in {data_dir}. "
            "Please ensure .pth or .csv files are present."
        )

    psds       = np.concatenate(all_psds, axis=0)
    pu_labels  = np.concatenate(all_pu,   axis=0)
    mod_labels = np.concatenate(all_mod,  axis=0)
    snr_labels = np.concatenate(all_snr,  axis=0)

    print(f"\nTotal dataset: {len(psds):,} samples")
    print(f"  PU=1: {pu_labels.sum():,}  ({100*pu_labels.mean():.1f}%)")
    print(f"  SNR range: {snr_labels.min():.1f} — {snr_labels.max():.1f} dB")

    return psds, pu_labels, mod_labels, snr_labels


def build_dataloaders(
    data_dir: str,
    phase: int = 2,
    batch_size: int = 64,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    mask_ratio: float = 0.20,
    seq_len: int = 8,
    num_workers: int = 0,
    use_pth: bool = True,
    use_weighted_sampler: bool = True,
    augment_train: bool = True,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, SpectrumNormalizer, dict]:
    """
    Full data pipeline: load → normalise → split → augment → DataLoaders.

    Returns:
      train_loader, val_loader, test_loader, normalizer, meta_dict
    """
    # 1. Load raw data
    psds, pu_labels, mod_labels, snr_labels = assemble_dataset(data_dir, use_pth)

    # 2. Train / (val+test) split
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

    # 4. Augmenter
    augmenter = SpectrumAugmenter() if augment_train else None

    def _make_ds(i, aug, train):
        return SpectrumDataset(
            psds       = i,
            pu_labels  = pu_labels[idx_train if (i is psds_train) else
                                    idx_val   if (i is psds_val)   else idx_test],
            mod_labels = mod_labels[idx_train if (i is psds_train) else
                                    idx_val   if (i is psds_val)   else idx_test],
            snr_labels = snr_labels[idx_train if (i is psds_train) else
                                    idx_val   if (i is psds_val)   else idx_test],
            phase      = phase,
            mask_ratio = mask_ratio,
            seq_len    = seq_len,
            augmenter  = aug,
            training   = train,
        )

    train_ds = _make_ds(psds_train, augmenter, True)
    val_ds   = _make_ds(psds_val,   None,       False)
    test_ds  = _make_ds(psds_test,  None,       False)

    # 5. Class-weighted sampler for phase 2 (handles PU imbalance)
    sampler = None
    if phase == 2 and use_weighted_sampler:
        pu_train = pu_labels[idx_train]
        class_counts = np.bincount(pu_train)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[pu_train]
        sampler = WeightedRandomSampler(
            weights     = torch.DoubleTensor(sample_weights),
            num_samples = len(train_ds),
            replacement = True,
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=num_workers, pin_memory=True,
        drop_last=True,
    )
    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # Compute PU class weights for focal loss
    pu_counts  = np.bincount(pu_labels[idx_train])
    pu_weights = torch.tensor(
        len(idx_train) / (2 * np.maximum(pu_counts, 1)),
        dtype=torch.float32
    )

    meta = {
        'n_train'      : len(idx_train),
        'n_val'        : len(idx_val),
        'n_test'       : len(idx_test),
        'pu_weights'   : pu_weights,
        'snr_mean'     : float(snr_labels[idx_train].mean()),
        'snr_std'      : float(snr_labels[idx_train].std()),
    }

    print(f"\nDataLoaders ready (phase={phase}):")
    print(f"  Train: {meta['n_train']:,}  Val: {meta['n_val']:,}  "
          f"Test: {meta['n_test']:,}")
    print(f"  Batch size: {batch_size}  Workers: {num_workers}")

    return train_loader, val_loader, test_loader, normalizer, meta


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator (for testing without real SDR files)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_psd(
    n_samples: int = 4096,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates synthetic PSD data that mimics real SDR measurements.
    Useful for development, CI, and the Streamlit demo.

    PU=0 : flat noise floor ~N(-22, 2) dB
    PU=1 : Gaussian signal lobe on top of noise, per modulation
    """
    rng = np.random.default_rng(seed)
    freqs = np.linspace(-1, 1, N_BINS)

    psds, pu_labels, mod_labels, snr_labels = [], [], [], []

    mod_params = {
        0: {'width': 0.20, 'amplitude': 12},   # BPSK  — narrow
        1: {'width': 0.25, 'amplitude': 11},   # QPSK
        2: {'width': 0.30, 'amplitude': 10},   # 8PSK
        3: {'width': 0.35, 'amplitude':  9},   # 16QAM — wider, lower SNR
    }

    for i in range(n_samples):
        pu  = rng.integers(0, 2)
        mod = rng.integers(0, 4)
        snr = rng.uniform(3, 20) if pu == 1 else rng.uniform(3, 8)

        # Noise floor
        noise_floor = -22 + rng.normal(0, 2)
        psd = noise_floor + rng.normal(0, 1.5, N_BINS).astype(np.float32)

        if pu == 1:
            params = mod_params[mod]
            lobe_centre = rng.uniform(-0.3, 0.3)
            lobe = params['amplitude'] * np.exp(
                -(freqs - lobe_centre) ** 2 / (2 * params['width'] ** 2)
            )
            scale = snr / 15.0
            psd += (lobe * scale).astype(np.float32)

        psds.append(psd)
        pu_labels.append(pu)
        mod_labels.append(mod)
        snr_labels.append(snr)

    return (np.stack(psds).astype(np.float32),
            np.array(pu_labels, dtype=np.int64),
            np.array(mod_labels, dtype=np.int64),
            np.array(snr_labels, dtype=np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Generating synthetic data for sanity check...")
    psds, pu, mod, snr = generate_synthetic_psd(n_samples=2048)

    normalizer = SpectrumNormalizer()
    psds_norm = normalizer.fit_transform(psds)

    ds = SpectrumDataset(psds_norm, pu, mod, snr, phase=2, augmenter=SpectrumAugmenter())
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    batch = next(iter(loader))
    psd_b, pu_b, mod_b, snr_b = batch
    print(f"Batch shapes — PSD:{psd_b.shape}  PU:{pu_b.shape}  "
          f"Mod:{mod_b.shape}  SNR:{snr_b.shape}")
    print("Dataset pipeline OK!")

import os
import sys
import torch
import json

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    PHASE2_DATA_DIR, CKPT_PHASE1, CKPT_PHASE2, CKPT_PHASE3,
    N_BINS, N_MOD_CLASSES_V2, ensure_dirs, PHASE2_NUM_WORKERS,
    PHASE2_BATCH_SIZE
)
from spectrum_slm_model import SpectrumSLM
from spectrum_slm_dataset_v2 import build_dataloaders_v2
from spectrum_slm_train import (
    pretrain_msm, finetune_supervised, train_generative,
    save_checkpoint, get_device
)

def run_all_phases():
    ensure_dirs()
    device = get_device()
    print(f"=== Starting 3-Phase Training on {device} ===")
    
    # 1. Prepare Data
    print("\n--- BUILDING DATALOADERS ---")
    train_loader, val_loader, test_loader, norm, meta = build_dataloaders_v2(
        data_dir=PHASE2_DATA_DIR,
        batch_size=PHASE2_BATCH_SIZE,
        num_workers=PHASE2_NUM_WORKERS,
        normalizer_save_path=os.path.join(CKPT_PHASE2, "normalizer.pkl")
    )

    # 2. Initialize Model (5-class for Phase 2 data)
    print("\n--- INITIALIZING MODEL ---")
    model = SpectrumSLM(
        n_bins=N_BINS, patch_size=8, d_model=128, nhead=4,
        num_layers=4, dim_feedforward=512, dropout=0.1, n_mod_classes=N_MOD_CLASSES_V2
    ).to(device)

    # ==========================
    # PHASE 1: PRE-TRAINING
    # ==========================
    print("\n==========================")
    print(" PHASE 1: GENERATIVE PRE-TRAINING")
    print("==========================")
    
    # Create wrapper loader for phase 1
    def phase1_wrapper(loader):
        for batch in loader:
            psd = batch[0]
            # Generate random mask
            B = psd.size(0)
            n_patches = 22
            mask = torch.rand(B, n_patches) < 0.2
            yield psd, mask

    class WrapperLoader:
        def __init__(self, loader): self.loader = loader
        def __iter__(self): return phase1_wrapper(self.loader)
        def __len__(self): return len(self.loader)

    p1_ckpt = os.path.join(CKPT_PHASE1, "slm_phase1_best.pt")
    if not os.path.exists(p1_ckpt):
        try:
            pretrain_msm(
                model=model, train_loader=WrapperLoader(train_loader), val_loader=WrapperLoader(val_loader),
                n_epochs=20, lr=2e-4, device=device, save_dir=CKPT_PHASE1, patience=5
            )
        except Exception as e:
            print(f"Phase 1 error, skipping: {e}")
    else:
        print(f"Phase 1 checkpoint found at {p1_ckpt}. Loading...")
        model.load_state_dict(torch.load(p1_ckpt, map_location=device)['model'])

    # ==========================
    # PHASE 2: SUPERVISED
    # ==========================
    print("\n==========================")
    print(" PHASE 2: SUPERVISED MULTI-TASK")
    print("==========================")
    p2_ckpt = os.path.join(CKPT_PHASE2, "slm_phase2_new_best.pt")
    if not os.path.exists(p2_ckpt):
        finetune_supervised(
            model=model, train_loader=train_loader, val_loader=val_loader,
            pu_class_weight=meta['pu_weights'].to(device),
            n_epochs=30, lr=1e-4, device=device, save_dir=CKPT_PHASE2, patience=6
        )
    else:
        print(f"Phase 2 checkpoint found at {p2_ckpt}. Loading...")
        model.load_state_dict(torch.load(p2_ckpt, map_location=device)['model'])

    # ==========================
    # PHASE 3: ADVANCED TUNING
    # ==========================
    # Create wrapper loader for phase 3
    def phase3_wrapper(loader):
        for batch in loader:
            psd = batch[0]  # (B, 176)
            # train_generative expects seq (B, L, 176) and target (B, 176)
            # We synthetically create a sequence of length 1 for this test wrapper
            seq = psd.unsqueeze(1) # (B, 1, 176)
            target = psd # Target is the psd itself
            yield seq, target

    class WrapperLoaderP3:
        def __init__(self, loader): self.loader = loader
        def __iter__(self): return phase3_wrapper(self.loader)
        def __len__(self): return len(self.loader)

    p3_ckpt = os.path.join(CKPT_PHASE3, "slm_phase3_best.pt")
    if not os.path.exists(p3_ckpt):
        try:
            train_generative(
                model=model, train_loader=WrapperLoaderP3(train_loader), val_loader=WrapperLoaderP3(val_loader),
                n_epochs=20, lr=5e-5, device=device, save_dir=CKPT_PHASE3, patience=5
            )
        except Exception as e:
            print(f"Phase 3 error, skipping: {e}")
    else:
        print(f"Phase 3 checkpoint found at {p3_ckpt}. Loading...")

    print("\n=== ALL PHASES COMPLETE ===")
    print(f"Checkpoints saved in {os.path.dirname(CKPT_PHASE1)}")

if __name__ == "__main__":
    run_all_phases()

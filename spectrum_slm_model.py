"""
spectrum_slm_model.py
=====================
Spectrum-SLM: A Small Language Model for Cognitive Radio Spectrum Sensing

Architecture:
  PatchEmbedding (176 bins -> 176 patches, patch_size=1) +
  FrequencyAwarePositionalEncoding +
  SpectrumTransformerEncoder (4 layers, 4 heads, d_model=128) +
  Multi-task heads: PU Detection | Modulation | SNR | Generative

Authors : Anjani, Ashish Joshi, Mayank
Guide   : Dr. Abhinandan S.P.
Dated   : March 2026
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spectrum Tokenizer — Patch Embedding
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Splits a 176-bin PSD vector into non-overlapping patches of size `patch_size`
    and projects each patch to d_model dimensions via a learned linear layer.

    176 bins / 1 bin-per-patch = 176 spectral tokens.

    Input  : (B, 176)
    Output : (B, 177, d_model)   [176 patches + 1 prepended CLS token]
    """

    def __init__(self, n_bins: int = 176, patch_size: int = 1, d_model: int = 128):
        super().__init__()
        assert n_bins % patch_size == 0, "n_bins must be divisible by patch_size"
        self.n_bins = n_bins
        self.patch_size = patch_size
        self.n_patches = n_bins // patch_size          # 176  (each bin = 1 patch)
        self.d_model = d_model

        # Linear projection for each patch  (patch_size -> d_model)
        self.projection = nn.Linear(patch_size, d_model)

        # Learnable CLS token (aggregates global spectrum context)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Layer norm after embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 176)
        returns : (B, 23, d_model)
        """
        B = x.size(0)
        # Reshape: (B, 176, 1)
        x = x.view(B, self.n_patches, self.patch_size)
        # Project: (B, 176, d_model)
        x = self.projection(x)
        # Prepend CLS token: (B, 177, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Frequency-Aware Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class FrequencyAwarePositionalEncoding(nn.Module):
    """
    Two-component positional encoding:
      (a) Learnable patch position embedding  (standard transformer PE)
      (b) Sinusoidal frequency-aware encoding tied to physical patch centre-frequency

    The combination informs the model both of *order* (patch 0 vs patch 175)
    and *physical meaning* (lower vs upper frequency region).

    Input  : (B, 177, d_model)
    Output : (B, 177, d_model)
    """

    def __init__(self, n_tokens: int = 177, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Learnable positional embedding
        self.pos_emb = nn.Embedding(n_tokens, d_model)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        # Fixed sinusoidal frequency-aware encoding
        # Position 0 = CLS (no frequency), positions 1-22 = patch centres
        pe = torch.zeros(n_tokens, d_model)
        position = torch.arange(n_tokens, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))          # (1, 23, d_model)

        # Blend weight (learnable scalar)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, L, d_model)"""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        learned = self.pos_emb(positions).unsqueeze(0)       # (1, L, d_model)
        sinusoidal = self.pe[:, :seq_len, :]                 # (1, L, d_model)

        alpha = torch.sigmoid(self.alpha)
        combined = alpha * learned + (1 - alpha) * sinusoidal
        return self.dropout(x + combined)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Transformer Encoder
# ─────────────────────────────────────────────────────────────────────────────

class SpectrumTransformerEncoder(nn.Module):
    """
    Stack of N standard TransformerEncoder layers with pre-LN (more stable
    training) and a final layer norm.

    Input  : (B, 177, d_model)
    Output : (B, 177, d_model)
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multi-Task Output Heads
# ─────────────────────────────────────────────────────────────────────────────

class PUDetectionHead(nn.Module):
    """Binary classification: PU Present (1) / Absent (0)"""
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 2)
        )
    def forward(self, cls_feat: torch.Tensor) -> torch.Tensor:
        return self.net(cls_feat)                # (B, 2) — raw logits


class ModulationHead(nn.Module):
    """4-class classification: BPSK=0, QPSK=1, 8PSK=2, 16QAM=3"""
    def __init__(self, d_model: int = 128, n_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, n_classes)
        )
    def forward(self, cls_feat: torch.Tensor) -> torch.Tensor:
        return self.net(cls_feat)                # (B, 4) — raw logits


class SNRHead(nn.Module):
    """Regression head — predict SNR in dB"""
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    def forward(self, cls_feat: torch.Tensor) -> torch.Tensor:
        return self.net(cls_feat).squeeze(-1)    # (B,)  — scalar SNR


class GenerativeHead(nn.Module):
    """
    Predicts the next PSD snapshot (176 bins) from the CLS representation.
    Used during Phase 3 (autoregressive generative modelling).
    """
    def __init__(self, d_model: int = 128, n_bins: int = 176):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, n_bins)
        )
    def forward(self, cls_feat: torch.Tensor) -> torch.Tensor:
        return self.net(cls_feat)               # (B, 176) — predicted PSD


# ─────────────────────────────────────────────────────────────────────────────
# 5. Masked Spectrum Modelling Head (Phase 1 — Pre-training)
# ─────────────────────────────────────────────────────────────────────────────

class MSMHead(nn.Module):
    """
    Reconstructs masked PSD patches during self-supervised pre-training.
    Operates on all 176 patch positions (not the CLS token).
    """
    def __init__(self, d_model: int = 128, patch_size: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, patch_size)
        )

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        patch_features : (B, 176, d_model) — the 176 patch token representations
        returns        : (B, 176, 1)       — reconstructed patch values (patch_size=1)
        """
        return self.net(patch_features)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main Model: SpectrumSLM
# ─────────────────────────────────────────────────────────────────────────────

class SpectrumSLM(nn.Module):
    """
    Spectrum-SLM: A Small Language Model for Cognitive Radio Spectrum Sensing.

    ┌─────────────────────────────────────────┐
    │  PSD Vector (176 bins)                  │
    │      ↓                                  │
    │  PatchEmbedding (176 patches + CLS)     │
    │      ↓                                  │
    │  FrequencyAware PositionalEncoding      │
    │      ↓                                  │
    │  TransformerEncoder (4L, 4H, d=128)     │
    │      ↓ CLS token                        │
    │  ┌────────────────────────────────┐     │
    │  │ PU Head  Mod Head  SNR  GenHd  │     │
    │  └────────────────────────────────┘     │
    └─────────────────────────────────────────┘

    ~1M parameters — edge-deployable.
    """

    def __init__(
        self,
        n_bins: int = 176,
        patch_size: int = 1,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        n_mod_classes: int = 4,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.d_model = d_model

        self.tokenizer = PatchEmbedding(n_bins, patch_size, d_model)
        self.pos_enc = FrequencyAwarePositionalEncoding(
            n_tokens=self.tokenizer.n_patches + 1,  # +1 for CLS
            d_model=d_model,
            dropout=dropout,
        )
        self.encoder = SpectrumTransformerEncoder(
            d_model=d_model, nhead=nhead,
            num_layers=num_layers, dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Downstream task heads
        self.pu_head   = PUDetectionHead(d_model)
        self.mod_head  = ModulationHead(d_model, n_mod_classes)
        self.snr_head  = SNRHead(d_model)
        self.gen_head  = GenerativeHead(d_model, n_bins)

        # Pre-training head
        self.msm_head  = MSMHead(d_model, patch_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        psd: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_msm: bool = False,
    ) -> dict:
        """
        psd  : (B, 176)  — normalised PSD vector
        mask : (B, 176)  — boolean mask (True = masked patches) for Phase 1
        return_msm : if True, also return MSM reconstruction

        Returns a dict with keys:
          'pu_logits'   : (B, 2)
          'mod_logits'  : (B, 4)
          'snr_pred'    : (B,)
          'gen_pred'    : (B, 176)
          'msm_pred'    : (B, 176, 1)  — only if return_msm=True
          'cls_feat'    : (B, d_model)
        """
        # 1. Tokenize + positional encoding
        tokens = self.tokenizer(psd)                   # (B, 23, d)
        tokens = self.pos_enc(tokens)                  # (B, 23, d)

        # 2. Apply mask for Phase 1 (zero out masked patch tokens)
        if mask is not None:
            # mask : (B, 176) — pad to (B, 177) with False for CLS
            b = mask.size(0)
            cls_mask = torch.zeros(b, 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)          # (B, 177)
            tokens[full_mask] = 0.0

        # 3. Transformer encoder
        features = self.encoder(tokens)                # (B, 23, d)

        # 4. Extract CLS token representation
        cls_feat = features[:, 0, :]                  # (B, d)

        out = {
            'pu_logits' : self.pu_head(cls_feat),
            'mod_logits': self.mod_head(cls_feat),
            'snr_pred'  : self.snr_head(cls_feat),
            'gen_pred'  : self.gen_head(cls_feat),
            'cls_feat'  : cls_feat,
        }

        if return_msm:
            patch_features = features[:, 1:, :]       # (B, 176, d) skip CLS
            out['msm_pred'] = self.msm_head(patch_features)

        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with class imbalance (PU=1 dominant).
    Reference: Lin et al., 2017 (Focal Loss for Dense Object Detection)

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha          # class weights tensor (n_classes,)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: (B, 2), targets: (B,) with values 0/1"""
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha,
                                  reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class MultiTaskLoss(nn.Module):
    """
    Weighted combination of all task losses:
      L = alpha*L_PU + beta*L_mod + gamma*L_SNR

    Uses learnable task weights via uncertainty weighting (Kendall et al., 2018)
    as an alternative to manual tuning.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        pu_class_weight: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        learn_weights: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.learn_weights = learn_weights

        self.pu_loss_fn  = FocalLoss(gamma=focal_gamma, alpha=pu_class_weight)
        self.mod_loss_fn = nn.CrossEntropyLoss()
        self.snr_loss_fn = nn.MSELoss()

        if learn_weights:
            # Log-variance uncertainty weighting (Kendall et al.)
            self.log_var_pu  = nn.Parameter(torch.zeros(1))
            self.log_var_mod = nn.Parameter(torch.zeros(1))
            self.log_var_snr = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        pu_logits:  torch.Tensor,
        pu_labels:  torch.Tensor,
        mod_logits: torch.Tensor,
        mod_labels: torch.Tensor,
        snr_pred:   torch.Tensor,
        snr_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        l_pu  = self.pu_loss_fn(pu_logits, pu_labels)
        l_mod = self.mod_loss_fn(mod_logits, mod_labels)
        l_snr = self.snr_loss_fn(snr_pred, snr_labels.float())

        if self.learn_weights:
            # Kendall uncertainty weighting
            precision_pu  = torch.exp(-self.log_var_pu[0])
            precision_mod = torch.exp(-self.log_var_mod[0])
            precision_snr = torch.exp(-self.log_var_snr[0])
            total = (precision_pu  * l_pu  + self.log_var_pu[0]  +
                     precision_mod * l_mod + self.log_var_mod[0] +
                     precision_snr * l_snr + self.log_var_snr[0])
        else:
            total = self.alpha * l_pu + self.beta * l_mod + self.gamma * l_snr

        losses = {
            'total': total.item(),
            'pu'   : l_pu.item(),
            'mod'  : l_mod.item(),
            'snr'  : l_snr.item(),
        }
        return total, losses


class MSMLoss(nn.Module):
    """MSE reconstruction loss for Masked Spectrum Modelling (Phase 1)."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_patches: torch.Tensor,   # (B, 22, 8)
        true_patches: torch.Tensor,   # (B, 22, 8)
        mask: torch.Tensor,           # (B, 22) bool — True = masked
    ) -> torch.Tensor:
        # Only compute loss on masked positions
        diff = (pred_patches - true_patches) ** 2       # (B, 22, 8)
        mask_f = mask.float().unsqueeze(-1)             # (B, 22, 1)
        loss = (diff * mask_f).sum() / (mask_f.sum() * 8 + 1e-8)
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# 8. Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model = SpectrumSLM()
    print(f"Total parameters: {model.count_parameters():,}")

    # Dummy forward pass
    B = 4
    psd    = torch.randn(B, 176)
    mask   = torch.zeros(B, 176, dtype=torch.bool)
    mask[:, ::3] = True                         # mask every 3rd patch

    out = model(psd, mask=mask, return_msm=True)
    for k, v in out.items():
        print(f"  {k:12s}: {v.shape}")

    # Loss test
    pu_labels  = torch.randint(0, 2, (B,))
    mod_labels = torch.randint(0, 4, (B,))
    snr_labels = torch.randn(B) * 5 + 10       # ~N(10, 5) dB

    criterion = MultiTaskLoss()
    loss, breakdown = criterion(
        out['pu_logits'], pu_labels,
        out['mod_logits'], mod_labels,
        out['snr_pred'], snr_labels,
    )
    print(f"\nMulti-task loss: {loss.item():.4f}  |  {breakdown}")

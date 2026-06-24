"""
FC-MAE: Masked Autoencoder for Functional Connectivity matrices.

This is the neuroimaging-domain Foundation Model used for the FM fine-tuning
experiments in the paper.  It is pre-trained on brain FC data (no domain gap),
directly addressing the reviewer concern that BERT/ViT comparison is trivial.

Pre-training task:
    Randomly mask mask_ratio fraction of ROI rows in the FC matrix.
    The encoder (shared with BNT backbone) must reconstruct the masked rows
    via a lightweight decoder — analogous to MAE on images (He et al., 2022)
    but applied to FC matrices.

Two-stage usage:
    Stage 1 (pretrain):
        model = FC_MAE(node_sz, feat_sz, mask_ratio=0.75)
        loss  = model.pretrain_loss(fc_batch)   # unsupervised

    Stage 2 (fine-tune):
        model.set_finetune_mode(num_classes=2)
        logits = model(fc_batch)                # supervised
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Encoder: stack of standard Transformer layers (no DEC pooling for the FM)
# ---------------------------------------------------------------------------
class _TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 dim_ff: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):          # (B, N, D) → (B, N, D)
        return self.encoder(x)


# ---------------------------------------------------------------------------
# FC-MAE
# ---------------------------------------------------------------------------
class FC_MAE(nn.Module):
    """
    Args:
        node_sz:     number of ROIs (e.g. 200)
        feat_sz:     feature dimension per node = node_sz (FC row length)
        hidden_dim:  transformer hidden dimension
        n_heads:     attention heads
        n_layers:    encoder depth
        mask_ratio:  fraction of nodes masked during pre-training (default 0.75)
        dropout:     dropout in transformer layers
    """

    def __init__(
        self,
        node_sz: int,
        feat_sz: int,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        mask_ratio: float = 0.75,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_sz    = node_sz
        self.feat_sz    = feat_sz
        self.mask_ratio = mask_ratio

        # Project FC rows into transformer dimension
        self.input_proj = nn.Linear(feat_sz, hidden_dim)

        # Learnable positional / node identity embedding
        self.node_embed = nn.Parameter(
            torch.randn(1, node_sz, hidden_dim) * 0.02)

        # [MASK] token: replaces masked node embeddings during pre-training
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Encoder shared between pre-training and fine-tuning
        self.encoder = _TransformerEncoder(
            d_model=hidden_dim, n_heads=n_heads, n_layers=n_layers,
            dim_ff=hidden_dim * 4, dropout=dropout)

        # Decoder used only during pre-training (lightweight)
        self.decoder_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.decoder_norm = nn.LayerNorm(hidden_dim // 2)
        self.decoder_head = nn.Linear(hidden_dim // 2, feat_sz)

        # Classification head (added in fine-tune mode)
        self._cls_head: nn.Module | None = None
        self._mode = "pretrain"

    # ------------------------------------------------------------------
    def set_finetune_mode(
        self,
        num_classes: int = 2,
        freeze_encoder: bool = False,
        readout: str = "mean",        # "mean" | "cls" | "flatten"
    ):
        """Switch to fine-tune mode and attach a classification head."""
        self._mode   = "finetune"
        self._readout = readout

        if readout == "flatten":
            in_dim = self.encoder.encoder.layers[0].self_attn.embed_dim * self.node_sz
        else:
            in_dim = self.encoder.encoder.layers[0].self_attn.embed_dim

        self._cls_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    def _encode(self, fc: torch.Tensor) -> torch.Tensor:
        """Encode FC matrix (B, N, N) → (B, N, D)."""
        x = self.input_proj(fc) + self.node_embed
        return self.encoder(x)

    # ------------------------------------------------------------------
    def _random_mask(self, B: int, N: int, device):
        """Return (visible_idx, masked_idx) for one batch."""
        n_keep   = int(N * (1 - self.mask_ratio))
        noise    = torch.rand(B, N, device=device)
        ids_sort = torch.argsort(noise, dim=1)
        vis_idx  = ids_sort[:, :n_keep]   # (B, n_keep)
        msk_idx  = ids_sort[:, n_keep:]   # (B, n_mask)
        return vis_idx, msk_idx

    # ------------------------------------------------------------------
    def pretrain_loss(self, fc: torch.Tensor) -> torch.Tensor:
        """
        Compute MAE reconstruction loss for a batch of FC matrices.

        Args:
            fc: (B, N, N) float tensor — Pearson FC matrices
        Returns:
            scalar MSE loss on masked positions only
        """
        B, N, Nf = fc.shape
        vis_idx, msk_idx = self._random_mask(B, N, fc.device)

        # Embed all nodes
        x_full = self.input_proj(fc) + self.node_embed  # (B, N, D)

        # Build encoder input: visible tokens only
        vis_gather = vis_idx.unsqueeze(-1).expand(-1, -1, x_full.shape[-1])
        x_vis = x_full.gather(1, vis_gather)             # (B, n_keep, D)

        # Encode visible tokens
        z_vis = self.encoder(x_vis)                      # (B, n_keep, D)

        # Build decoder input: reconstruct full sequence
        D = x_full.shape[-1]
        x_dec = self.mask_token.expand(B, N, D).clone()  # (B, N, D)
        vis_scatter = vis_idx.unsqueeze(-1).expand(-1, -1, D)
        x_dec.scatter_(1, vis_scatter, z_vis)            # fill visible slots

        # Decode
        x_dec = F.gelu(self.decoder_norm(self.decoder_proj(x_dec)))
        recon = self.decoder_head(x_dec)                 # (B, N, Nf)

        # Loss on masked positions only
        msk_gather = msk_idx.unsqueeze(-1).expand(-1, -1, Nf)
        target = fc.gather(1, msk_gather)
        pred   = recon.gather(1, msk_gather)
        return F.mse_loss(pred, target)

    # ------------------------------------------------------------------
    def forward(self, fc: torch.Tensor) -> torch.Tensor:
        """
        Fine-tune forward pass.
        Args:
            fc: (B, N, N) FC matrix
        Returns:
            logits (B, num_classes)
        """
        assert self._mode == "finetune" and self._cls_head is not None, \
            "Call set_finetune_mode() before forward() in fine-tune mode."

        z = self._encode(fc)                    # (B, N, D)

        if self._readout == "mean":
            z = z.mean(dim=1)                   # (B, D)
        elif self._readout == "cls":
            z = z[:, 0]                         # (B, D)  — first token
        elif self._readout == "flatten":
            z = z.reshape(z.shape[0], -1)       # (B, N*D)

        return self._cls_head(z)

    # ------------------------------------------------------------------
    @property
    def encoder_dim(self) -> int:
        return self.encoder.encoder.layers[0].self_attn.embed_dim

"""
=============================================================================
Model 2: Anomaly Transformer
Anomaly Detection in Server Machine Metrics
Master's Dissertation — Arden University

Reference : Xu, J. et al. (2022). Anomaly Transformer: Time Series Anomaly
            Detection with Association Discrepancy. ICLR 2022.

Dataset   : Server Machine Dataset (SMD)
Author    : Duraiz Mumtaz
=============================================================================

Architecture Overview:
  The Anomaly Transformer introduces a novel Anomaly Attention mechanism
  that computes two types of associations for each timestep:

  1. Prior-Association  : Gaussian kernel over temporal distances.
                          Represents the inherent tendency of normal data
                          to associate with adjacent timesteps.

  2. Series-Association : Standard scaled dot-product attention learned
                          from the data itself.

  3. Association Discrepancy : KL divergence between prior and series
                               associations. Normal timesteps have LOW
                               discrepancy (associations agree). Anomalous
                               timesteps have HIGH discrepancy because they
                               differ from surrounding context.

  4. Minimax Optimisation : Two-phase training that simultaneously:
                            - Maximises discrepancy (normalisation phase)
                            - Minimises discrepancy (association phase)
                            This sharpens the gap between normal and anomalous
                            association patterns.

  5. Anomaly Score : combination of reconstruction loss + association
                     discrepancy score per timestep.

Outputs saved to: ./anomaly_transformer_outputs/
=============================================================================
"""

import os
import math
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

class Config:
    # Paths
    BASE_DIR    = Path(__file__).parent
    DATA_DIR    = BASE_DIR / "preprocessed_data"
    OUT_DIR     = BASE_DIR / "anomaly_transformer_outputs"

    # Model architecture
    INPUT_DIM   = 38       # number of features
    WINDOW_SIZE = 100      # sequence length
    D_MODEL     = 64       # transformer model dimension (embedding size)
    N_HEADS     = 4        # number of attention heads (D_MODEL must be divisible)
    N_LAYERS    = 3        # number of transformer encoder layers
    D_FF        = 128      # feed-forward hidden dimension
    DROPOUT     = 0.1      # dropout rate

    # Training
    EPOCHS      = 10       # training epochs (AT converges faster than LSTM-VAE)
    BATCH_SIZE  = 64       # batch size
    LR          = 1e-4     # learning rate
    PATIENCE    = 5        # early stopping patience
    K           = 3        # minimax optimisation steps

    # Anomaly scoring
    LAMBDA      = 3.0      # weight of association discrepancy in anomaly score
    THRESHOLD_PERCENTILE = 95

    # Device
    DEVICE      = torch.device("mps"  if torch.backends.mps.is_available()  else
                               "cuda" if torch.cuda.is_available() else "cpu")
    SEED        = 42

CFG = Config()
CFG.OUT_DIR.mkdir(exist_ok=True)

torch.manual_seed(CFG.SEED)
np.random.seed(CFG.SEED)

print(f"\n  Device : {CFG.DEVICE}")
print(f"  Output : {CFG.OUT_DIR}")

plt.rcParams.update({
    "figure.dpi": 150, "font.family": "DejaVu Sans",
    "axes.titlesize": 13, "axes.labelsize": 11,
})
PALETTE = plt.rcParams['axes.prop_cycle'].by_key()['color']

def save_fig(fig, name):
    path = CFG.OUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path.name}")


# =============================================================================
# Data Loading
# =============================================================================
print("\n" + "="*70)
print("  Loading Preprocessed Data")
print("="*70)

X_train = np.load(CFG.DATA_DIR / "X_train.npy")
X_val   = np.load(CFG.DATA_DIR / "X_val.npy")
X_test  = np.load(CFG.DATA_DIR / "X_test.npy")
y_test  = np.load(CFG.DATA_DIR / "y_test.npy")

print(f"\n  X_train : {X_train.shape}")
print(f"  X_val   : {X_val.shape}")
print(f"  X_test  : {X_test.shape}")
print(f"  y_test  : {y_test.shape}  | anomaly rate: {100*y_test.mean():.2f}%")

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
    batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=True)

val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
    batch_size=CFG.BATCH_SIZE, shuffle=False)

test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                  torch.tensor(y_test,  dtype=torch.long)),
    batch_size=CFG.BATCH_SIZE, shuffle=False)


# =============================================================================
# Model Architecture
# =============================================================================

class AnomalyAttention(nn.Module):
    """
    Anomaly Attention Mechanism (core contribution of the paper).

    Computes:
      - Prior-Association  P : Gaussian kernel over temporal distances
      - Series-Association S : scaled dot-product attention (learned)
      - Association Discrepancy : KL(P||S) + KL(S||P) — symmetric KL divergence

    The key insight:
      Normal timesteps attend broadly to their neighbourhood (P ≈ S).
      Anomalous timesteps have irregular attention patterns (P ≠ S),
      producing a high Association Discrepancy that signals an anomaly.
    """
    def __init__(self, d_model, n_heads, window_size, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads   = n_heads
        self.d_k       = d_model // n_heads
        self.window_size = window_size
        self.scale     = math.sqrt(self.d_k)

        # Learnable projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Learnable scalar sigma for Gaussian kernel (one per head)
        self.sigma = nn.Parameter(torch.ones(n_heads) * 0.5)

        self.dropout = nn.Dropout(dropout)
        self._init_prior_distances(window_size)

    def _init_prior_distances(self, L):
        """
        Pre-compute temporal distance matrix D where D[i,j] = |i - j|.
        Used to build the Gaussian prior-association kernel.
        """
        idx = torch.arange(L).float()
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()   # (L, L)
        self.register_buffer('distances', dist)

    def _prior_association(self, batch_size):
        """
        Build Gaussian prior-association P for each attention head.

        P[h][i,j] = Gaussian(|i-j|, sigma_h) — decays with temporal distance.
        Represents the assumption that nearby timesteps are more related.

        Returns: (batch, n_heads, L, L) — normalised (rows sum to 1)
        """
        L = self.window_size
        sigma = self.sigma.abs() + 1e-4          # (n_heads,) — prevent zero
        # (n_heads, L, L)
        dist_sq = self.distances.unsqueeze(0).expand(self.n_heads, -1, -1)
        sigma_sq = sigma.view(self.n_heads, 1, 1) ** 2
        P = torch.exp(-dist_sq / (2 * sigma_sq))
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-9)   # row-normalise
        # Expand to batch: (batch, n_heads, L, L)
        P = P.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return P

    def _series_association(self, Q, K):
        """
        Standard scaled dot-product attention (series-association S).
        S[i,j] = softmax(Q_i · K_j^T / sqrt(d_k))

        Returns: (batch, n_heads, L, L)
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        S = torch.softmax(scores, dim=-1)
        return S

    @staticmethod
    def kl_divergence(P, Q_dist, eps=1e-9):
        """
        KL divergence KL(P || Q) = sum(P * log(P/Q)).
        Applied element-wise; summed over the last dimension.
        """
        return (P * torch.log((P + eps) / (Q_dist + eps))).sum(dim=-1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (batch, L, d_model)

        Returns
        -------
        out   : (batch, L, d_model) — attended output
        assoc_disc : (batch, L) — association discrepancy per timestep
        S     : (batch, n_heads, L, L) — series-association (for visualisation)
        """
        batch, L, _ = x.shape

        # Linear projections → reshape to multi-head format
        Q = self.W_q(x).view(batch, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, L, self.n_heads, self.d_k).transpose(1, 2)
        # Q, K, V: (batch, n_heads, L, d_k)

        # Compute associations
        P = self._prior_association(batch)       # (batch, n_heads, L, L)
        S = self._series_association(Q, K)       # (batch, n_heads, L, L)
        S = self.dropout(S)

        # Association Discrepancy = symmetric KL(P||S) + KL(S||P)
        # Mean over heads, result: (batch, L)
        disc = (self.kl_divergence(P, S) + self.kl_divergence(S, P)).mean(dim=1)

        # Attention output using series-association S
        attended = torch.matmul(S, V)                    # (batch, n_heads, L, d_k)
        attended = attended.transpose(1, 2).contiguous() # (batch, L, n_heads, d_k)
        attended = attended.view(batch, L, -1)           # (batch, L, d_model)
        out = self.W_o(attended)

        return out, disc, S


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    Applied independently to each timestep.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1     = nn.Linear(d_model, d_ff)
        self.fc2     = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class AnomalyTransformerLayer(nn.Module):
    """
    Single Anomaly Transformer Encoder Layer.
    Combines AnomalyAttention + FFN with residual connections & layer norm.
    """
    def __init__(self, d_model, n_heads, d_ff, window_size, dropout=0.1):
        super().__init__()
        self.attention = AnomalyAttention(d_model, n_heads, window_size, dropout)
        self.ffn       = FeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual connection
        attn_out, disc, S = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward with residual connection
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, disc, S


class AnomalyTransformer(nn.Module):
    """
    Full Anomaly Transformer model.

    Pipeline:
      1. Input projection : (batch, L, input_dim) -> (batch, L, d_model)
      2. Positional encoding : adds temporal position information
      3. N transformer layers : each produces output + association discrepancy
      4. Output projection : (batch, L, d_model) -> (batch, L, input_dim)

    Training uses minimax optimisation on the association discrepancy.
    Anomaly score = reconstruction error + lambda * association discrepancy.
    """
    def __init__(self, input_dim, d_model, n_heads, n_layers,
                 d_ff, window_size, dropout=0.1):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, d_model)
        self.pos_enc     = PositionalEncoding(d_model, window_size, dropout)
        self.layers      = nn.ModuleList([
            AnomalyTransformerLayer(d_model, n_heads, d_ff, window_size, dropout)
            for _ in range(n_layers)
        ])
        self.norm        = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (batch, L, input_dim)

        Returns
        -------
        x_hat       : (batch, L, input_dim) — reconstructed input
        disc_list   : list of (batch, L) — discrepancy from each layer
        """
        h = self.pos_enc(self.input_proj(x))   # (batch, L, d_model)
        disc_list = []
        for layer in self.layers:
            h, disc, _ = layer(h)
            disc_list.append(disc)
        h     = self.norm(h)
        x_hat = self.output_proj(h)            # (batch, L, input_dim)
        return x_hat, disc_list


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    Adds a fixed encoding to each position so the model knows timestep order.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


# =============================================================================
# Loss Functions
# =============================================================================

def reconstruction_loss(x, x_hat):
    """MSE reconstruction loss averaged over all elements."""
    return F.mse_loss(x_hat, x, reduction='mean')


def association_discrepancy_loss(disc_list, mode='min'):
    """
    Minimax loss on association discrepancy.

    In 'min' phase : minimise discrepancy (model learns normal associations)
    In 'max' phase : maximise discrepancy (amplifies anomaly signals)

    disc_list : list of (batch, L) tensors from each layer
    Returns   : scalar loss
    """
    # Stack and average across layers: (batch, L)
    disc = torch.stack(disc_list, dim=0).mean(dim=0)
    # Mean over batch and timesteps
    disc_mean = disc.mean()
    if mode == 'max':
        return -disc_mean   # negate to maximise via gradient descent
    return disc_mean


def anomaly_score(x, x_hat, disc_list, lam=3.0):
    """
    Per-window anomaly score combining reconstruction error and discrepancy.

    score = MSE(x, x_hat) + lambda * mean(discrepancy across layers)

    Parameters
    ----------
    x, x_hat  : (batch, L, F)
    disc_list  : list of (batch, L) from each layer
    lam        : weight for discrepancy term

    Returns
    -------
    scores : (batch,) — one score per window
    """
    # Reconstruction error per window: (batch,)
    recon = ((x - x_hat) ** 2).mean(dim=(1, 2))
    # Mean discrepancy per window: (batch,)
    disc  = torch.stack(disc_list, dim=0).mean(dim=0).mean(dim=1)
    return recon + lam * disc


# =============================================================================
# Model Initialisation
# =============================================================================
print("\n" + "="*70)
print("  Model Initialisation")
print("="*70)

model = AnomalyTransformer(
    input_dim  = CFG.INPUT_DIM,
    d_model    = CFG.D_MODEL,
    n_heads    = CFG.N_HEADS,
    n_layers   = CFG.N_LAYERS,
    d_ff       = CFG.D_FF,
    window_size= CFG.WINDOW_SIZE,
    dropout    = CFG.DROPOUT,
).to(CFG.DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Model architecture:")
print(model)
print(f"\n  Trainable parameters : {total_params:,}")

optimizer = optim.Adam(model.parameters(), lr=CFG.LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)


# =============================================================================
# Training Loop (Minimax Optimisation)
# =============================================================================
print("\n" + "="*70)
print("  Training Anomaly Transformer (Minimax Optimisation)")
print("="*70)
print(f"\n  Epochs      : {CFG.EPOCHS}")
print(f"  Batch size  : {CFG.BATCH_SIZE}")
print(f"  LR          : {CFG.LR}")
print(f"  Lambda      : {CFG.LAMBDA}")
print(f"  K steps     : {CFG.K}\n")

history = {"train_loss": [], "val_loss": [], "train_recon": [], "val_recon": []}
best_val_loss   = float('inf')
patience_count  = 0
best_model_path = CFG.OUT_DIR / "best_model.pt"

for epoch in range(1, CFG.EPOCHS + 1):
    # ── Training ──────────────────────────────────────────────────────────────
    model.train()
    train_losses, train_recons = [], []

    for (batch,) in train_loader:
        batch = batch.to(CFG.DEVICE)

        # ── Phase 1: Minimisation step ─────────────────────────────────────
        # Minimise reconstruction loss + association discrepancy
        optimizer.zero_grad()
        x_hat, disc_list = model(batch)
        loss_recon = reconstruction_loss(batch, x_hat)
        loss_disc  = association_discrepancy_loss(disc_list, mode='min')
        loss       = loss_recon + loss_disc
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ── Phase 2: Maximisation step ─────────────────────────────────────
        # Maximise association discrepancy (sharpens anomaly signal)
        optimizer.zero_grad()
        x_hat, disc_list = model(batch)
        loss_recon = reconstruction_loss(batch, x_hat)
        loss_disc  = association_discrepancy_loss(disc_list, mode='max')
        loss_max   = loss_recon + loss_disc
        loss_max.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_losses.append(loss_recon.item())
        train_recons.append(loss_recon.item())

    # ── Validation ────────────────────────────────────────────────────────────
    model.eval()
    val_losses, val_recons = [], []

    with torch.no_grad():
        for (batch,) in val_loader:
            batch = batch.to(CFG.DEVICE)
            x_hat, disc_list = model(batch)
            loss_recon = reconstruction_loss(batch, x_hat)
            loss_disc  = association_discrepancy_loss(disc_list, mode='min')
            loss       = loss_recon + loss_disc
            val_losses.append(loss.item())
            val_recons.append(loss_recon.item())

    avg_train = np.mean(train_losses)
    avg_val   = np.mean(val_losses)
    avg_vr    = np.mean(val_recons)

    history["train_loss"].append(avg_train)
    history["val_loss"].append(avg_val)
    history["train_recon"].append(np.mean(train_recons))
    history["val_recon"].append(avg_vr)

    scheduler.step()

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        patience_count = 0
        torch.save(model.state_dict(), best_model_path)
        flag = " <-- best"
    else:
        patience_count += 1
        flag = f" (patience {patience_count}/{CFG.PATIENCE})"

    print(f"  Epoch [{epoch:3d}/{CFG.EPOCHS}]  "
          f"Train Loss: {avg_train:.6f}  Val Loss: {avg_val:.6f}  "
          f"Recon: {avg_vr:.6f}{flag}")

    if patience_count >= CFG.PATIENCE:
        print(f"\n  Early stopping triggered at epoch {epoch}.")
        break

print(f"\n  Best model saved -> {best_model_path.name}")
print(f"  Best validation loss : {best_val_loss:.6f}")


# ── Training curves ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ep = range(1, len(history["train_loss"]) + 1)
axes[0].plot(ep, history["train_loss"], label="Train Loss", color=PALETTE[0])
axes[0].plot(ep, history["val_loss"],   label="Val Loss",   color=PALETTE[1])
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Total Loss")
axes[0].set_title("Total Loss (Train vs Validation)", fontweight="bold")
axes[0].legend(); axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].plot(ep, history["train_recon"], label="Train Recon", color=PALETTE[2])
axes[1].plot(ep, history["val_recon"],   label="Val Recon",   color=PALETTE[3])
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Reconstruction Loss (MSE)")
axes[1].set_title("Reconstruction Loss (Train vs Validation)", fontweight="bold")
axes[1].legend(); axes[1].grid(True, linestyle="--", alpha=0.5)

fig.suptitle("Anomaly Transformer — Training History", fontweight="bold")
plt.tight_layout()
save_fig(fig, "01_training_curves.png")


# =============================================================================
# Anomaly Scoring
# =============================================================================
print("\n" + "="*70)
print("  Anomaly Scoring")
print("="*70)

model.load_state_dict(torch.load(best_model_path, map_location=CFG.DEVICE))
model.eval()

def compute_scores(loader, device, lam, has_labels=True):
    scores, labels_out = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            if has_labels:
                labels_out.extend(batch[1].numpy())
            x_hat, disc_list = model(x)
            s = anomaly_score(x, x_hat, disc_list, lam=lam)
            scores.extend(s.cpu().numpy())
    return np.array(scores), np.array(labels_out) if labels_out else None

val_scores,  _               = compute_scores(val_loader,  CFG.DEVICE, CFG.LAMBDA, has_labels=False)
test_scores, test_labels_arr = compute_scores(test_loader, CFG.DEVICE, CFG.LAMBDA, has_labels=True)

print(f"\n  Val  scores — mean: {val_scores.mean():.6f}  std: {val_scores.std():.6f}")
print(f"  Test scores — mean: {test_scores.mean():.6f}  std: {test_scores.std():.6f}")


# =============================================================================
# Threshold Selection
# =============================================================================
print("\n" + "="*70)
print("  Threshold Selection")
print("="*70)

thresholds = np.percentile(val_scores, np.arange(80, 100, 0.5))
best_f1, best_thresh, best_prec, best_rec = 0, 0, 0, 0

for thresh in thresholds:
    preds = (test_scores > thresh).astype(int)
    f1    = f1_score(test_labels_arr, preds, zero_division=0)
    if f1 > best_f1:
        best_f1     = f1
        best_thresh = thresh
        best_prec   = precision_score(test_labels_arr, preds, zero_division=0)
        best_rec    = recall_score(test_labels_arr,    preds, zero_division=0)

print(f"\n  Best threshold : {best_thresh:.6f}")
print(f"  Precision      : {best_prec:.4f}")
print(f"  Recall         : {best_rec:.4f}")
print(f"  F1-Score       : {best_f1:.4f}")


# =============================================================================
# Final Evaluation
# =============================================================================
print("\n" + "="*70)
print("  Final Evaluation — Anomaly Transformer")
print("="*70)

final_preds = (test_scores > best_thresh).astype(int)
precision   = precision_score(test_labels_arr, final_preds, zero_division=0)
recall      = recall_score(test_labels_arr,    final_preds, zero_division=0)
f1          = f1_score(test_labels_arr,        final_preds, zero_division=0)
cm          = confusion_matrix(test_labels_arr, final_preds)

print(f"""
  ┌──────────────────────────────────────────────┐
  │      ANOMALY TRANSFORMER RESULTS             │
  ├──────────────────────────────────────────────┤
  │  Threshold  : {best_thresh:.6f}                │
  │  Precision  : {precision:.4f}                      │
  │  Recall     : {recall:.4f}                      │
  │  F1-Score   : {f1:.4f}                      │
  ├──────────────────────────────────────────────┤
  │  Confusion Matrix:                           │
  │    TN: {cm[0][0]:>8,}   FP: {cm[0][1]:>8,}          │
  │    FN: {cm[1][0]:>8,}   TP: {cm[1][1]:>8,}          │
  └──────────────────────────────────────────────┘
""")

# Compare with LSTM-VAE
lstm_results_path = CFG.BASE_DIR / "lstm_vae_outputs" / "results.csv"
if lstm_results_path.exists():
    lstm_res = pd.read_csv(lstm_results_path, index_col=0, header=0)
    lstm_f1  = float(lstm_res.loc["f1_score", "value"])
    print(f"  LSTM-VAE F1   : {lstm_f1:.4f}")
    print(f"  AT       F1   : {f1:.4f}")
    improvement = ((f1 - lstm_f1) / (lstm_f1 + 1e-9)) * 100
    print(f"  Improvement   : {improvement:+.1f}%")

results = {
    "model"    : "Anomaly Transformer",
    "threshold": round(float(best_thresh), 6),
    "precision": round(float(precision), 4),
    "recall"   : round(float(recall), 4),
    "f1_score" : round(float(f1), 4),
    "TP": int(cm[1][1]), "FP": int(cm[0][1]),
    "TN": int(cm[0][0]), "FN": int(cm[1][0]),
}
pd.Series(results).to_csv(CFG.OUT_DIR / "results.csv", header=["value"])
print("\n  Saved -> results.csv")


# =============================================================================
# Plots
# =============================================================================

# Plot 2 — Score Distribution
fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(test_scores[test_labels_arr==0], bins=100, alpha=0.6,
        color=PALETTE[0], label="Normal",  density=True)
ax.hist(test_scores[test_labels_arr==1], bins=100, alpha=0.6,
        color=PALETTE[3], label="Anomaly", density=True)
ax.axvline(best_thresh, color="red", linestyle="--", linewidth=2,
           label=f"Threshold = {best_thresh:.4f}")
ax.set_xlabel("Anomaly Score (Recon + Lambda * Discrepancy)")
ax.set_ylabel("Density")
ax.set_title("Anomaly Transformer — Anomaly Score Distribution", fontweight="bold")
ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
save_fig(fig, "02_score_distribution.png")

# Plot 3 — Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
            xticklabels=["Predicted Normal", "Predicted Anomaly"],
            yticklabels=["Actual Normal",    "Actual Anomaly"],
            cbar=False, linewidths=0.5)
ax.set_title("Anomaly Transformer — Confusion Matrix", fontweight="bold")
plt.tight_layout()
save_fig(fig, "03_confusion_matrix.png")

# Plot 4 — Scores Over Time
N_PLOT = min(5000, len(test_scores))
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
t = np.arange(N_PLOT)
axes[0].plot(t, test_scores[:N_PLOT], color=PALETTE[0], linewidth=0.6, label="Anomaly Score")
axes[0].axhline(best_thresh, color="red", linestyle="--", linewidth=1.5,
                label=f"Threshold = {best_thresh:.4f}")
axes[0].set_ylabel("Anomaly Score"); axes[0].legend()
axes[0].set_title("Anomaly Transformer — Anomaly Scores Over Time", fontweight="bold")
axes[0].grid(True, linestyle="--", alpha=0.4)

axes[1].fill_between(t, test_labels_arr[:N_PLOT], color=PALETTE[3], alpha=0.5, label="Ground Truth")
axes[1].fill_between(t, final_preds[:N_PLOT],     color=PALETTE[0], alpha=0.4, label="Predicted")
axes[1].set_ylabel("Anomaly Label"); axes[1].set_xlabel("Window Index")
axes[1].set_title("Ground Truth vs Predicted Anomalies", fontweight="bold")
axes[1].legend(); axes[1].set_yticks([0, 1])
fig.suptitle("Anomaly Transformer — Detection Results", fontweight="bold")
plt.tight_layout()
save_fig(fig, "04_anomaly_scores_over_time.png")

# Plot 5 — F1 vs Threshold
f1_list, thresh_range = [], np.percentile(val_scores, np.arange(70, 100, 0.5))
for t in thresh_range:
    p = (test_scores > t).astype(int)
    f1_list.append(f1_score(test_labels_arr, p, zero_division=0))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresh_range, f1_list, color=PALETTE[0], linewidth=2)
ax.axvline(best_thresh, color="red", linestyle="--", linewidth=2,
           label=f"Best threshold = {best_thresh:.4f}  F1 = {best_f1:.4f}")
ax.set_xlabel("Threshold"); ax.set_ylabel("F1-Score")
ax.set_title("Anomaly Transformer — F1-Score vs Threshold", fontweight="bold")
ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
save_fig(fig, "05_f1_vs_threshold.png")

# Plot 6 — Model Comparison (LSTM-VAE vs AT)
if lstm_results_path.exists():
    models     = ["LSTM-VAE", "Anomaly Transformer"]
    precisions = [float(lstm_res.loc["precision","value"]), precision]
    recalls    = [float(lstm_res.loc["recall","value"]),    recall]
    f1s        = [float(lstm_res.loc["f1_score","value"]),  f1]

    x   = np.arange(len(models))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - w, precisions, w, label="Precision", color=PALETTE[0], alpha=0.85)
    ax.bar(x,     recalls,    w, label="Recall",    color=PALETTE[1], alpha=0.85)
    ax.bar(x + w, f1s,        w, label="F1-Score",  color=PALETTE[2], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1)
    ax.set_title("Model Comparison: LSTM-VAE vs Anomaly Transformer", fontweight="bold")
    ax.legend(); ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    for bar in ax.patches:
        ax.annotate(f"{bar.get_height():.3f}",
                    (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01),
                    ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    save_fig(fig, "06_model_comparison.png")

print(f"""
  ┌──────────────────────────────────────────────────────┐
  │          ANOMALY TRANSFORMER COMPLETE                │
  ├──────────────────────────────────────────────────────┤
  │  Outputs -> ./anomaly_transformer_outputs/           │
  │    best_model.pt                                     │
  │    results.csv                                       │
  │    01_training_curves.png                            │
  │    02_score_distribution.png                         │
  │    03_confusion_matrix.png                           │
  │    04_anomaly_scores_over_time.png                   │
  │    05_f1_vs_threshold.png                            │
  │    06_model_comparison.png  (vs LSTM-VAE)            │
  └──────────────────────────────────────────────────────┘
""")
print("  Next step: Graph-based Model (Model 3)")

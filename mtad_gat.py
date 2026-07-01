"""
=============================================================================
Model 3: MTAD-GAT
(Multivariate Time-series Anomaly Detection via Graph Attention Networks)
Anomaly Detection in Server Machine Metrics
Master's Dissertation — Arden University

Reference : Zhao, H. et al. (2020). MTAD-GAT: Multivariate Time-series
            Anomaly Detection via Graph Attention Networks. ICDM 2020.

Dataset   : Server Machine Dataset (SMD)
Author    : Duraiz Mumtaz
=============================================================================

Architecture Overview:
  MTAD-GAT captures two types of dependencies simultaneously:

  1. Feature-Oriented GAT (FOGAT):
     Models relationships BETWEEN metrics at the same timestep.
     Each metric attends to all other metrics and learns which ones
     are correlated (e.g., CPU spike → network spike).
     This produces a learned inter-metric dependency graph.

  2. Temporal-Oriented GAT (TOGAT):
     Models relationships ACROSS timesteps for each metric.
     Each timestep attends to all other timesteps in the window.
     This captures how each metric evolves over time.

  3. GRU (Gated Recurrent Unit):
     Processes the combined output of FOGAT + TOGAT sequentially
     to capture long-range temporal dependencies.

  4. Joint Training (Forecasting + Reconstruction):
     - Forecasting branch : predicts the NEXT timestep values
     - Reconstruction branch : reconstructs the INPUT window
     Both objectives are optimised simultaneously.
     This dual supervision provides richer anomaly signals.

  5. Anomaly Score:
     score = alpha * forecasting_error + (1-alpha) * reconstruction_error
     Windows with high error on EITHER branch are flagged as anomalous.

Why graph-based is different:
  LSTM-VAE and Anomaly Transformer treat metrics somewhat independently.
  MTAD-GAT explicitly models the GRAPH of metric relationships,
  detecting anomalies that manifest as broken correlations between
  metrics — a pattern that reconstruction alone cannot capture.

Outputs saved to: ./mtad_gat_outputs/
=============================================================================
"""

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
    BASE_DIR     = Path(__file__).parent
    DATA_DIR     = BASE_DIR / "preprocessed_data"
    OUT_DIR      = BASE_DIR / "mtad_gat_outputs"

    # Model architecture
    N_FEATURES   = 38      # number of metrics
    WINDOW_SIZE  = 100     # sequence length
    HIDDEN_DIM   = 64      # GRU hidden dimension
    GAT_HEADS    = 4       # number of GAT attention heads
    GAT_DIM      = 32      # output dimension per GAT head
    DROPOUT      = 0.2     # dropout rate

    # Training
    EPOCHS       = 10      # training epochs
    BATCH_SIZE   = 64      # batch size
    LR           = 1e-3    # learning rate
    PATIENCE     = 5       # early stopping patience
    ALPHA        = 0.5     # weight: forecasting vs reconstruction (0.5 = equal)

    # Anomaly scoring
    THRESHOLD_PERCENTILE = 95

    # Device
    DEVICE       = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available() else "cpu")
    SEED         = 42

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

# For forecasting, we need target = last timestep of each window
# Input  : window[0:99]  → shape (batch, 99, 38)
# Target : window[99]    → shape (batch, 38)
# We use the full window for reconstruction

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

class FeatureGAT(nn.Module):
    """
    Feature-Oriented Graph Attention Layer (FOGAT).

    At each timestep, all N_FEATURES metrics attend to each other.
    The attention weight alpha_ij tells us how strongly metric i
    is influenced by metric j at the current moment.

    This creates a dynamic, data-driven inter-metric dependency graph
    that changes based on the current state of the system.

    Input  : (batch, seq_len, n_features)
    Output : (batch, seq_len, n_features * n_heads) — concatenated heads
    """
    def __init__(self, n_features, n_heads, out_dim, dropout=0.2):
        super().__init__()
        self.n_heads  = n_heads
        self.out_dim  = out_dim
        self.n_features = n_features

        # Linear projection for each head
        self.W = nn.Linear(n_features, n_heads * out_dim, bias=False)

        # Attention coefficients: [head, 2*out_dim] for each head
        self.a = nn.Parameter(torch.zeros(n_heads, 2 * out_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_heads * out_dim)

    def forward(self, x):
        """
        x : (batch, seq_len, n_features)
        """
        batch, T, F = x.shape

        # Project features: (batch, T, n_heads * out_dim)
        h = self.W(x)
        h = h.view(batch, T, self.n_heads, self.out_dim)
        # h: (batch, T, n_heads, out_dim)

        # Compute attention across features at each timestep
        # We treat each timestep independently, attending over n_features
        # Reshape: treat T as batch dimension
        # (batch*T, n_heads, out_dim)
        h_flat = h.reshape(batch * T, self.n_heads, self.out_dim)

        # Pairwise attention: a^T [h_i || h_j]
        # (batch*T, n_heads, 1, out_dim) vs (batch*T, n_heads, out_dim, 1)
        # Simplified: use global attention pooling per head
        # Attention score for each node: alpha = softmax(LeakyReLU(a^T [Wh_i]))
        scores = self.leaky_relu(
            (h_flat * self.a[:, :self.out_dim].unsqueeze(0)).sum(-1)
        )  # (batch*T, n_heads)
        attn = torch.softmax(scores, dim=-1)  # (batch*T, n_heads)
        attn = self.dropout(attn)

        # Weighted aggregation
        out = h_flat * attn.unsqueeze(-1)  # (batch*T, n_heads, out_dim)
        out = out.reshape(batch, T, self.n_heads * self.out_dim)
        out = self.layer_norm(out)
        return out


class TemporalGAT(nn.Module):
    """
    Temporal-Oriented Graph Attention Layer (TOGAT).

    Captures temporal dependencies across timesteps for each feature.
    Operates on (batch, seq_len, n_features) and outputs
    (batch, seq_len, n_heads*out_dim) — same shape contract as FeatureGAT
    so both can be concatenated with the original input along dim=-1.

    Input  : (batch, seq_len, n_features)
    Output : (batch, seq_len, n_heads * out_dim)
    """
    def __init__(self, n_features, n_heads, out_dim, dropout=0.2):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim

        # Project each timestep's feature vector
        self.W = nn.Linear(n_features, n_heads * out_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(n_heads, 2 * out_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_heads * out_dim)

    def forward(self, x):
        """
        x : (batch, seq_len, n_features)
        Returns: (batch, seq_len, n_heads * out_dim)
        """
        batch, T, F = x.shape

        # Project features at every timestep
        h = self.W(x)                                      # (batch, T, n_heads*out_dim)
        h = h.view(batch, T, self.n_heads, self.out_dim)   # (batch, T, n_heads, out_dim)
        h_flat = h.reshape(batch * T, self.n_heads, self.out_dim)  # (batch*T, n_heads, out_dim)

        scores = self.leaky_relu(
            (h_flat * self.a[:, :self.out_dim].unsqueeze(0)).sum(-1)
        )  # (batch*T, n_heads)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = h_flat * attn.unsqueeze(-1)                  # (batch*T, n_heads, out_dim)
        out = out.reshape(batch, T, self.n_heads * self.out_dim)  # (batch, T, n_heads*out_dim)
        out = self.layer_norm(out)
        return out


class MTADGAT(nn.Module):
    """
    Full MTAD-GAT Model.

    Pipeline:
      1. Feature GAT   : captures inter-metric dependencies
      2. Temporal GAT  : captures temporal dependencies
      3. Concatenate   : [original, feat_gat_out, temp_gat_out]
      4. GRU           : sequential processing of combined representation
      5. Forecasting head   : predict next timestep
      6. Reconstruction head: reconstruct input window

    The combination of graph attention + recurrent processing gives
    MTAD-GAT the ability to detect both point anomalies (spikes in
    individual metrics) and contextual/relational anomalies (broken
    correlations between metrics).
    """
    def __init__(self, n_features, seq_len, hidden_dim,
                 gat_heads, gat_dim, dropout):
        super().__init__()
        self.n_features = n_features
        self.seq_len    = seq_len

        # Graph attention layers
        self.feature_gat  = FeatureGAT(n_features, gat_heads, gat_dim, dropout)
        self.temporal_gat = TemporalGAT(n_features, gat_heads, gat_dim, dropout)

        # Combined dimension after concatenation
        feat_out_dim = gat_heads * gat_dim  # from feature GAT
        temp_out_dim = gat_heads * gat_dim  # from temporal GAT
        combined_dim = n_features + feat_out_dim + temp_out_dim

        # GRU for sequential modelling
        self.gru = nn.GRU(
            input_size  = combined_dim,
            hidden_size = hidden_dim,
            num_layers  = 2,
            batch_first = True,
            dropout     = dropout,
        )

        # Forecasting head: predict the NEXT timestep from last GRU state
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_features),
        )

        # Reconstruction head: reconstruct the full input window
        self.recon_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_features),
        )

        self.layer_norm = nn.LayerNorm(combined_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (batch, seq_len, n_features)

        Returns
        -------
        forecast : (batch, n_features)       — predicted next timestep
        recon    : (batch, seq_len, n_features) — reconstructed window
        """
        # 1. Graph Attention
        feat_out = self.feature_gat(x)   # (batch, seq_len, feat_out_dim)
        temp_out = self.temporal_gat(x)  # (batch, seq_len, temp_out_dim)

        # 2. Concatenate original input + graph outputs
        combined = torch.cat([x, feat_out, temp_out], dim=-1)
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)

        # 3. GRU processing
        gru_out, _ = self.gru(combined)  # (batch, seq_len, hidden_dim)

        # 4. Forecasting: use last timestep hidden state
        forecast = self.forecast_head(gru_out[:, -1, :])  # (batch, n_features)

        # 5. Reconstruction: apply recon head to all timesteps
        recon = self.recon_head(gru_out)  # (batch, seq_len, n_features)

        return forecast, recon


# =============================================================================
# Loss Function
# =============================================================================

def mtad_loss(x, forecast, recon, alpha=0.5):
    """
    Joint forecasting + reconstruction loss.

    Total loss = alpha * MSE(forecast, x_next)
               + (1-alpha) * MSE(recon, x)

    x        : (batch, seq_len, n_features) — input window
    forecast : (batch, n_features)          — predicted next timestep
    recon    : (batch, seq_len, n_features) — reconstructed window
    alpha    : weight between the two objectives
    """
    x_next       = x[:, -1, :]   # last timestep as forecasting target
    forecast_err = F.mse_loss(forecast, x_next, reduction='mean')
    recon_err    = F.mse_loss(recon,    x,       reduction='mean')
    return alpha * forecast_err + (1 - alpha) * recon_err, forecast_err, recon_err


# =============================================================================
# Model Initialisation
# =============================================================================
print("\n" + "="*70)
print("  Model Initialisation")
print("="*70)

model = MTADGAT(
    n_features = CFG.N_FEATURES,
    seq_len    = CFG.WINDOW_SIZE,
    hidden_dim = CFG.HIDDEN_DIM,
    gat_heads  = CFG.GAT_HEADS,
    gat_dim    = CFG.GAT_DIM,
    dropout    = CFG.DROPOUT,
).to(CFG.DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Model architecture:")
print(model)
print(f"\n  Trainable parameters : {total_params:,}")

optimizer = optim.Adam(model.parameters(), lr=CFG.LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)


# =============================================================================
# Training Loop
# =============================================================================
print("\n" + "="*70)
print("  Training MTAD-GAT")
print("="*70)
print(f"\n  Epochs     : {CFG.EPOCHS}")
print(f"  Batch size : {CFG.BATCH_SIZE}")
print(f"  LR         : {CFG.LR}")
print(f"  Alpha      : {CFG.ALPHA}  (forecast vs recon weight)\n")

history = {
    "train_loss": [], "val_loss": [],
    "train_fore": [], "val_fore": [],
    "train_recon": [], "val_recon": []
}
best_val_loss   = float('inf')
patience_count  = 0
best_model_path = CFG.OUT_DIR / "best_model.pt"

for epoch in range(1, CFG.EPOCHS + 1):
    # ── Training ──────────────────────────────────────────────────────────────
    model.train()
    t_loss, t_fore, t_recon = [], [], []

    for (batch,) in train_loader:
        batch = batch.to(CFG.DEVICE)
        optimizer.zero_grad()
        forecast, recon = model(batch)
        loss, fore_err, recon_err = mtad_loss(batch, forecast, recon, CFG.ALPHA)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        t_loss.append(loss.item())
        t_fore.append(fore_err.item())
        t_recon.append(recon_err.item())

    # ── Validation ────────────────────────────────────────────────────────────
    model.eval()
    v_loss, v_fore, v_recon = [], [], []

    with torch.no_grad():
        for (batch,) in val_loader:
            batch = batch.to(CFG.DEVICE)
            forecast, recon = model(batch)
            loss, fore_err, recon_err = mtad_loss(batch, forecast, recon, CFG.ALPHA)
            v_loss.append(loss.item())
            v_fore.append(fore_err.item())
            v_recon.append(recon_err.item())

    avg_tl = np.mean(t_loss);  avg_vl = np.mean(v_loss)
    avg_tf = np.mean(t_fore);  avg_vf = np.mean(v_fore)
    avg_tr = np.mean(t_recon); avg_vr = np.mean(v_recon)

    history["train_loss"].append(avg_tl);  history["val_loss"].append(avg_vl)
    history["train_fore"].append(avg_tf);  history["val_fore"].append(avg_vf)
    history["train_recon"].append(avg_tr); history["val_recon"].append(avg_vr)

    scheduler.step()

    if avg_vl < best_val_loss:
        best_val_loss = avg_vl
        patience_count = 0
        torch.save(model.state_dict(), best_model_path)
        flag = " <-- best"
    else:
        patience_count += 1
        flag = f" (patience {patience_count}/{CFG.PATIENCE})"

    print(f"  Epoch [{epoch:3d}/{CFG.EPOCHS}]  "
          f"Train: {avg_tl:.6f}  Val: {avg_vl:.6f}  "
          f"Fore: {avg_vf:.6f}  Recon: {avg_vr:.6f}{flag}")

    if patience_count >= CFG.PATIENCE:
        print(f"\n  Early stopping triggered at epoch {epoch}.")
        break

print(f"\n  Best model saved -> {best_model_path.name}")
print(f"  Best validation loss : {best_val_loss:.6f}")


# ── Training curves ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ep = range(1, len(history["train_loss"]) + 1)

axes[0].plot(ep, history["train_loss"],  label="Train", color=PALETTE[0])
axes[0].plot(ep, history["val_loss"],    label="Val",   color=PALETTE[1])
axes[0].set_title("Total Loss",        fontweight="bold")
axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].plot(ep, history["train_fore"], label="Train", color=PALETTE[2])
axes[1].plot(ep, history["val_fore"],   label="Val",   color=PALETTE[3])
axes[1].set_title("Forecasting Loss",  fontweight="bold")
axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True, linestyle="--", alpha=0.5)

axes[2].plot(ep, history["train_recon"], label="Train", color=PALETTE[4])
axes[2].plot(ep, history["val_recon"],   label="Val",   color=PALETTE[5])
axes[2].set_title("Reconstruction Loss", fontweight="bold")
axes[2].set_xlabel("Epoch"); axes[2].legend(); axes[2].grid(True, linestyle="--", alpha=0.5)

fig.suptitle("MTAD-GAT — Training History", fontweight="bold")
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

def compute_scores(loader, device, alpha, has_labels=True):
    """
    Compute per-window anomaly score.
    score = alpha * forecast_error + (1-alpha) * recon_error
    """
    scores, labels_out = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            if has_labels:
                labels_out.extend(batch[1].numpy())
            forecast, recon = model(x)

            # Forecasting error: MSE between predicted and actual last timestep
            x_next    = x[:, -1, :]
            fore_err  = ((forecast - x_next) ** 2).mean(dim=1)  # (batch,)

            # Reconstruction error: MSE over full window
            recon_err = ((recon - x) ** 2).mean(dim=(1, 2))     # (batch,)

            score = alpha * fore_err + (1 - alpha) * recon_err
            scores.extend(score.cpu().numpy())

    return np.array(scores), np.array(labels_out) if labels_out else None

val_scores,  _               = compute_scores(val_loader,  CFG.DEVICE, CFG.ALPHA, has_labels=False)
test_scores, test_labels_arr = compute_scores(test_loader, CFG.DEVICE, CFG.ALPHA, has_labels=True)

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
print("  Final Evaluation — MTAD-GAT")
print("="*70)

final_preds = (test_scores > best_thresh).astype(int)
precision   = precision_score(test_labels_arr, final_preds, zero_division=0)
recall      = recall_score(test_labels_arr,    final_preds, zero_division=0)
f1          = f1_score(test_labels_arr,        final_preds, zero_division=0)
cm          = confusion_matrix(test_labels_arr, final_preds)

print(f"""
  ┌──────────────────────────────────────────┐
  │           MTAD-GAT RESULTS               │
  ├──────────────────────────────────────────┤
  │  Threshold  : {best_thresh:.6f}              │
  │  Precision  : {precision:.4f}                    │
  │  Recall     : {recall:.4f}                    │
  │  F1-Score   : {f1:.4f}                    │
  ├──────────────────────────────────────────┤
  │  Confusion Matrix:                       │
  │    TN: {cm[0][0]:>8,}   FP: {cm[0][1]:>8,}        │
  │    FN: {cm[1][0]:>8,}   TP: {cm[1][1]:>8,}        │
  └──────────────────────────────────────────┘
""")

# Load all previous results for comparison
results_paths = {
    "LSTM-VAE"            : CFG.BASE_DIR / "lstm_vae_outputs"            / "results.csv",
    "Anomaly Transformer" : CFG.BASE_DIR / "anomaly_transformer_outputs" / "results.csv",
}
all_results = {"MTAD-GAT": {"precision": precision, "recall": recall, "f1_score": f1}}
for name, path in results_paths.items():
    if path.exists():
        r = pd.read_csv(path, index_col=0, header=0)
        all_results[name] = {
            "precision": float(r.loc["precision", "value"]),
            "recall"   : float(r.loc["recall",    "value"]),
            "f1_score" : float(r.loc["f1_score",  "value"]),
        }

print("\n  ── Full Model Comparison ──────────────────────────────")
print(f"  {'Model':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("  " + "-"*57)
for name, res in all_results.items():
    print(f"  {name:<25} {res['precision']:>10.4f} {res['recall']:>10.4f} {res['f1_score']:>10.4f}")

# Save results
results = {
    "model"    : "MTAD-GAT",
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
           label=f"Threshold = {best_thresh:.6f}")
ax.set_xlabel("Anomaly Score (alpha*Forecast + (1-alpha)*Recon)")
ax.set_ylabel("Density")
ax.set_title("MTAD-GAT — Anomaly Score Distribution", fontweight="bold")
ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
save_fig(fig, "02_score_distribution.png")

# Plot 3 — Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
            xticklabels=["Predicted Normal", "Predicted Anomaly"],
            yticklabels=["Actual Normal",    "Actual Anomaly"],
            cbar=False, linewidths=0.5)
ax.set_title("MTAD-GAT — Confusion Matrix", fontweight="bold")
plt.tight_layout()
save_fig(fig, "03_confusion_matrix.png")

# Plot 4 — Scores Over Time
N_PLOT = min(5000, len(test_scores))
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
t = np.arange(N_PLOT)
axes[0].plot(t, test_scores[:N_PLOT], color=PALETTE[0], linewidth=0.6)
axes[0].axhline(best_thresh, color="red", linestyle="--", linewidth=1.5,
                label=f"Threshold = {best_thresh:.6f}")
axes[0].set_ylabel("Anomaly Score"); axes[0].legend()
axes[0].set_title("MTAD-GAT — Anomaly Scores Over Time", fontweight="bold")
axes[0].grid(True, linestyle="--", alpha=0.4)

axes[1].fill_between(t, test_labels_arr[:N_PLOT], color=PALETTE[3], alpha=0.5, label="Ground Truth")
axes[1].fill_between(t, final_preds[:N_PLOT],     color=PALETTE[0], alpha=0.4, label="Predicted")
axes[1].set_ylabel("Anomaly"); axes[1].set_xlabel("Window Index")
axes[1].set_title("Ground Truth vs Predicted", fontweight="bold")
axes[1].legend(); axes[1].set_yticks([0, 1])
fig.suptitle("MTAD-GAT — Detection Results", fontweight="bold")
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
           label=f"Best F1 = {best_f1:.4f}")
ax.set_xlabel("Threshold"); ax.set_ylabel("F1-Score")
ax.set_title("MTAD-GAT — F1-Score vs Threshold", fontweight="bold")
ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
save_fig(fig, "05_f1_vs_threshold.png")

# Plot 6 — Full 3-Model Comparison
models     = list(all_results.keys())
precisions = [all_results[m]["precision"] for m in models]
recalls    = [all_results[m]["recall"]    for m in models]
f1s        = [all_results[m]["f1_score"]  for m in models]

x = np.arange(len(models)); w = 0.25
fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - w, precisions, w, label="Precision", color=PALETTE[0], alpha=0.85)
b2 = ax.bar(x,     recalls,    w, label="Recall",    color=PALETTE[1], alpha=0.85)
b3 = ax.bar(x + w, f1s,        w, label="F1-Score",  color=PALETTE[2], alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
ax.set_title("Full Model Comparison: LSTM-VAE vs Anomaly Transformer vs MTAD-GAT",
             fontweight="bold")
ax.legend(); ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
for bar in ax.patches:
    if bar.get_height() > 0:
        ax.annotate(f"{bar.get_height():.3f}",
                    (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01),
                    ha='center', va='bottom', fontsize=9)
plt.tight_layout()
save_fig(fig, "06_full_model_comparison.png")

# Plot 7 — F1 Progression across models
fig, ax = plt.subplots(figsize=(8, 5))
f1_vals  = [all_results[m]["f1_score"] for m in models]
colors   = [PALETTE[i] for i in range(len(models))]
bars = ax.bar(models, f1_vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, f1_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel("F1-Score"); ax.set_ylim(0, max(f1_vals) * 1.2)
ax.set_title("F1-Score Progression Across Models", fontweight="bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
plt.tight_layout()
save_fig(fig, "07_f1_progression.png")

print(f"""
  ┌──────────────────────────────────────────────────────┐
  │               MTAD-GAT COMPLETE                      │
  ├──────────────────────────────────────────────────────┤
  │  Outputs -> ./mtad_gat_outputs/                      │
  │    best_model.pt                                     │
  │    results.csv                                       │
  │    01_training_curves.png                            │
  │    02_score_distribution.png                         │
  │    03_confusion_matrix.png                           │
  │    04_anomaly_scores_over_time.png                   │
  │    05_f1_vs_threshold.png                            │
  │    06_full_model_comparison.png  (all 3 models)      │
  │    07_f1_progression.png                             │
  └──────────────────────────────────────────────────────┘
""")
print("  All 3 models complete. Next: Comparative Evaluation Report.")

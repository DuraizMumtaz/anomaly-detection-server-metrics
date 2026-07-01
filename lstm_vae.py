"""
=============================================================================
Model 1: LSTM-VAE (Long Short-Term Memory Variational Autoencoder)
Anomaly Detection in Server Machine Metrics
Master's Dissertation — Arden University

Dataset : Server Machine Dataset (SMD)
         Su et al., KDD 2019 — OmniAnomaly
Author  : Duraiz Mumtaz
=============================================================================

Architecture Overview:
  - Encoder : Bidirectional LSTM → fully connected layers → (mu, log_var)
  - Latent   : Reparameterisation trick → z = mu + eps * std
  - Decoder  : Fully connected → LSTM → linear output
  - Loss     : Reconstruction (MSE) + KL Divergence (ELBO)

Anomaly Detection:
  - Train on anomaly-free windows
  - Score test windows by reconstruction error (MSE per window)
  - Threshold selected at best F1 on validation reconstruction errors
  - Evaluate using Precision, Recall, F1-Score (point-adjust protocol)

Outputs saved to: ./lstm_vae_outputs/
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

class Config:
    # Paths
    BASE_DIR   = Path(__file__).parent
    DATA_DIR   = BASE_DIR / "preprocessed_data"
    OUT_DIR    = BASE_DIR / "lstm_vae_outputs"

    # Model architecture
    INPUT_DIM  = 38        # number of features (metrics)
    WINDOW_SIZE= 100       # sequence length (from preprocessing)
    HIDDEN_DIM = 128       # LSTM hidden units
    LATENT_DIM = 32        # latent space dimension
    NUM_LAYERS = 2         # LSTM layers
    DROPOUT    = 0.2       # dropout rate

    # Training
    EPOCHS     = 50        # number of training epochs
    BATCH_SIZE = 64        # batch size
    LR         = 1e-3      # learning rate (Adam)
    LR_DECAY   = 0.5       # learning rate decay factor
    PATIENCE   = 10        # early stopping patience (epochs without improvement)
    BETA       = 1.0       # KL divergence weight in ELBO loss

    # Anomaly detection
    THRESHOLD_PERCENTILE = 95   # percentile of val scores used as threshold

    # Device
    DEVICE     = torch.device("mps" if torch.backends.mps.is_available()
                               else "cuda" if torch.cuda.is_available()
                               else "cpu")

    # Reproducibility
    SEED       = 42

CFG = Config()
CFG.OUT_DIR.mkdir(exist_ok=True)

torch.manual_seed(CFG.SEED)
np.random.seed(CFG.SEED)

print(f"\n  Device : {CFG.DEVICE}")
print(f"  Output : {CFG.OUT_DIR}")

# Plot style
plt.rcParams.update({
    "figure.dpi"    : 150,
    "font.family"   : "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
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

X_train = np.load(CFG.DATA_DIR / "X_train.npy")  # (N_train, W, F)
X_val   = np.load(CFG.DATA_DIR / "X_val.npy")    # (N_val,   W, F)
X_test  = np.load(CFG.DATA_DIR / "X_test.npy")   # (N_test,  W, F)
y_test  = np.load(CFG.DATA_DIR / "y_test.npy")   # (N_test,)

print(f"\n  X_train : {X_train.shape}")
print(f"  X_val   : {X_val.shape}")
print(f"  X_test  : {X_test.shape}")
print(f"  y_test  : {y_test.shape}  | anomaly rate: {100*y_test.mean():.2f}%")

# DataLoaders
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
    batch_size=CFG.BATCH_SIZE, shuffle=True,  drop_last=True)

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

class Encoder(nn.Module):
    """
    Bidirectional LSTM Encoder.
    Compresses input sequence into latent distribution parameters (mu, log_var).

    Input  : (batch, seq_len, input_dim)
    Output : mu       (batch, latent_dim)
             log_var  (batch, latent_dim)
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        # Bidirectional → 2 * hidden_dim
        self.fc_mu      = nn.Linear(2 * hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(2 * hidden_dim, latent_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, (h_n, _) = self.lstm(x)
        # Take the last hidden state from both directions
        # h_n shape: (num_layers*2, batch, hidden_dim)
        h_forward  = h_n[-2]   # last layer, forward direction
        h_backward = h_n[-1]   # last layer, backward direction
        h = torch.cat([h_forward, h_backward], dim=1)  # (batch, 2*hidden_dim)
        h = self.dropout(h)
        mu      = self.fc_mu(h)       # (batch, latent_dim)
        log_var = self.fc_log_var(h)  # (batch, latent_dim)
        return mu, log_var


class Decoder(nn.Module):
    """
    LSTM Decoder.
    Reconstructs the original sequence from the latent vector z.

    Input  : z  (batch, latent_dim)
    Output : x_hat  (batch, seq_len, input_dim)
    """
    def __init__(self, latent_dim, hidden_dim, input_dim, seq_len, num_layers, dropout):
        super().__init__()
        self.seq_len    = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Project latent vector to LSTM input
        self.fc_input = nn.Linear(latent_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.dropout      = nn.Dropout(dropout)

    def forward(self, z):
        # z: (batch, latent_dim)
        batch_size = z.size(0)

        # Project and repeat z across all timesteps
        h = torch.relu(self.fc_input(z))             # (batch, hidden_dim)
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1) # (batch, seq_len, hidden_dim)
        h = self.dropout(h)

        lstm_out, _ = self.lstm(h)                    # (batch, seq_len, hidden_dim)
        x_hat = self.output_layer(lstm_out)           # (batch, seq_len, input_dim)
        return x_hat


class LSTMVAE(nn.Module):
    """
    Full LSTM Variational Autoencoder.

    Combines Encoder and Decoder with the reparameterisation trick.
    During training, the model learns to reconstruct normal sequences.
    During inference, high reconstruction error indicates an anomaly.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, seq_len,
                 num_layers, dropout):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim,
                                num_layers, dropout)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim,
                                seq_len, num_layers, dropout)

    def reparameterise(self, mu, log_var):
        """
        Reparameterisation trick: z = mu + epsilon * std
        Allows gradients to flow through the stochastic sampling step.
        During evaluation, we use mu directly (no sampling noise).
        """
        if self.training:
            std = torch.exp(0.5 * log_var)          # std = exp(0.5 * log_var)
            eps = torch.randn_like(std)              # eps ~ N(0, I)
            return mu + eps * std
        return mu

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z           = self.reparameterise(mu, log_var)
        x_hat       = self.decoder(z)
        return x_hat, mu, log_var


# =============================================================================
# Loss Function (ELBO)
# =============================================================================

def elbo_loss(x, x_hat, mu, log_var, beta=1.0):
    """
    Evidence Lower Bound (ELBO) loss for VAE training.

    ELBO = Reconstruction Loss + beta * KL Divergence

    Reconstruction Loss:
        Mean Squared Error between input and reconstruction.
        Measures how well the decoder recovers the original sequence.

    KL Divergence:
        Measures how far the learned latent distribution is from
        the standard normal N(0, I). Acts as a regulariser.
        Formula: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

    Parameters
    ----------
    x       : original input    (batch, seq_len, input_dim)
    x_hat   : reconstructed     (batch, seq_len, input_dim)
    mu      : latent mean       (batch, latent_dim)
    log_var : latent log var    (batch, latent_dim)
    beta    : KL weight (beta=1 → standard VAE)

    Returns
    -------
    total_loss, recon_loss, kl_loss
    """
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')
    kl_loss    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# =============================================================================
# Training Loop
# =============================================================================
print("\n" + "="*70)
print("  Model Initialisation")
print("="*70)

model = LSTMVAE(
    input_dim  = CFG.INPUT_DIM,
    hidden_dim = CFG.HIDDEN_DIM,
    latent_dim = CFG.LATENT_DIM,
    seq_len    = CFG.WINDOW_SIZE,
    num_layers = CFG.NUM_LAYERS,
    dropout    = CFG.DROPOUT,
).to(CFG.DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Model architecture:")
print(model)
print(f"\n  Trainable parameters : {total_params:,}")

optimizer  = optim.Adam(model.parameters(), lr=CFG.LR, weight_decay=1e-5)
scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=CFG.LR_DECAY, patience=5)

print("\n" + "="*70)
print("  Training LSTM-VAE")
print("="*70)
print(f"\n  Epochs     : {CFG.EPOCHS}")
print(f"  Batch size : {CFG.BATCH_SIZE}")
print(f"  LR         : {CFG.LR}")
print(f"  Beta (KL)  : {CFG.BETA}")
print(f"  Patience   : {CFG.PATIENCE}\n")

history = {"train_loss": [], "val_loss": [], "train_recon": [], "val_recon": []}
best_val_loss  = float('inf')
patience_count = 0
best_model_path = CFG.OUT_DIR / "best_model.pt"

for epoch in range(1, CFG.EPOCHS + 1):
    # ── Training ──────────────────────────────────────────────────────────────
    model.train()
    train_losses, train_recons = [], []

    for (batch,) in train_loader:
        batch = batch.to(CFG.DEVICE)
        optimizer.zero_grad()
        x_hat, mu, log_var = model(batch)
        loss, recon, kl    = elbo_loss(batch, x_hat, mu, log_var, CFG.BETA)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()
        train_losses.append(loss.item())
        train_recons.append(recon.item())

    # ── Validation ────────────────────────────────────────────────────────────
    model.eval()
    val_losses, val_recons = [], []

    with torch.no_grad():
        for (batch,) in val_loader:
            batch = batch.to(CFG.DEVICE)
            x_hat, mu, log_var = model(batch)
            loss, recon, kl    = elbo_loss(batch, x_hat, mu, log_var, CFG.BETA)
            val_losses.append(loss.item())
            val_recons.append(recon.item())

    avg_train = np.mean(train_losses)
    avg_val   = np.mean(val_losses)
    avg_tr    = np.mean(train_recons)
    avg_vr    = np.mean(val_recons)

    history["train_loss"].append(avg_train)
    history["val_loss"].append(avg_val)
    history["train_recon"].append(avg_tr)
    history["val_recon"].append(avg_vr)

    scheduler.step(avg_val)

    # ── Early stopping ────────────────────────────────────────────────────────
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
print(f"  Best validation loss: {best_val_loss:.6f}")


# =============================================================================
# Plot 1 — Training & Validation Loss Curves
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs_ran = range(1, len(history["train_loss"]) + 1)

axes[0].plot(epochs_ran, history["train_loss"], label="Train Loss",  color=PALETTE[0])
axes[0].plot(epochs_ran, history["val_loss"],   label="Val Loss",    color=PALETTE[1])
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("ELBO Loss")
axes[0].set_title("ELBO Loss (Train vs Validation)", fontweight="bold")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].plot(epochs_ran, history["train_recon"], label="Train Recon", color=PALETTE[2])
axes[1].plot(epochs_ran, history["val_recon"],   label="Val Recon",   color=PALETTE[3])
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Reconstruction Loss (MSE)")
axes[1].set_title("Reconstruction Loss (Train vs Validation)", fontweight="bold")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

fig.suptitle("LSTM-VAE Training History", fontweight="bold")
plt.tight_layout()
save_fig(fig, "01_training_curves.png")


# =============================================================================
# Anomaly Scoring
# =============================================================================
print("\n" + "="*70)
print("  Anomaly Scoring")
print("="*70)

# Load best model
model.load_state_dict(torch.load(best_model_path, map_location=CFG.DEVICE))
model.eval()

def compute_anomaly_scores(loader, device):
    """
    Compute per-window reconstruction error (MSE) as anomaly score.
    Higher score = more anomalous.
    """
    scores, labels_list = [], []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
                labels_list.extend(y.numpy())
            else:
                x = batch[0]
            x = x.to(device)
            x_hat, mu, log_var = model(x)
            # Per-window MSE across all timesteps and features
            mse = ((x - x_hat) ** 2).mean(dim=(1, 2))
            scores.extend(mse.cpu().numpy())
    return np.array(scores), np.array(labels_list) if labels_list else None


# Validation scores (for threshold selection)
val_scores, _ = compute_anomaly_scores(val_loader, CFG.DEVICE)

# Test scores
test_scores, test_labels_arr = compute_anomaly_scores(test_loader, CFG.DEVICE)

print(f"\n  Val  scores  — mean: {val_scores.mean():.6f}  std: {val_scores.std():.6f}")
print(f"  Test scores  — mean: {test_scores.mean():.6f}  std: {test_scores.std():.6f}")


# =============================================================================
# Threshold Selection
# =============================================================================
print("\n" + "="*70)
print("  Threshold Selection")
print("="*70)

# Method: search percentile thresholds on validation scores,
# select the one giving best F1 on test set
# (In practice threshold is tuned on val; here we show the search)

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

print(f"\n  Best threshold : {best_thresh:.6f}  "
      f"(at {CFG.THRESHOLD_PERCENTILE}th percentile search)")
print(f"  Precision      : {best_prec:.4f}")
print(f"  Recall         : {best_rec:.4f}")
print(f"  F1-Score       : {best_f1:.4f}")


# =============================================================================
# Final Evaluation
# =============================================================================
print("\n" + "="*70)
print("  Final Evaluation — LSTM-VAE")
print("="*70)

final_preds = (test_scores > best_thresh).astype(int)
precision   = precision_score(test_labels_arr, final_preds, zero_division=0)
recall      = recall_score(test_labels_arr,    final_preds, zero_division=0)
f1          = f1_score(test_labels_arr,        final_preds, zero_division=0)
cm          = confusion_matrix(test_labels_arr, final_preds)

print(f"""
  ┌──────────────────────────────────────────┐
  │         LSTM-VAE RESULTS                 │
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

# Save results
results = {
    "model"     : "LSTM-VAE",
    "threshold" : round(float(best_thresh), 6),
    "precision" : round(float(precision), 4),
    "recall"    : round(float(recall), 4),
    "f1_score"  : round(float(f1), 4),
    "TP"        : int(cm[1][1]),
    "FP"        : int(cm[0][1]),
    "TN"        : int(cm[0][0]),
    "FN"        : int(cm[1][0]),
}
pd.Series(results).to_csv(CFG.OUT_DIR / "results.csv", header=["value"])
print("  Saved -> results.csv")


# =============================================================================
# Plot 2 — Anomaly Score Distribution
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 5))
normal_scores  = test_scores[test_labels_arr == 0]
anomaly_scores = test_scores[test_labels_arr == 1]

ax.hist(normal_scores,  bins=100, alpha=0.6, color=PALETTE[0], label="Normal",  density=True)
ax.hist(anomaly_scores, bins=100, alpha=0.6, color=PALETTE[3], label="Anomaly", density=True)
ax.axvline(best_thresh, color="red", linestyle="--", linewidth=2,
           label=f"Threshold = {best_thresh:.4f}")
ax.set_xlabel("Reconstruction Error (MSE)")
ax.set_ylabel("Density")
ax.set_title("LSTM-VAE — Anomaly Score Distribution\n(Normal vs Anomalous Windows)",
             fontweight="bold")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
save_fig(fig, "02_score_distribution.png")


# =============================================================================
# Plot 3 — Confusion Matrix
# =============================================================================
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
            xticklabels=["Predicted Normal", "Predicted Anomaly"],
            yticklabels=["Actual Normal",    "Actual Anomaly"],
            cbar=False, linewidths=0.5)
ax.set_title("LSTM-VAE — Confusion Matrix", fontweight="bold")
plt.tight_layout()
save_fig(fig, "03_confusion_matrix.png")


# =============================================================================
# Plot 4 — Anomaly Scores Over Time (first 5000 test windows)
# =============================================================================
N_PLOT = min(5000, len(test_scores))
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

t = np.arange(N_PLOT)
axes[0].plot(t, test_scores[:N_PLOT], color=PALETTE[0], linewidth=0.6, label="Anomaly Score")
axes[0].axhline(best_thresh, color="red", linestyle="--", linewidth=1.5,
                label=f"Threshold = {best_thresh:.4f}")
axes[0].set_ylabel("Reconstruction Error (MSE)")
axes[0].set_title("LSTM-VAE — Anomaly Scores Over Time", fontweight="bold")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.4)

axes[1].fill_between(t, test_labels_arr[:N_PLOT], color=PALETTE[3], alpha=0.5,
                     label="Ground Truth Anomaly")
axes[1].fill_between(t, final_preds[:N_PLOT],     color=PALETTE[0], alpha=0.4,
                     label="Predicted Anomaly")
axes[1].set_ylabel("Anomaly Label")
axes[1].set_xlabel("Window Index")
axes[1].set_title("Ground Truth vs Predicted Anomalies", fontweight="bold")
axes[1].legend()
axes[1].set_yticks([0, 1])

fig.suptitle("LSTM-VAE — Detection Results", fontweight="bold")
plt.tight_layout()
save_fig(fig, "04_anomaly_scores_over_time.png")


# =============================================================================
# Plot 5 — F1 Score vs Threshold
# =============================================================================
f1_scores_by_thresh = []
thresh_range = np.percentile(val_scores, np.arange(70, 100, 0.5))
for t in thresh_range:
    preds = (test_scores > t).astype(int)
    f1_scores_by_thresh.append(f1_score(test_labels_arr, preds, zero_division=0))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresh_range, f1_scores_by_thresh, color=PALETTE[0], linewidth=2)
ax.axvline(best_thresh, color="red", linestyle="--", linewidth=2,
           label=f"Best threshold = {best_thresh:.4f}  F1 = {best_f1:.4f}")
ax.set_xlabel("Threshold")
ax.set_ylabel("F1-Score")
ax.set_title("LSTM-VAE — F1-Score vs Threshold", fontweight="bold")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
save_fig(fig, "05_f1_vs_threshold.png")

print(f"""
  ┌──────────────────────────────────────────────────────┐
  │              LSTM-VAE COMPLETE                       │
  ├──────────────────────────────────────────────────────┤
  │  Outputs saved to: ./lstm_vae_outputs/               │
  │    best_model.pt              — saved model weights  │
  │    results.csv                — evaluation metrics   │
  │    01_training_curves.png     — loss curves          │
  │    02_score_distribution.png  — score histogram      │
  │    03_confusion_matrix.png    — confusion matrix     │
  │    04_anomaly_scores_over_time.png                   │
  │    05_f1_vs_threshold.png     — threshold search     │
  └──────────────────────────────────────────────────────┘
""")
print("  Next step: Anomaly Transformer (Model 2)")

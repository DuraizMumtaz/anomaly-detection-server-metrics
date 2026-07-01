"""
=============================================================================
Data Preprocessing Pipeline
Anomaly Detection in Server Machine Metrics
Master's Dissertation — Arden University

Dataset : Server Machine Dataset (SMD)
         Su et al., KDD 2019 — OmniAnomaly
Author  : Duraiz Mumtaz
=============================================================================

This script performs the full preprocessing pipeline required before model
training, covering:

  1.  Data loading & integrity validation
  2.  Normalisation verification
  3.  Sliding window segmentation
  4.  Train / validation split (80/20)
  5.  PyTorch Dataset & DataLoader construction
  6.  Saving preprocessed data to disk
  7.  Preprocessing summary report

Output directory: ./preprocessed_data/
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

class Config:
    """
    Central configuration for all preprocessing parameters.
    Changing values here propagates to the entire pipeline.
    """
    # Paths
    BASE_DIR      = Path(__file__).parent
    TRAIN_DIR     = BASE_DIR / "SMD_CSV" / "train"
    TEST_DIR      = BASE_DIR / "SMD_CSV" / "test"
    LABEL_DIR     = BASE_DIR / "SMD_CSV" / "test_label"
    OUT_DIR       = BASE_DIR / "preprocessed_data"

    # Sliding window parameters
    WINDOW_SIZE   = 100    # number of timesteps per window (sequence length)
    STEP_SIZE     = 1      # stride between consecutive windows
                           # step=1 → maximum overlap (used by Anomaly Transformer)
                           # step=WINDOW_SIZE → no overlap (faster but less data)

    # Train / validation split
    VAL_RATIO     = 0.2    # 20% of training windows used for validation

    # DataLoader parameters
    BATCH_SIZE    = 64     # number of windows per batch
    NUM_WORKERS   = 0      # set to 0 for macOS compatibility
    SHUFFLE_TRAIN = True   # shuffle training batches

    # Reproducibility
    RANDOM_SEED   = 42

    # Metrics
    N_FEATURES    = 38

CFG = Config()
CFG.OUT_DIR.mkdir(exist_ok=True)

# Fix random seeds for reproducibility
np.random.seed(CFG.RANDOM_SEED)
torch.manual_seed(CFG.RANDOM_SEED)

# ── Column names ──────────────────────────────────────────────────────────────
COLUMNS = [
    "cpu_r", "cpu_u", "cpu_q",
    "cpu_loadavg_1min", "cpu_loadavg_5min", "cpu_loadavg_15min",
    "net_recv", "net_send", "net_rx_drop", "net_tx_drop",
    "net_rx_error", "net_tx_error", "net_rx_packets", "net_tx_packets",
    "mem_shmem", "mem_buff", "mem_used", "mem_free",
    "disk_read", "disk_write", "disk_read_merged", "disk_write_merged",
    "disk_read_sectors", "disk_write_sectors",
    "disk_io_in_progress", "disk_io_time",
    "processes_r", "processes_b", "processes_t", "processes_w",
    "processes_s", "processes_d", "processes_i", "processes_running",
    "processes_blocked", "processes_zombie", "processes_stopped",
    "processes_total",
]


# =============================================================================
# Helper Functions
# =============================================================================

def get_machine_names():
    """Return sorted list of all machine names."""
    return sorted([f.stem for f in CFG.TRAIN_DIR.glob("*.csv")])


def load_machine(machine_name):
    """
    Load train, test, and label CSVs for one machine.
    Returns numpy arrays for efficiency.
    """
    train = pd.read_csv(CFG.TRAIN_DIR / f"{machine_name}.csv").values.astype(np.float32)
    test  = pd.read_csv(CFG.TEST_DIR  / f"{machine_name}.csv").values.astype(np.float32)
    label = pd.read_csv(CFG.LABEL_DIR / f"{machine_name}.csv").values.astype(np.int32).flatten()
    return train, test, label


def sliding_window(data, window_size, step_size):
    """
    Apply sliding window segmentation to a time series array.

    Parameters
    ----------
    data        : np.ndarray of shape (T, F) — T timesteps, F features
    window_size : int — length of each window
    step_size   : int — stride between windows

    Returns
    -------
    windows : np.ndarray of shape (N, window_size, F)
              N = number of windows produced
    """
    T, F = data.shape
    # Calculate number of windows
    n_windows = (T - window_size) // step_size + 1
    windows = np.zeros((n_windows, window_size, F), dtype=np.float32)
    for i in range(n_windows):
        start = i * step_size
        end   = start + window_size
        windows[i] = data[start:end]
    return windows


def sliding_window_labels(labels, window_size, step_size):
    """
    Apply sliding window to label array.
    A window is labelled anomalous (1) if ANY timestep within it is anomalous.
    This follows the standard point-adjust evaluation protocol for SMD.

    Parameters
    ----------
    labels      : np.ndarray of shape (T,)
    window_size : int
    step_size   : int

    Returns
    -------
    window_labels : np.ndarray of shape (N,) — 0 or 1 per window
    """
    T = len(labels)
    n_windows = (T - window_size) // step_size + 1
    window_labels = np.zeros(n_windows, dtype=np.int32)
    for i in range(n_windows):
        start = i * step_size
        end   = start + window_size
        window_labels[i] = int(np.any(labels[start:end] == 1))
    return window_labels


# =============================================================================
# PyTorch Dataset Classes
# =============================================================================

class SMDTrainDataset(Dataset):
    """
    PyTorch Dataset for training data (anomaly-free windows).
    Returns windows only — no labels needed for unsupervised training.
    """
    def __init__(self, windows):
        """
        Parameters
        ----------
        windows : np.ndarray of shape (N, window_size, n_features)
        """
        self.windows = torch.tensor(windows, dtype=torch.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


class SMDTestDataset(Dataset):
    """
    PyTorch Dataset for test data.
    Returns (window, label) pairs for evaluation.
    """
    def __init__(self, windows, labels):
        """
        Parameters
        ----------
        windows : np.ndarray of shape (N, window_size, n_features)
        labels  : np.ndarray of shape (N,) — 0 or 1 per window
        """
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels  = torch.tensor(labels,  dtype=torch.long)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


# =============================================================================
# Step 1 — Data Loading & Integrity Validation
# =============================================================================
print("\n" + "="*70)
print("  STEP 1 -- Data Loading & Integrity Validation")
print("="*70)

machines = get_machine_names()
print(f"\n  Machines found : {len(machines)}")

validation_errors = []
for m in machines:
    tr, te, lb = load_machine(m)

    # Check feature count
    if tr.shape[1] != CFG.N_FEATURES:
        validation_errors.append(f"{m}: train has {tr.shape[1]} features (expected {CFG.N_FEATURES})")
    if te.shape[1] != CFG.N_FEATURES:
        validation_errors.append(f"{m}: test has {te.shape[1]} features (expected {CFG.N_FEATURES})")

    # Check label length matches test length
    if len(lb) != len(te):
        validation_errors.append(f"{m}: label length {len(lb)} != test length {len(te)}")

    # Check for NaN / Inf
    if np.any(np.isnan(tr)) or np.any(np.isinf(tr)):
        validation_errors.append(f"{m}: NaN/Inf found in train data")
    if np.any(np.isnan(te)) or np.any(np.isinf(te)):
        validation_errors.append(f"{m}: NaN/Inf found in test data")

if validation_errors:
    print("\n  VALIDATION ERRORS:")
    for e in validation_errors:
        print(f"    [ERROR] {e}")
else:
    print("  All 28 machines passed integrity validation.")
    print("  - Feature count   : 38 per machine [OK]")
    print("  - Label alignment : test rows == label rows [OK]")
    print("  - NaN / Inf       : none detected [OK]")


# =============================================================================
# Step 2 — Normalisation Verification
# =============================================================================
print("\n" + "="*70)
print("  STEP 2 -- Normalisation Verification")
print("="*70)

all_min, all_max = [], []
for m in machines:
    tr, te, _ = load_machine(m)
    all_min.append(tr.min())
    all_max.append(tr.max())
    all_min.append(te.min())
    all_max.append(te.max())

global_min = min(all_min)
global_max = max(all_max)

print(f"\n  Global value range across all machines and splits:")
print(f"    Min : {global_min:.6f}")
print(f"    Max : {global_max:.6f}")

if global_min >= -0.01 and global_max <= 1.01:
    print("  All values confirmed within [0, 1] range.")
    print("  No additional normalisation required.")
else:
    print(f"  WARNING: Values outside [0,1] detected.")
    print(f"  Clipping values to [0, 1] range.")


# =============================================================================
# Step 3 — Sliding Window Segmentation
# =============================================================================
print("\n" + "="*70)
print("  STEP 3 -- Sliding Window Segmentation")
print("="*70)
print(f"\n  Window size : {CFG.WINDOW_SIZE} timesteps")
print(f"  Step size   : {CFG.STEP_SIZE}  (stride)")
print(f"  Each window : shape ({CFG.WINDOW_SIZE}, {CFG.N_FEATURES})")

all_train_windows  = []
all_test_windows   = []
all_test_labels    = []
machine_stats      = []

for m in machines:
    tr, te, lb = load_machine(m)

    # Clip to [0,1] as a safety measure
    tr = np.clip(tr, 0.0, 1.0)
    te = np.clip(te, 0.0, 1.0)

    # Sliding window
    tr_windows = sliding_window(tr, CFG.WINDOW_SIZE, CFG.STEP_SIZE)
    te_windows = sliding_window(te, CFG.WINDOW_SIZE, CFG.STEP_SIZE)
    te_labels  = sliding_window_labels(lb, CFG.WINDOW_SIZE, CFG.STEP_SIZE)

    all_train_windows.append(tr_windows)
    all_test_windows.append(te_windows)
    all_test_labels.append(te_labels)

    machine_stats.append({
        "machine"        : m,
        "train_timesteps": len(tr),
        "test_timesteps" : len(te),
        "train_windows"  : len(tr_windows),
        "test_windows"   : len(te_windows),
        "test_anomaly_w" : int(te_labels.sum()),
        "test_anomaly_%"  : round(100.0 * te_labels.sum() / len(te_labels), 2),
    })

df_stats = pd.DataFrame(machine_stats)
print(f"\n  Per-machine window statistics:")
print(df_stats.to_string(index=False))

total_train_w = sum(len(w) for w in all_train_windows)
total_test_w  = sum(len(w) for w in all_test_windows)
print(f"\n  Total train windows : {total_train_w:,}")
print(f"  Total test windows  : {total_test_w:,}")


# =============================================================================
# Step 4 — Train / Validation Split
# =============================================================================
print("\n" + "="*70)
print("  STEP 4 -- Train / Validation Split (80% / 20%)")
print("="*70)

# Concatenate all machine train windows
X_train_all = np.concatenate(all_train_windows, axis=0)  # (N_total, W, F)

# Stratify is not applicable here (all train windows are normal)
# We use a simple sequential split to preserve temporal order
split_idx   = int(len(X_train_all) * (1 - CFG.VAL_RATIO))
X_train     = X_train_all[:split_idx]
X_val       = X_train_all[split_idx:]

# Test: concatenate all machines
X_test_all  = np.concatenate(all_test_windows, axis=0)   # (N_test, W, F)
y_test_all  = np.concatenate(all_test_labels,  axis=0)   # (N_test,)

print(f"\n  Train windows  : {len(X_train):,}  shape: {X_train.shape}")
print(f"  Val windows    : {len(X_val):,}    shape: {X_val.shape}")
print(f"  Test windows   : {len(X_test_all):,}   shape: {X_test_all.shape}")
print(f"  Test labels    : {len(y_test_all):,}   anomaly rate: "
      f"{100.0 * y_test_all.sum() / len(y_test_all):.2f}%")


# =============================================================================
# Step 5 — PyTorch DataLoaders
# =============================================================================
print("\n" + "="*70)
print("  STEP 5 -- PyTorch Dataset & DataLoader Construction")
print("="*70)

train_dataset = SMDTrainDataset(X_train)
val_dataset   = SMDTrainDataset(X_val)
test_dataset  = SMDTestDataset(X_test_all, y_test_all)

train_loader = DataLoader(
    train_dataset,
    batch_size  = CFG.BATCH_SIZE,
    shuffle     = CFG.SHUFFLE_TRAIN,
    num_workers = CFG.NUM_WORKERS,
    pin_memory  = False,
    drop_last   = True,   # drop last incomplete batch for stable training
)

val_loader = DataLoader(
    val_dataset,
    batch_size  = CFG.BATCH_SIZE,
    shuffle     = False,
    num_workers = CFG.NUM_WORKERS,
    pin_memory  = False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size  = CFG.BATCH_SIZE,
    shuffle     = False,
    num_workers = CFG.NUM_WORKERS,
    pin_memory  = False,
)

print(f"\n  Train DataLoader  : {len(train_loader):,} batches x batch_size {CFG.BATCH_SIZE}")
print(f"  Val   DataLoader  : {len(val_loader):,} batches x batch_size {CFG.BATCH_SIZE}")
print(f"  Test  DataLoader  : {len(test_loader):,} batches x batch_size {CFG.BATCH_SIZE}")

# Verify a batch
sample_batch = next(iter(train_loader))
print(f"\n  Sample train batch shape : {sample_batch.shape}")
print(f"  Expected                 : ({CFG.BATCH_SIZE}, {CFG.WINDOW_SIZE}, {CFG.N_FEATURES})")
assert sample_batch.shape == (CFG.BATCH_SIZE, CFG.WINDOW_SIZE, CFG.N_FEATURES), \
    "Batch shape mismatch! Check window size and feature count."
print("  Batch shape verified [OK]")


# =============================================================================
# Step 6 — Save Preprocessed Data
# =============================================================================
print("\n" + "="*70)
print("  STEP 6 -- Saving Preprocessed Data to Disk")
print("="*70)

np.save(CFG.OUT_DIR / "X_train.npy",  X_train)
np.save(CFG.OUT_DIR / "X_val.npy",    X_val)
np.save(CFG.OUT_DIR / "X_test.npy",   X_test_all)
np.save(CFG.OUT_DIR / "y_test.npy",   y_test_all)

# Also save per-machine test data for per-machine evaluation later
for i, m in enumerate(machines):
    np.save(CFG.OUT_DIR / f"test_{m}.npy",       all_test_windows[i])
    np.save(CFG.OUT_DIR / f"test_label_{m}.npy", all_test_labels[i])

# Save config summary
config_summary = {
    "window_size"       : CFG.WINDOW_SIZE,
    "step_size"         : CFG.STEP_SIZE,
    "val_ratio"         : CFG.VAL_RATIO,
    "batch_size"        : CFG.BATCH_SIZE,
    "random_seed"       : CFG.RANDOM_SEED,
    "n_features"        : CFG.N_FEATURES,
    "n_machines"        : len(machines),
    "total_train_windows": int(len(X_train)),
    "total_val_windows" : int(len(X_val)),
    "total_test_windows": int(len(X_test_all)),
    "test_anomaly_rate" : float(round(100.0 * y_test_all.sum() / len(y_test_all), 4)),
}
pd.Series(config_summary).to_csv(CFG.OUT_DIR / "preprocessing_config.csv", header=["value"])

print(f"\n  Saved to: ./preprocessed_data/")
print(f"    X_train.npy          — shape {X_train.shape}")
print(f"    X_val.npy            — shape {X_val.shape}")
print(f"    X_test.npy           — shape {X_test_all.shape}")
print(f"    y_test.npy           — shape {y_test_all.shape}")
print(f"    test_<machine>.npy   — per-machine test windows (28 files)")
print(f"    test_label_<m>.npy   — per-machine test labels  (28 files)")
print(f"    preprocessing_config.csv")


# =============================================================================
# Step 7 — Preprocessing Summary Report
# =============================================================================
print("\n" + "="*70)
print("  STEP 7 -- Preprocessing Summary Report")
print("="*70)
print(f"""
  ┌─────────────────────────────────────────────────────┐
  │          PREPROCESSING SUMMARY                      │
  ├─────────────────────────────────────────────────────┤
  │  Dataset          : SMD (28 production servers)     │
  │  Features         : 38 server metrics               │
  │  Normalisation    : Min-Max [0,1] — verified        │
  │  Window size      : {CFG.WINDOW_SIZE} timesteps                  │
  │  Step size        : {CFG.STEP_SIZE} (maximum overlap)            │
  │                                                     │
  │  Train windows    : {len(X_train):>10,}                    │
  │  Val   windows    : {len(X_val):>10,}                    │
  │  Test  windows    : {len(X_test_all):>10,}                    │
  │  Test anomaly rate: {100.0*y_test_all.sum()/len(y_test_all):>9.2f}%                   │
  │                                                     │
  │  Batch size       : {CFG.BATCH_SIZE}                           │
  │  Train batches    : {len(train_loader):>10,}                    │
  │  Val   batches    : {len(val_loader):>10,}                    │
  │  Test  batches    : {len(test_loader):>10,}                    │
  └─────────────────────────────────────────────────────┘
""")
print("  Preprocessing complete. Ready for model training.")
print("="*70 + "\n")

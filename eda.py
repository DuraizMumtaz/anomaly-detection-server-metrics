"""
=============================================================================
Exploratory Data Analysis (EDA)
Anomaly Detection in Server Machine Metrics
Dataset : Server Machine Dataset (SMD)
         Su et al., KDD 2019 — OmniAnomaly
Author  : Duraiz Mumtaz
=============================================================================

This script performs a comprehensive EDA on the SMD dataset covering:
  1.  Dataset structure and overview
  2.  Shape and size statistics across all machines
  3.  Missing value analysis
  4.  Descriptive statistics per metric group
  5.  Anomaly label distribution (per machine & overall)
  6.  Metric-level value distributions (box plots)
  7.  Correlation heatmap across all 38 metrics
  8.  Time series visualisation with anomaly highlights (sample machine)
  9.  Mean metric values: anomaly vs normal
  10. Inter-machine variability analysis

All figures are saved to  ./eda_outputs/  for use in the final report.
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
TRAIN_DIR  = BASE_DIR / "SMD_CSV" / "train"
TEST_DIR   = BASE_DIR / "SMD_CSV" / "test"
LABEL_DIR  = BASE_DIR / "SMD_CSV" / "test_label"
OUT_DIR    = BASE_DIR / "eda_outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── Column names (38 metrics) ─────────────────────────────────────────────────
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

METRIC_GROUPS = {
    "CPU"      : COLUMNS[0:6],
    "Network"  : COLUMNS[6:14],
    "Memory"   : COLUMNS[14:18],
    "Disk"     : COLUMNS[18:26],
    "Processes": COLUMNS[26:38],
}

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi"      : 150,
    "font.family"     : "DejaVu Sans",
    "axes.titlesize"  : 13,
    "axes.labelsize"  : 11,
    "xtick.labelsize" : 9,
    "ytick.labelsize" : 9,
    "legend.fontsize" : 9,
    "figure.titlesize": 15,
})
PALETTE = sns.color_palette("tab10")

# =============================================================================
# Helper functions
# =============================================================================

def load_machine(machine_name):
    """Load train, test, and label CSVs for one machine."""
    train = pd.read_csv(TRAIN_DIR / f"{machine_name}.csv")
    test  = pd.read_csv(TEST_DIR  / f"{machine_name}.csv")
    label = pd.read_csv(LABEL_DIR / f"{machine_name}.csv")
    return train, test, label


def get_machine_names():
    return sorted([f.stem for f in TRAIN_DIR.glob("*.csv")])


def save_fig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path.name}")


# =============================================================================
# 1. Dataset Overview
# =============================================================================
print("\n" + "="*70)
print("  SECTION 1 -- Dataset Overview")
print("="*70)

machines = get_machine_names()
print(f"\n  Total machines      : {len(machines)}")
print(f"  Machine groups      : 3  (machine-1-x, machine-2-x, machine-3-x)")
print(f"  Metrics per machine : {len(COLUMNS)}")
print(f"  Metric groups       : {list(METRIC_GROUPS.keys())}")
print(f"\n  Machines: {machines}")


# =============================================================================
# 2. Shape & Size Statistics
# =============================================================================
print("\n" + "="*70)
print("  SECTION 2 -- Shape & Size Statistics")
print("="*70)

rows_train, rows_test = [], []
for m in machines:
    tr, te, lb = load_machine(m)
    rows_train.append({"machine": m, "train_rows": len(tr), "train_cols": tr.shape[1]})
    rows_test.append({"machine": m, "test_rows": len(te), "test_cols": te.shape[1],
                      "label_rows": len(lb)})

df_shapes = pd.merge(
    pd.DataFrame(rows_train),
    pd.DataFrame(rows_test),
    on="machine"
)

print("\n  Per-machine row counts:")
print(df_shapes.to_string(index=False))
print(f"\n  Train rows -- mean: {df_shapes.train_rows.mean():.0f} | "
      f"min: {df_shapes.train_rows.min()} | max: {df_shapes.train_rows.max()}")
print(f"  Test rows  -- mean: {df_shapes.test_rows.mean():.0f}  | "
      f"min: {df_shapes.test_rows.min()}  | max: {df_shapes.test_rows.max()}")

total_train = df_shapes.train_rows.sum()
total_test  = df_shapes.test_rows.sum()
print(f"\n  TOTAL train timesteps : {total_train:,}")
print(f"  TOTAL test  timesteps : {total_test:,}")
print(f"  TOTAL timesteps       : {total_train + total_test:,}")

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(machines))
w = 0.4
ax.bar(x - w/2, df_shapes.train_rows, width=w, label="Train", color=PALETTE[0], alpha=0.85)
ax.bar(x + w/2, df_shapes.test_rows,  width=w, label="Test",  color=PALETTE[1], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(machines, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Number of Timesteps")
ax.set_title("Train vs Test Timestep Counts per Machine", fontweight="bold")
ax.legend()
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
fig.suptitle("SMD Dataset -- Per-Machine Data Volume", fontweight="bold", y=1.01)
save_fig(fig, "01_train_test_sizes.png")


# =============================================================================
# 3. Missing Value Analysis
# =============================================================================
print("\n" + "="*70)
print("  SECTION 3 -- Missing Value Analysis")
print("="*70)

total_missing = 0
for m in machines:
    tr, te, lb = load_machine(m)
    missing = tr.isnull().sum().sum() + te.isnull().sum().sum()
    total_missing += missing

print(f"\n  Total missing values across all machines: {total_missing}")
if total_missing == 0:
    print("  No missing values detected -- dataset is complete.")
else:
    print("  WARNING: Missing values found -- requires imputation before modelling.")


# =============================================================================
# 4. Descriptive Statistics
# =============================================================================
print("\n" + "="*70)
print("  SECTION 4 -- Descriptive Statistics (pooled train data)")
print("="*70)

all_train_frames = []
for m in machines:
    tr, _, _ = load_machine(m)
    all_train_frames.append(tr)

df_all_train = pd.concat(all_train_frames, ignore_index=True)
desc = df_all_train.describe().T
desc["range"] = desc["max"] - desc["min"]
desc["cv"]    = desc["std"] / (desc["mean"] + 1e-9)

print("\n  Summary statistics (all 38 metrics, pooled train data):")
print(desc[["mean", "std", "min", "50%", "max", "range", "cv"]].round(4).to_string())

desc.to_csv(OUT_DIR / "descriptive_statistics.csv")
print("\n  Saved -> descriptive_statistics.csv")


# =============================================================================
# 5. Anomaly Label Distribution
# =============================================================================
print("\n" + "="*70)
print("  SECTION 5 -- Anomaly Label Distribution")
print("="*70)

anomaly_stats = []
for m in machines:
    _, te, lb = load_machine(m)
    n_total   = len(lb)
    n_anomaly = int(lb["anomaly"].sum())
    n_normal  = n_total - n_anomaly
    pct       = 100.0 * n_anomaly / n_total
    anomaly_stats.append({
        "machine"    : m,
        "total"      : n_total,
        "normal"     : n_normal,
        "anomaly"    : n_anomaly,
        "anomaly_pct": round(pct, 2),
    })

df_anomaly = pd.DataFrame(anomaly_stats)
total_anomaly = df_anomaly.anomaly.sum()
total_rows    = df_anomaly.total.sum()
overall_pct   = 100.0 * total_anomaly / total_rows

print(f"\n  Overall anomaly rate : {overall_pct:.2f}%  "
      f"({total_anomaly:,} / {total_rows:,} timesteps)")
print(f"\n  Per-machine anomaly rates:")
print(df_anomaly.to_string(index=False))

df_anomaly.to_csv(OUT_DIR / "anomaly_distribution.csv", index=False)
print("\n  Saved -> anomaly_distribution.csv")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors = [PALETTE[3] if p > overall_pct else PALETTE[0] for p in df_anomaly.anomaly_pct]
axes[0].bar(df_anomaly.machine, df_anomaly.anomaly_pct, color=colors, alpha=0.85)
axes[0].axhline(overall_pct, color="red", linestyle="--", linewidth=1.5,
                label=f"Overall mean: {overall_pct:.2f}%")
axes[0].set_xticklabels(df_anomaly.machine, rotation=45, ha="right", fontsize=8)
axes[0].set_ylabel("Anomaly Rate (%)")
axes[0].set_title("Anomaly Rate per Machine", fontweight="bold")
axes[0].legend()
axes[0].yaxis.grid(True, linestyle="--", alpha=0.5)
axes[0].set_axisbelow(True)

axes[1].bar(df_anomaly.machine, df_anomaly.normal,  label="Normal",  color=PALETTE[0], alpha=0.85)
axes[1].bar(df_anomaly.machine, df_anomaly.anomaly, bottom=df_anomaly.normal,
            label="Anomaly", color=PALETTE[3], alpha=0.85)
axes[1].set_xticklabels(df_anomaly.machine, rotation=45, ha="right", fontsize=8)
axes[1].set_ylabel("Timestep Count")
axes[1].set_title("Normal vs Anomalous Timesteps per Machine", fontweight="bold")
axes[1].legend()
axes[1].yaxis.grid(True, linestyle="--", alpha=0.5)
axes[1].set_axisbelow(True)

fig.suptitle("SMD Dataset -- Anomaly Distribution Analysis", fontweight="bold")
plt.tight_layout()
save_fig(fig, "02_anomaly_distribution.png")


# =============================================================================
# 6. Metric Value Distributions (Box Plots)
# =============================================================================
print("\n" + "="*70)
print("  SECTION 6 -- Metric Value Distributions")
print("="*70)

fig, axes = plt.subplots(len(METRIC_GROUPS), 1, figsize=(16, 22))
fig.suptitle("SMD Dataset -- Metric Value Distributions by Group\n(Pooled Train Data, All Machines)",
             fontweight="bold", y=1.01)

for idx, (ax, (group_name, group_cols)) in enumerate(zip(axes, METRIC_GROUPS.items())):
    data_to_plot = [df_all_train[col].values for col in group_cols]
    bp = ax.boxplot(data_to_plot, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE[idx])
        patch.set_alpha(0.7)
    ax.set_xticklabels(group_cols, rotation=30, ha="right", fontsize=8)
    ax.set_title(f"{group_name} Metrics", fontweight="bold")
    ax.set_ylabel("Normalised Value (0-1)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

plt.tight_layout()
save_fig(fig, "03_metric_distributions.png")
print("  Box plots saved.")


# =============================================================================
# 7. Correlation Heatmap
# =============================================================================
print("\n" + "="*70)
print("  SECTION 7 -- Correlation Heatmap (38 Metrics)")
print("="*70)

corr = df_all_train.corr()

fig, ax = plt.subplots(figsize=(18, 15))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, cmap="RdBu_r", center=0,
    vmin=-1, vmax=1, linewidths=0.3, linecolor="white",
    annot=False, square=True, ax=ax,
    cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"}
)
ax.set_title("Pearson Correlation Matrix -- All 38 Server Metrics\n(Pooled Train Data, All 28 Machines)",
             fontweight="bold", pad=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
save_fig(fig, "04_correlation_heatmap.png")
print("  Correlation heatmap saved.")

print("\n  Highly correlated metric pairs (|r| > 0.85):")
corr_pairs = []
for i in range(len(COLUMNS)):
    for j in range(i+1, len(COLUMNS)):
        r = corr.iloc[i, j]
        if abs(r) > 0.85:
            corr_pairs.append((COLUMNS[i], COLUMNS[j], round(r, 3)))
corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for c1, c2, r in corr_pairs[:15]:
    print(f"    {c1:30s} <-> {c2:30s}  r = {r:+.3f}")


# =============================================================================
# 8. Time Series with Anomaly Highlights
# =============================================================================
print("\n" + "="*70)
print("  SECTION 8 -- Time Series with Anomaly Highlights (machine-1-1)")
print("="*70)

SAMPLE_MACHINE = "machine-1-1"
_, te_sample, lb_sample = load_machine(SAMPLE_MACHINE)
anomaly_idx = lb_sample["anomaly"].values

REPRESENTATIVE = {
    "CPU"      : "cpu_r",
    "Network"  : "net_recv",
    "Memory"   : "mem_used",
    "Disk"     : "disk_write",
    "Processes": "processes_running",
}

fig, axes = plt.subplots(len(REPRESENTATIVE), 1, figsize=(18, 14), sharex=True)
fig.suptitle(f"Time Series with Anomaly Highlights -- {SAMPLE_MACHINE}\n"
             f"(One representative metric per group)",
             fontweight="bold")

for ax, (group, metric) in zip(axes, REPRESENTATIVE.items()):
    values = te_sample[metric].values
    time   = np.arange(len(values))
    ax.plot(time, values, color=PALETTE[0], linewidth=0.7, label=metric)

    in_anomaly = False
    start = 0
    for t in range(len(anomaly_idx)):
        if anomaly_idx[t] == 1 and not in_anomaly:
            start = t
            in_anomaly = True
        elif anomaly_idx[t] == 0 and in_anomaly:
            ax.axvspan(start, t, color="red", alpha=0.2)
            in_anomaly = False
    if in_anomaly:
        ax.axvspan(start, len(anomaly_idx), color="red", alpha=0.2)

    ax.set_ylabel(f"{group}\n({metric})", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

axes[-1].set_xlabel("Timestep")
normal_patch  = mpatches.Patch(color=PALETTE[0], label="Normal signal")
anomaly_patch = mpatches.Patch(color="red", alpha=0.4, label="Anomaly region")
fig.legend(handles=[normal_patch, anomaly_patch], loc="upper right", bbox_to_anchor=(0.98, 0.98))
plt.tight_layout()
save_fig(fig, "05_timeseries_anomaly_highlights.png")
print(f"  Time series plot saved for {SAMPLE_MACHINE}.")


# =============================================================================
# 9. Mean Metric Values: Anomaly vs Normal
# =============================================================================
print("\n" + "="*70)
print("  SECTION 9 -- Mean Metric Values: Anomaly vs Normal")
print("="*70)

normal_frames, anomaly_frames = [], []
for m in machines:
    _, te, lb = load_machine(m)
    normal_frames.append(te[lb["anomaly"] == 0])
    anomaly_frames.append(te[lb["anomaly"] == 1])

df_normal_data  = pd.concat(normal_frames,  ignore_index=True)
df_anomaly_data = pd.concat(anomaly_frames, ignore_index=True)

mean_normal  = df_normal_data.mean()
mean_anomaly = df_anomaly_data.mean()
diff         = (mean_anomaly - mean_normal).abs().sort_values(ascending=False)

print("\n  Top 15 metrics with largest mean difference (anomaly vs normal):")
print(diff.head(15).round(4).to_string())

fig, ax = plt.subplots(figsize=(16, 6))
x = np.arange(len(COLUMNS))
ax.bar(x - 0.2, mean_normal,  width=0.4, label="Normal",  color=PALETTE[0], alpha=0.85)
ax.bar(x + 0.2, mean_anomaly, width=0.4, label="Anomaly", color=PALETTE[3], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(COLUMNS, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Mean Normalised Value")
ax.set_title("Mean Metric Values: Normal vs Anomalous Timesteps\n(All 28 Machines Pooled)",
             fontweight="bold")
ax.legend()
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
save_fig(fig, "06_normal_vs_anomaly_means.png")


# =============================================================================
# 10. Inter-Machine Variability
# =============================================================================
print("\n" + "="*70)
print("  SECTION 10 -- Inter-Machine Variability")
print("="*70)

machine_means = []
for m in machines:
    tr, _, _ = load_machine(m)
    row = tr.mean().to_dict()
    row["machine"] = m
    machine_means.append(row)

df_machine_means = pd.DataFrame(machine_means).set_index("machine")
variability = df_machine_means.std()
top_variable = variability.sort_values(ascending=False).head(10)

print("\n  Top 10 most variable metrics across machines:")
print(top_variable.round(4).to_string())

fig, ax = plt.subplots(figsize=(12, 5))
top_variable.plot(kind="bar", ax=ax, color=PALETTE[2], alpha=0.85, edgecolor="black", linewidth=0.5)
ax.set_title("Top 10 Most Variable Metrics Across 28 Machines\n(Std. Dev. of per-machine mean values)",
             fontweight="bold")
ax.set_ylabel("Standard Deviation of Machine Mean")
ax.set_xlabel("Metric")
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
save_fig(fig, "07_inter_machine_variability.png")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("  EDA COMPLETE -- Summary")
print("="*70)
print(f"""
  Dataset       : Server Machine Dataset (SMD)
  Machines      : {len(machines)}
  Metrics       : {len(COLUMNS)} per machine
  Train samples : {total_train:,} timesteps (anomaly-free)
  Test samples  : {total_test:,} timesteps
  Anomaly rate  : {overall_pct:.2f}% of test data
  Missing values: {total_missing}

  Outputs saved to: ./eda_outputs/
    01_train_test_sizes.png
    02_anomaly_distribution.png
    03_metric_distributions.png
    04_correlation_heatmap.png
    05_timeseries_anomaly_highlights.png
    06_normal_vs_anomaly_means.png
    07_inter_machine_variability.png
    descriptive_statistics.csv
    anomaly_distribution.csv
""")
print("="*70 + "\n")

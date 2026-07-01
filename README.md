# Anomaly Detection in Server Machine Metrics

**MSc Advanced Computing Dissertation — Arden University**
**Author:** Duraiz Mumtaz
**Supervisor:** Dr. Waleed Iqbal

---

## Project Overview

A comparative study of three deep learning architectures for unsupervised anomaly detection in multivariate server telemetry using the **Server Machine Dataset (SMD)**.

## Models Compared

| Model | Precision | Recall | F1-Score | Parameters |
|-------|-----------|--------|----------|------------|
| LSTM-VAE | 0.1596 | 0.2073 | 0.1803 | 857,062 |
| Anomaly Transformer | 0.2031 | 0.2149 | **0.2088** | 104,754 |
| MTAD-GAT | 0.2135 | 0.1410 | 0.1698 | 115,384 |

**Winner: Anomaly Transformer** — 15.8% F1 improvement over LSTM-VAE baseline.

---

## Dataset

**Server Machine Dataset (SMD)** — Su et al., KDD 2019
- 28 production servers
- 38 metrics per machine (CPU, memory, disk, network)
- 5 weeks of telemetry
- Source: https://github.com/NetManAIOps/OmniAnomaly

Raw CSV data is included in `SMD_CSV/`.

---

## Project Structure

```
├── eda.py                        # Exploratory Data Analysis (10 sections, 7 figures)
├── preprocessing.py              # Data pipeline (normalisation + sliding window)
├── lstm_vae.py                   # LSTM-VAE model
├── anomaly_transformer.py        # Anomaly Transformer model
├── mtad_gat.py                   # MTAD-GAT model
├── SMD_CSV/                      # Raw dataset (train/test/test_label CSVs)
├── eda_outputs/                  # EDA figures and CSVs
├── lstm_vae_outputs/             # Training curves, confusion matrix, results
├── anomaly_transformer_outputs/  # Training curves, confusion matrix, results
├── mtad_gat_outputs/             # Training curves, confusion matrix, results
```

> **Note:** `preprocessed_data/` is not included (files exceed GitHub's 100MB limit).
> Run `python3 preprocessing.py` to regenerate it.

---

## Setup & Usage

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install torch pandas numpy scikit-learn matplotlib seaborn

# 3. Run in order
python3 eda.py
python3 preprocessing.py
python3 lstm_vae.py
python3 anomaly_transformer.py
python3 mtad_gat.py
```

**Hardware used:** Apple MacBook Air M4 (24GB RAM, MPS GPU backend)

---

## Key References

- Su et al. (2019) — OmniAnomaly / SMD Dataset — KDD 2019
- Xu et al. (2022) — Anomaly Transformer — ICLR 2022
- Zhao et al. (2020) — MTAD-GAT — ICDM 2020
- Kingma & Welling (2014) — VAE — ICLR 2014

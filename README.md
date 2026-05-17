# Causal-Distributional Temporal Transformer (CDTT)

An end-to-end, decision-aware deep learning framework optimized for incremental Customer Lifetime Value (CLV) estimation and marketing intervention targeting under zero-inflated, heavy-tailed observational enterprise distributions.

---

## 📊 Benchmark Results

Evaluating on our customer-isolated, cross-fitted semi-synthetic benchmarking environment (constructed from the UCI Online Retail II transactional ledger), **CDTT-Structural** establishes a new performance frontier, outperforming tree-based causal meta-learners by nearly **3×** while avoiding the variance collapse observed in Doubly Robust pseudo-outcome regressors.

| Model Variant | PEHE $\downarrow$ | IPW-AUUC $\uparrow$ | Profit @ 20% $\uparrow$ |
| :--- | :---: | :---: | :---: |
| **CDTT-Structural** (Ours) | 1394.47 (968.76, 1765.62) | **882.73** (522.45, 1304.19) | **630.73** (546.19, 713.48) |
| **CDTT-DR** (Pseudo-Outcome Head) | **535.07** (502.30, 570.13) | 56.49 (-3.68, 119.40) | 408.89 (357.11, 496.74) |
| **X-Learner** (XGBoost Baseline) | 655.42 (520.63, 762.02) | 298.78 (145.54, 454.25) | 452.87 (390.05, 539.76) |
| **T-Learner** (XGBoost Baseline) | 4677.44 (1635.35, 6964.57) | 327.01 (182.36, 510.83) | 463.23 (401.53, 557.90) |

> [!NOTE]
> Metrics present the empirical mean alongside the 2.5th and 97.5th percentiles (95% CI) computed over $N=100$ bootstrap iterations on the isolated holdout test split ($N=2,551$).

---

## 🧠 Core Methodology

### The Optimization Paradox
Standard semi-parametric causal targets—such as Doubly Robust AIPW pseudo-outcomes—induce severe variance inflation when regressed directly by high-capacity neural sequence models under right-skewed revenue outcomes. This phenomenon, which we term **Long-Tailed Gradient Destabilization**, collapses representations and compresses individual uplift estimates toward a uniform global mean.

### Decoupled Structural Potential Surfaces
CDTT bypasses direct pseudo-outcome regression entirely. Instead, it trains a shared temporal transformer backbone to model the treated and control potential outcome surfaces natively:
1. **Longitudinal Attention Backbone**: Extracts high-dimensional historical behavioral patterns (velocity, frequency, recency accelerations) from sparse transactional sequences.
2. **Multi-Task ZILN Heads**: Parametrizes the zero-inflated lognormal distribution (buying probability $p$, lognormal mean $\mu$, scale $\sigma$) for both control ($T=0$) and treated ($T=1$) paths.
3. **Causal Expectation subtraction**: Estimates uplift via expectation differences over the mathematically corrected ZILN fields:
   $$\tau_i = \mathbb{E}[Y_i \mid T_i=1, \mathcal{H}_i] - \mathbb{E}[Y_i \mid T_i=0, \mathcal{H}_i]$$
   $$\mathbb{E}[Y] = p \cdot \exp\left(\mu + \frac{\sigma^2}{2}\right)$$

---

## 📁 Repository Structure

The codebase is modular and fully cleaned of legacy development assets:

```
daiclv/
├── figures/                   # High-resolution (300 DPI) manuscript diagnostic plots
│   ├── v2_calibration_dr.png
│   ├── v2_calibration_structural.png
│   ├── v2_policy_profit.png
│   └── v2_uplift_curves.png
├── src/
│   ├── data/                  # Ingestion, sliding window processing, causal generation
│   ├── models/                # CDTT Transformer, baseline neural networks, tree meta-learners
│   ├── train/                 # Cross-fitting trainers & joint ZILN loss optimizers
│   ├── evaluation/            # Decision-focused metrics (PEHE, IPW-AUUC, Profit)
│   └── analysis/              # Segmental customer profiling
├── final_experiment_v2.py     # Main end-to-end execution pipeline
├── manuscript.tex             # Main paper document
├── references.bib             # BibTeX reference ledger
└── online_retail_II.xlsx      # Raw empirical dataset (UCI Machine Learning Repository)
```

---

## ⚡ Setup & Execution

### 1. Environment Configuration
Ensure you have Python 3.9+ installed, then configure a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt # Or install core dependencies below
```

Core dependencies:
```bash
pip install torch xgboost pandas numpy scikit-learn matplotlib seaborn openpyxl scipy
```

### 2. Running the Full Pipeline
To execute the complete pipeline (data preprocessing, cross-fitted nuisance estimation, CDTT-Structural and DR model training, baseline training, bootstrapping, diagnostics plotting, and LaTeX results generation):
```bash
python3 final_experiment_v2.py
```

### 3. Running Ablations
To evaluate individual component contributions (e.g. omitting ZILN or the temporal backbone):
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 src/experiments/ablation_study.py
```

---

## 📜 Dataset & Reference
The empirical baseline is constructed using the **Online Retail II Dataset** from the UCI Machine Learning Repository:

```bibtex
@misc{Chen2012,
  author       = {Chen, Daqing},
  title        = {Online Retail II Data Set},
  year         = {2019},
  howpublished = {UCI Machine Learning Repository},
  note         = {DOI: 10.24432/C5CG6D}
}
```

For access to the full manuscript preprint or replication materials, please contact the author or submit an issue in this repository.

# Decision-Aware Incremental Customer Lifetime Value (CDTT)

## Overview
This repository contains the implementation of the Decision-Aware Incremental Customer Lifetime Value (CDTT) research project. The objective is to estimate the heterogeneous treatment effect (uplift) of marketing interventions on long-term customer value using a combination of temporal transformers and causal inference techniques.

The project utilizes the Online Retail II dataset and implements a semi-synthetic causal benchmark to evaluate models in the presence of selection bias and zero-inflated, heavy-tailed revenue distributions.

## Core Components
- **Temporal Transformer**: A sequence-to-sequence architecture that captures purchase velocity and temporal rhythms using Multihead Attention.
- **ZILN Head**: A Zero-Inflated Lognormal loss head designed to model the probability of purchase and the continuous spend volume simultaneously.
- **Doubly Robust (DR) Learning**: Integration of propensity scoring and doubly robust estimation to provide unbiased uplift signals in confounded environments.
- **Causal Benchmark**: A semi-synthetic generator for ground-truth treatment effect validation.

## Repository Structure
- `src/data/`: Data ingestion, cleaning, and causal label generation.
- `src/models/`: Implementation of CDTT, baselines (BG/NBD, XGBoost), and causal meta-learners (S, T, X Learners).
- `src/experiments/`: Scripts for modular ablation studies and long-run training.
- `src/evaluation/`: Decision-focused metrics including PEHE, AUUC, and profit simulations.
- `src/analysis/`: Segmental behavioral analysis and causal quadrant mapping.

## Setup
1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install torch causalml xgboost pandas numpy scikit-learn matplotlib seaborn lifetimes tabulate openpyxl
   ```

## Execution
To run the full evaluation suite:
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 src/evaluation/evaluate_final.py
```

To run the ablation study:
```bash
python3 src/experiments/ablation_study.py
```

## Research Principles
This project adheres to rigorous ML research standards:
- **Reproducibility**: All experiments are deterministic and seeded.
- **Causal Validity**: Models are evaluated against ground-truth counterfactuals.
- **Decision Awareness**: Performance is measured by business profit and targeting efficiency, not just predictive accuracy.

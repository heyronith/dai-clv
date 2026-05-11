# Decision-Aware Incremental Customer Lifetime Value

## A Causal-Distributional Temporal Framework for Marketing Intervention Targeting

**Short title:** CDTT: Causal-Distributional Temporal Transformers for Incremental Customer Lifetime Value  
**Duration:** 16 weeks  
**Research area:** Applied machine learning, causal inference, customer analytics  
**Primary output:** Reproducible paper, codebase, benchmarks, and arXiv-ready manuscript

---

## 1. Executive Summary

This project develops a decision-aware machine learning framework for estimating incremental customer lifetime value (CLV) under marketing interventions. Instead of predicting only which customers will spend the most, the project estimates which customers are likely to generate additional value because of a specific action, such as an email campaign, discount, push notification, or retargeting intervention.

The proposed framework, CDTT, combines temporal customer behavior modeling, zero-inflated distributional revenue forecasting, causal uplift estimation, and profit-aware decision evaluation. The intended research contribution is not merely a new deep learning architecture; it is a reproducible framework for making better customer targeting decisions under uncertainty.

The final deliverables include a cleaned research dataset pipeline, semi-synthetic causal benchmark, baseline models, CDTT implementation, evaluation suite, GitHub repository, and manuscript suitable for arXiv and workshop submission.

---

## 2. Core Research Question

Most CLV models answer:

```text
How much will this customer spend in the future?
```

This project asks a more decision-relevant question:

```text
How much additional future value will this customer generate because of a specific intervention?
```

Research question:

> Can a model estimate incremental customer lifetime value, not just predicted customer lifetime value, while accounting for temporal behavior, zero-inflated revenue, heavy-tailed spend, treatment-selection bias, and uncertainty?

---

## 3. Research Objective

Develop and evaluate a causal-distributional temporal modeling framework that estimates:

```text
tau_i = E[Y_i(1) - Y_i(0) | H_i]
```

| Symbol | Meaning |
|---|---|
| H_i | Customer behavioral history before the intervention |
| T_i = 1 | Customer receives a marketing intervention |
| T_i = 0 | Customer does not receive the intervention |
| Y_i(1) | Future CLV if treated |
| Y_i(0) | Future CLV if untreated |
| tau_i | Incremental CLV, also called uplift |

The goal is better budget allocation, uplift targeting, and profit-aware customer decisions, not only lower prediction error.

---

## 4. Research Gap and Contribution

Existing CLV research is fragmented. Classical CLV models are interpretable but often rely on simplified recency, frequency, and monetary summaries. ML models improve predictive accuracy but often remain correlation-based. Deep sequence models capture temporal behavior but may ignore treatment effects. Uplift models estimate intervention impact but often focus on conversion rather than long-horizon customer value.

This project contributes a unified framework for:

- Estimating incremental CLV rather than only predicted CLV.
- Modeling zero-spend customers and heavy-tailed high-value customers with a distributional CLV head.
- Representing customer behavior as a temporal event sequence.
- Correcting for treatment-selection bias using causal uplift and doubly robust methods.
- Evaluating success through profit-aware decision metrics, not only statistical prediction metrics.

Clean novelty statement:

> We propose a unified and reproducible framework for estimating distributional incremental CLV under marketing interventions, combining temporal customer representations, zero-inflated revenue modeling, doubly robust causal estimation, and profit-aware policy evaluation.

---

## 5. Project Scope

| In Scope | Out of Scope |
|---|---|
| Offline modeling and evaluation using public datasets | Live deployment into a production marketing platform |
| Semi-synthetic treatment-effect benchmark | Claiming real causal lift without treatment/control data |
| Single-treatment and binary-treatment settings | Full multi-action reinforcement learning system |
| Revenue and profit-aware uplift evaluation | Financial forecasting beyond customer-level CLV |
| arXiv/workshop-ready research manuscript | Guarantee of top-tier conference acceptance |

---

## 6. Datasets

| Dataset | Primary Use | Reason |
|---|---|---|
| Online Retail II | Sequential CLV modeling and semi-synthetic treatment simulation | Contains customer-level purchase histories suitable for temporal revenue modeling. |
| Hillstrom Email Marketing | Real randomized marketing treatment with spend/conversion outcome | Useful for validating uplift and campaign targeting logic. |
| Criteo Uplift | Large-scale treatment/control uplift benchmark | Useful for testing scalability and uplift modeling performance. |

Online Retail II should not be treated as a real causal dataset because it does not contain randomized marketing treatment assignment. It is used for temporal CLV modeling and for controlled semi-synthetic causal experiments.

---

## 7. Proposed Framework: CDTT

CDTT stands for Causal-Distributional Temporal Transformer. The framework has four modules.

### 7.1 Module 1: Temporal Customer Encoder

```text
H_i = {e_i1, e_i2, ..., e_it}
h_i = Transformer(H_i)
```

Each event may include purchase amount, time gap, product or category embedding, quantity, country, return flag, discount flag, and basket-level features. The plan should also test TCN, GRU, and LSTM encoders because Transformers may not always dominate on small or medium transaction datasets.

### 7.2 Module 2: Distributional CLV Head

Future CLV is usually zero-inflated and heavy-tailed. Use a Zero-Inflated Lognormal head:

```text
P(Y_i = 0 | h_i) = pi_i
log(Y_i | Y_i > 0, h_i) ~ Normal(mu_i, sigma_i^2)
Model output = (pi_i, mu_i, sigma_i)
```

### 7.3 Module 3: Potential Outcome Heads

```text
Y_i(0) ~ ZILN(pi_0i, mu_0i, sigma_0i)
Y_i(1) ~ ZILN(pi_1i, mu_1i, sigma_1i)
tau_i = E[Y_i(1) | H_i] - E[Y_i(0) | H_i]
```

### 7.4 Module 4: Doubly Robust Correction

For observational or confounded treatment settings, estimate treatment propensity:

```text
e_i = P(T_i = 1 | H_i)
```

The model then uses propensity-weighted, doubly robust, or DML-style correction to reduce bias under observed confounding. The causal claim must remain conditional on standard assumptions such as ignorability, overlap, and correct measurement of relevant confounders.

---

## 8. Research Hypotheses

| Hypothesis | Claim | Evaluation |
|---|---|---|
| H1 | Distributional modeling improves CLV calibration. | Zero-spend prediction, high-value ranking, decile calibration, normalized Gini. |
| H2 | Temporal sequence modeling improves customer representation. | Transformer/TCN/GRU/LSTM vs static RFM and tabular baselines. |
| H3 | Causal correction improves uplift estimation under confounding. | PEHE, ATE error, CATE error, Qini, AUUC under synthetic/confounded regimes. |
| H4 | Incremental CLV targeting improves profit. | Top-k ROI, policy value, net incremental profit, regret against oracle policy. |

---

## 9. Experimental Design

### 9.1 Semi-Synthetic Causal Benchmark

Create semi-synthetic treatment effects on Online Retail II. This benchmark provides known ground-truth CATE while preserving realistic transaction histories.

1. Build untreated baseline outcome using real future revenue/profit or a smoothed baseline function: `Y_i(0) = g(H_i) + epsilon_i`.
2. Define heterogeneous treatment effect as a nonlinear function of customer behavior.
3. Generate treated outcome: `Y_i(1) = Y_i(0) + tau_i + epsilon_i`.
4. Assign treatment under randomized, confounded, and policy-based regimes.
5. Generate observed outcome: `Y_i = T_i * Y_i(1) + (1 - T_i) * Y_i(0)`.

| Regime | Treatment Assignment | Purpose |
|---|---|---|
| A: Randomized | `T_i ~ Bernoulli(0.5)` | Clean experimental benchmark. |
| B: Confounded | `P(T_i = 1 | X_i) = sigmoid(w^T X_i)` | Simulates selection bias in real campaigns. |
| C: Policy-based | Treat high-RFM or high-churn-risk customers | Simulates legacy marketing rules. |

### 9.2 Baselines

| Baseline Group | Models |
|---|---|
| Classical CLV | BG/NBD + Gamma-Gamma |
| Tabular ML | Elastic Net, Random Forest, XGBoost, LightGBM |
| Deep predictive | MLP with ZILN, GRU, LSTM, TCN, Transformer |
| Causal/uplift | S-Learner, T-Learner, X-Learner, DR-Learner, Causal Forest, Uplift Random Forest, revenue uplift model |

### 9.3 Evaluation Metrics

| Metric Type | Metrics |
|---|---|
| Predictive | MAE, RMSE, normalized Gini, decile calibration, top-decile lift, zero-spend AUC, high-value ranking quality |
| Causal | PEHE, ATE error, CATE error, Qini coefficient, AUUC, uplift by decile, treatment-control profit difference |
| Decision | Expected campaign profit, net incremental revenue, top-k targeting ROI, regret against oracle policy, budget-constrained policy value |
| Uncertainty | Prediction interval coverage, interval width, calibration error, selective targeting performance under uncertainty |

### 9.4 Required Ablations

| Ablation | Question Answered |
|---|---|
| CDTT without Transformer | Does temporal sequence modeling help? |
| CDTT without ZILN | Does distributional revenue modeling help? |
| CDTT without causal correction | Does doubly robust correction help? |
| CDTT without uncertainty | Does calibrated uncertainty improve decisions? |
| CDTT using RFM only | Is the sequence encoder actually useful? |
| CDTT using MSE instead of NLL | Is ZILN likelihood necessary? |

---

## 10. 16-Week Project Timeline

| Phase | Weeks | Focus | Core Deliverables |
|---|---|---|---|
| 1 | 1-2 | Literature review and problem formalization | Literature review draft, notation, paper skeleton, research questions. |
| 2 | 3-5 | Data engineering | Processed datasets, train/validation/test splits, dataset cards. |
| 3 | 5-6 | Semi-synthetic causal benchmark | Treatment generator, known CATE, randomized/confounded/policy regimes. |
| 4 | 7-8 | Baseline models | Classical, ML, sequence, and uplift baseline results. |
| 5 | 9-11 | CDTT development | Temporal encoder, ZILN heads, potential outcome heads, DR correction. |
| 6 | 12-13 | Evaluation and ablation | Metric tables, ablations, ROI simulation, uncertainty results. |
| 7 | 14 | Interpretability and business analysis | Customer segment analysis, error analysis, attention/feature importance. |
| 8 | 15-16 | Manuscript and release | Paper PDF, GitHub repo, reproducibility package, arXiv-ready submission. |

---

## 11. Phase-by-Phase Work Plan

### Phase 1: Literature Review and Problem Formalization (Weeks 1-2)

**Objective:** Build the theoretical foundation and define the research problem precisely.

**Tasks:**

- Review BG/NBD, Pareto/NBD, Gamma-Gamma, RFM-based ML, sequence CLV, ZILN CLV, revenue uplift, and doubly robust causal ML.
- Formalize predictive CLV, incremental CLV, treatment effect, decision policy, and profit function.
- Create the manuscript skeleton and notation table.

**Outputs:**

- 4-6 page literature review draft.
- Formal problem statement.
- Initial paper outline.

### Phase 2: Data Engineering (Weeks 3-5)

**Objective:** Prepare clean, reproducible datasets for temporal CLV and causal uplift evaluation.

**Tasks:**

- Process Online Retail II into observation windows of 6, 9, 12, and 18 months with prediction windows of 3 and 6 months.
- Build customer event sequences, RFM features, time-gap features, product/category features, return/cancellation handling, and revenue/profit labels.
- Prepare Hillstrom and Criteo uplift datasets for treatment/control experiments.

**Outputs:**

- Data processing pipeline.
- Train/validation/test splits.
- Dataset cards documenting assumptions and limitations.

### Phase 3: Semi-Synthetic Causal Benchmark (Weeks 5-6)

**Objective:** Create a benchmark with realistic transaction histories and known treatment-effect ground truth.

**Tasks:**

- Define untreated baseline outcome Y_i(0).
- Define nonlinear heterogeneous treatment effect tau_i.
- Generate Y_i(1), treatment assignments, and observed outcomes.
- Implement randomized, confounded, and policy-based regimes.

**Outputs:**

- Benchmark generator.
- Known CATE labels.
- Documentation for reproducibility.

### Phase 4: Baseline Models (Weeks 7-8)

**Objective:** Establish strong baselines before evaluating CDTT.

**Tasks:**

- Train classical CLV models.
- Train tabular ML and deep predictive models.
- Train causal and uplift baselines.
- Create baseline result tables.

**Outputs:**

- Baseline scripts.
- Baseline result table.
- Initial performance ranking.

### Phase 5: CDTT Model Development (Weeks 9-11)

**Objective:** Build the proposed model architecture and training pipeline.

**Tasks:**

- Implement temporal encoder.
- Implement ZILN distributional heads.
- Implement potential outcome heads.
- Implement propensity and doubly robust correction components.
- Train model under multiple experimental regimes.

**Outputs:**

- CDTT model code.
- Config files.
- Training logs.
- Saved checkpoints.

### Phase 6: Evaluation and Ablation (Weeks 12-13)

**Objective:** Test whether each part of the framework improves prediction, causal estimation, and decision quality.

**Tasks:**

- Run predictive, causal, decision, and uncertainty metrics.
- Run required ablations.
- Compare targeting by predicted CLV versus predicted incremental CLV.
- Create figures and tables for the manuscript.

**Outputs:**

- Final result tables.
- Ablation study.
- ROI simulation.
- Uncertainty analysis.

### Phase 7: Interpretability and Business Analysis (Week 14)

**Objective:** Explain what the model learned and how it supports marketing decisions.

**Tasks:**

- Analyze attention or feature importance.
- Segment customers into sure things, persuadables, lost causes, and sleeping dogs.
- Analyze errors for whales and zero-spend customers.
- Document fairness, privacy, and deployment limits.

**Outputs:**

- Interpretability visuals.
- Business segment analysis.
- Limitations section draft.

### Phase 8: Manuscript and Release (Weeks 15-16)

**Objective:** Prepare the research for public release and submission.

**Tasks:**

- Write final manuscript.
- Clean code and README.
- Add reproducibility instructions.
- Prepare arXiv and workshop submission package.

**Outputs:**

- Paper PDF.
- GitHub repository.
- Reproducibility package.
- Submission checklist.

---

## 12. Final Deliverables

| Deliverable | Description |
|---|---|
| Research manuscript | Paper with introduction, related work, problem formulation, method, experiments, results, ablations, limitations, and conclusion. |
| CDTT codebase | PyTorch implementation of temporal encoder, ZILN heads, potential outcome heads, and causal correction modules. |
| Benchmark generator | Semi-synthetic treatment-effect generator with randomized, confounded, and policy-based regimes. |
| Baseline suite | Classical CLV, ML, sequence, and uplift model baselines. |
| Evaluation package | Predictive, causal, decision, and uncertainty metrics. |
| Reproducibility package | Environment file, configs, scripts, result logs, and README. |

---

## 13. Repository Structure

```text
cdtt-incremental-clv/
  README.md
  paper/
  data_processing/
  configs/
  models/
    transformer_encoder.py
    ziln_head.py
    causal_heads.py
    dr_learner.py
  baselines/
    bgnbd_gamma_gamma.py
    xgboost.py
    lightgbm.py
    causal_forest.py
    dr_learner.py
  experiments/
  evaluation/
    clv_metrics.py
    uplift_metrics.py
    roi_simulation.py
  notebooks/
  requirements.txt
```

---

## 14. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Semi-synthetic benchmark may be criticized as artificial. | Use Hillstrom and Criteo as real treatment/control validation and present Online Retail II as a controlled benchmark, not real causal proof. |
| Transformer may not outperform simpler models. | Position CDTT as a framework and include TCN/GRU/LSTM/LightGBM comparisons. Do not rely solely on Transformer superiority. |
| Causal claims may be overinterpreted. | State assumptions clearly: ignorability, overlap, consistency, and no unmeasured confounding in observational settings. |
| Model may appear overengineered. | Run ablations to prove which modules matter. |
| Revenue lift may not equal profit lift. | Evaluate net incremental profit after campaign and discount costs. |
| arXiv is not peer review. | Use arXiv for dissemination, then submit to appropriate workshop or applied ML venue. |

---

## 15. Publication Strategy

| Stage | Target | Goal |
|---|---|---|
| Stage 1 | arXiv preprint | Establish public priority and share reproducible work. |
| Stage 2 | KDD, RecSys, CIKM, WWW, AdKDD, or causal ML workshop | Receive peer feedback and validate applied relevance. |
| Stage 3 | Extended journal/conference version | Strengthen with additional data, multi-treatment setup, or real partner campaign. |

---

## 16. Success Criteria

- CDTT improves profit-aware targeting compared with targeting by predicted CLV alone.
- ZILN distributional heads improve calibration and high-value ranking compared with MSE/MAE heads.
- Causal correction improves uplift estimation under confounded treatment assignment.
- Ablation results demonstrate that each major component has measurable value.
- The codebase and benchmark generator are reproducible by an external reader.
- The final manuscript is honest about limitations and avoids exaggerated SOTA claims.

---

## 17. Final Positioning

This project should be positioned as a causal decision framework for customer value optimization, not simply as a new neural network. The strongest claim is that estimating incremental CLV under uncertainty leads to better marketing decisions than ranking customers by predicted CLV alone.

Recommended final claim:

> CDTT improves marketing decision quality by estimating distributional incremental customer lifetime value under intervention, uncertainty, and treatment-selection bias.

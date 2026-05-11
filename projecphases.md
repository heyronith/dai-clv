I have categorized the 29 core tasks into the eight phases defined in our project plan. These tasks represent the specific milestones required to move from data to a published manuscript.

Phase 1: Formalization & Literature (4 Tasks)

Task 1.1: Conduct a targeted literature review of ZILN-based CLV and revenue uplift modeling.

Task 1.2: Formalize the structural causal model (SCM) and the potential outcome framework for τ 
i
​	
 .

Task 1.3: Define the profit-aware objective function, explicitly accounting for intervention costs.

Task 1.4: Create the LaTeX paper skeleton and notation table for consistency across the team.

Phase 2: Data Engineering (4 Tasks)

Task 2.1: Implement the sliding window pipeline for Online Retail II (6-18 month observation windows).

Task 2.2: Engineer temporal features (inter-purchase time, sequential basket embeddings, seasonality).

Task 2.3: Process the Hillstrom and Criteo benchmarks for real-world (randomized) validation.

Task 2.4: Produce Dataset Cards documenting treatment assignment mechanisms and selection bias.

Phase 3: Semi-Synthetic Benchmark (4 Tasks)

Task 3.1: Develop the Y 
i
​	
 (0) outcome generator based on baseline customer trajectories.

Task 3.2: Define a nonlinear τ 
i
​	
  function to test the model's ability to capture heterogeneous effects.

Task 3.3: Build treatment assignment modules for Randomized, Confounded, and Policy-based regimes.

Task 3.4: Validate ground-truth CATE labels for PEHE and AUUC metric calculation.

Phase 4: Baseline Implementation (4 Tasks)

Task 4.1: Fit classical probabilistic models (BG/NBD + Gamma-Gamma) as a baseline for interpretability.

Task 4.2: Train tabular ML models (XGBoost/LightGBM) using aggregated RFM features.

Task 4.3: Train predictive-only sequence models (LSTM/TCN) to isolate temporal benefits.

Task 4.4: Implement standard uplift Meta-Learners (S, T, X, and DR-Learners).

Phase 5: CDTT Development (5 Tasks)

Task 5.1: Design the Temporal Encoder (Transformer/TCN) with multi-head attention over customer events.

Task 5.2: Implement the ZILN loss head to handle zero-inflation and the heavy tail of revenue.

Task 5.3: Integrate potential outcome heads (Y(1) and Y(0)) with shared representation layers.

Task 5.4: Implement the Doubly Robust (DR) correction module for observational data regimes.

Task 5.5: Build the training loop with early stopping based on validation uplift (Qini/AUUC).

Phase 6: Evaluation & Ablation (3 Tasks)

Task 6.1: Execute the full metric suite (Predictive, Causal, Decision, and Uncertainty metrics).

Task 6.2: Perform the 6 planned ablation runs to justify each component of CDTT.

Task 6.3: Conduct ROI simulations comparing predicted CLV targeting vs. incremental CLV targeting.

Phase 7 & 8: Interpretation & Submission (5 Tasks)

Task 7.1: Perform attention attribution analysis to identify key temporal predictors.

Task 7.2: Segment the test set into the four causal quadrants (Sure Things, Persuadables, etc.).

Task 7.3: Write the discussion on "Failure Modes" and "Threats to Validity."

Task 7.4: Finalize the manuscript, targeting KDD or a top-tier Causal ML workshop.

Task 7.5: Release the GitHub repository with a single-command reproducibility script.
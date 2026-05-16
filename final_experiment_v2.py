import torch
import numpy as np
import pandas as pd
import json
from src.data.online_retail_dataset import OnlineRetailDataProcessor, PipelineConfig, CausalDataset, causal_collate_fn
from src.data.causal_benchmark import CausalGenerator
from src.data.data_utils import DataSplitter
from src.train.nuisance_trainer import NuisanceTrainer
from src.train.dr_utils import generate_dr_targets
from src.train.train_dr_model import DRTrainer, TemporalTransformer
from src.evaluation.baselines import BaselineEvaluator
from src.evaluation.metrics import CausalMetrics
from src.evaluation.visualization import VisualizationSuite


def run_v2_pipeline():
    # ===== Audit Fix C4 + M4: Deterministic seeding =====
    np.random.seed(42)
    torch.manual_seed(42)

    print("=== STARTING CDTT V2 FINAL EXPERIMENT PIPELINE ===")

    # 1. Data Generation
    config = PipelineConfig(slide_step_months=3)
    processor = OnlineRetailDataProcessor('online_retail_II.xlsx', config)
    processor.load_and_clean()
    processor.create_sliding_windows()
    generator = CausalGenerator(processor.processed_samples)
    causal_df = generator.generate_causal_labels(kappa=2.0)

    true_ate = causal_df['tau'].mean()
    print(f"Ground Truth ATE: {true_ate:.4f}")

    # 2. Data Splitting (60/20/20)
    splitter = DataSplitter(causal_df, seed=42)
    train_idx, val_idx, test_idx = splitter.split()

    # 3. Nuisance Estimation (Cross-Fitting K=5)
    print("\n--- Phase 1: Nuisance Cross-Fitting (K=5) ---")
    # Create CausalDataset without DR targets (not yet computed)
    causal_dataset = CausalDataset(processor.processed_samples, causal_df)
    folds = splitter.get_cross_fit_indices(k=5)
    n_trainer = NuisanceTrainer(epochs=10)
    nuisance_df = n_trainer.train_cross_fit(causal_dataset, causal_df, folds)

    # 4. DR-Learner Training
    print("\n--- Phase 2: DR-Learner Training ---")
    df_with_dr = generate_dr_targets(causal_df, nuisance_df)

    # Create new CausalDataset WITH DR targets for the DR trainer
    dr_dataset = CausalDataset(processor.processed_samples, causal_df,
                               dr_targets=df_with_dr['dr_target'].values)
    model = TemporalTransformer().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dr_trainer = DRTrainer(lmbda=0.8, epochs=15)
    dr_history = dr_trainer.train_dr_model(model, dr_dataset, train_idx, val_idx)

    # 5. Baselines
    print("\n--- Phase 3: Baseline Training (Out-of-Sample) ---")
    evaluator = BaselineEvaluator(seed=42)
    full_rfm_df = evaluator.prepare_rfm_features(processor.processed_samples, causal_df)

    b_train_df = full_rfm_df.loc[train_idx]
    b_test_df = full_rfm_df.loc[test_idx]

    t_learner = evaluator.train_t_learner(b_train_df)
    x_learner = evaluator.train_x_learner(b_train_df)

    # 6. Final Evaluation & Bootstrapping
    print("\n--- Phase 4: Final Metrics & Bootstrapping (N=100) ---")

    # CDTT predictions on test set (un-scaled)
    model.eval()
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dr_dataset, test_idx),
        batch_size=64, shuffle=False, collate_fn=causal_collate_fn
    )

    cdtt_test_preds = []      # From uplift head (out_u)
    cdtt_struct_preds = []    # From structural T-Learner heads: E[Y|T=1] - E[Y|T=0]
    with torch.no_grad():
        for seq, static, label, lengths, t_batch, dr_target_batch in test_loader:
            seq = seq.to(dr_trainer.device)
            static = static.to(dr_trainer.device)
            lengths = lengths.to(dr_trainer.device)

            (logits_c, mu_c, sigma_c), (logits_t, mu_t, sigma_t), _, tau_hat_norm = model(seq, static, lengths)

            # 1. DR Uplift Head prediction (un-scaled)
            tau_hat = tau_hat_norm * dr_trainer.target_std + dr_trainer.target_mean
            cdtt_test_preds.append(tau_hat.cpu().numpy())

            # 2. Structural prediction: E[Y|T=1] - E[Y|T=0] via ZILN expectation
            # Clamp exponent to prevent exp() overflow (consistent with nuisance trainer)
            exp_t = torch.clamp(mu_t + 0.5 * torch.pow(sigma_t, 2), max=15.0)
            exp_c = torch.clamp(mu_c + 0.5 * torch.pow(sigma_c, 2), max=15.0)
            ey1 = torch.sigmoid(logits_t) * torch.exp(exp_t)
            ey0 = torch.sigmoid(logits_c) * torch.exp(exp_c)
            tau_struct = ey1 - ey0
            cdtt_struct_preds.append(tau_struct.cpu().numpy())

    cdtt_test_preds = np.concatenate(cdtt_test_preds).flatten()
    cdtt_struct_preds = np.concatenate(cdtt_struct_preds).flatten()

    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression
    rho_struct, _ = spearmanr(cdtt_struct_preds, causal_df.loc[test_idx, 'tau'])
    rho_dr, _ = spearmanr(cdtt_test_preds, causal_df.loc[test_idx, 'tau'])
    reg = LinearRegression().fit(cdtt_struct_preds.reshape(-1, 1), causal_df.loc[test_idx, 'tau'])
    slope = reg.coef_[0]
    
    print(f"\n[Diagnostic] CDTT-DR Uplift Head | mean={cdtt_test_preds.mean():.2f}, std={cdtt_test_preds.std():.2f}, rho={rho_dr:.4f}")
    print(f"[Diagnostic] CDTT-Structural     | mean={cdtt_struct_preds.mean():.2f}, std={cdtt_struct_preds.std():.2f}, rho={rho_struct:.4f}, slope={slope:.4f}")
    print(f"[Diagnostic] Ground Truth τ      | mean={causal_df.loc[test_idx, 'tau'].mean():.2f}")

    # Baseline predictions
    t_test_preds = evaluator.predict_uplift(t_learner, b_test_df, method='T-Learner')
    x_test_preds = evaluator.predict_uplift(x_learner, b_test_df, method='X-Learner')

    pred_dict = {
        'CDTT-Structural': cdtt_struct_preds,
        'CDTT-DR': cdtt_test_preds,
        'T-Learner': t_test_preds,
        'X-Learner': x_test_preds
    }

    # Assemble test metrics dataframe
    test_df_metrics = causal_df.loc[test_idx].copy()
    test_df_metrics['propensity_pred'] = nuisance_df.loc[test_idx, 'propensity_pred']

    metrics_tool = CausalMetrics()
    final_report, curve_storage = metrics_tool.bootstrap_metrics(test_df_metrics, pred_dict, n_iterations=100)

    # 7. Visualization (Audit Fix S4: real uplift curves)
    print("\n--- Phase 5: Generating Manuscript Visuals ---")
    viz = VisualizationSuite()

    # Figure 1: Real IPW-adjusted uplift curves with bootstrapped CI bands
    viz.plot_uplift_curves(curve_storage)

    # Figure 2: Calibration — both CDTT variants
    viz.plot_calibration(test_df_metrics['tau'].values, cdtt_struct_preds, "CDTT-Structural", output_path='v2_calibration_structural.png')
    viz.plot_calibration(test_df_metrics['tau'].values, cdtt_test_preds, "CDTT-DR", output_path='v2_calibration_dr.png')

    # Figure 3: Policy Value at multiple K thresholds (all bootstrapped — Audit Fix M1)
    viz.plot_policy_profit(final_report)

    # 8. LaTeX Table Reporting
    print("\n" + "=" * 60)
    print("MASTER RESULTS TABLE (LaTeX Format)")
    print("=" * 60)

    header = "Model & PEHE $\\downarrow$ & IPW-AUUC $\\uparrow$ & Profit@20\\% $\\uparrow$ \\\\"
    print(header)
    print("\\hline")

    for name in ['T-Learner', 'X-Learner', 'CDTT-Structural', 'CDTT-DR']:
        m = final_report[name]
        pehe_str = f"{m['pehe']['mean']:.4f} ({m['pehe']['ci_low']:.4f}, {m['pehe']['ci_high']:.4f})"
        auuc_str = f"{m['auuc']['mean']:.4f} ({m['auuc']['ci_low']:.4f}, {m['auuc']['ci_high']:.4f})"
        profit_str = f"{m['profit_20pct']['mean']:.2f} ({m['profit_20pct']['ci_low']:.2f}, {m['profit_20pct']['ci_high']:.2f})"
        print(f"{name} & {pehe_str} & {auuc_str} & {profit_str} \\\\")
        if name == 'X-Learner':
            print("\\hline")

    print("=" * 60)

    # Save results
    with open('v2_test_results.json', 'w') as f:
        json.dump(final_report, f, indent=4)
    print("\n[Pipeline Complete] Results saved to v2_test_results.json")


if __name__ == "__main__":
    run_v2_pipeline()

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.data.online_retail_dataset import OnlineRetailDataProcessor, PipelineConfig, OnlineRetailDataset, collate_fn
from src.data.causal_benchmark import CausalGenerator
from src.models.cdtt_dr import TemporalTransformer
from src.models.baselines import BaselineEvaluator
from src.models.meta_learners import MetaLearnerEvaluator
from src.evaluation.causal_metrics import calculate_pehe, get_uplift_curve, calculate_auuc, simulate_profit
from tabulate import tabulate

def get_cdtt_predictions(model, loader):
    model.eval()
    all_tau = []
    with torch.no_grad():
        for seq, static, label, lengths in loader:
            (logits_c, mu_c, sigma_c), (logits_t, mu_t, sigma_t), _ = model(seq, static, lengths)
            
            # Expected value: p * exp(mu + 0.5 * sigma^2)
            y_hat_t = torch.sigmoid(logits_t) * torch.exp(mu_t + 0.5 * sigma_t**2)
            y_hat_c = torch.sigmoid(logits_c) * torch.exp(mu_c + 0.5 * sigma_c**2)
            
            tau = (y_hat_t - y_hat_c).cpu().numpy().flatten()
            all_tau.extend(tau)
    return np.array(all_tau)

def run_final_evaluation():
    print("=== Phase 6: Comprehensive Causal Evaluation ===")
    
    # 1. Setup Data
    config = PipelineConfig(slide_step_months=3)
    processor = OnlineRetailDataProcessor('online_retail_II.xlsx', config)
    processor.load_and_clean()
    processor.create_sliding_windows()
    generator = CausalGenerator(processor.processed_samples)
    causal_df = generator.generate_causal_labels(kappa=2.0)
    
    dataset = OnlineRetailDataset(processor.processed_samples)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    # 2. Get CDTT Predictions
    print("Loading CDTT Champion...")
    cdtt_model = TemporalTransformer()
    cdtt_model.load_state_dict(torch.load('cdtt_dr_model.pth'))
    cdtt_tau = get_cdtt_predictions(cdtt_model, loader)
    
    # 3. Get X-Learner Predictions
    print("Training X-Learner Baseline...")
    evaluator = BaselineEvaluator(causal_df, processor.processed_samples)
    meta_eval = MetaLearnerEvaluator(evaluator.data)
    meta_preds = meta_eval.train_evaluate()
    x_learner_tau = meta_preds['X-Learner']
    
    # 4. Get XGBoost Predictions (Predictor)
    print("Training XGBoost Predictor...")
    xgb_preds = evaluator.run_xgboost_baseline()
    # For a pure predictor, 'uplift' is just the prediction itself (naive assumption)
    xgb_tau = xgb_preds 
    
    # 5. Metrics & Curves
    true_tau = causal_df['tau'].values
    y_obs = causal_df['y_obs'].values
    treatment = causal_df['treatment'].values
    
    models = {
        'CDTT (Champion)': cdtt_tau,
        'X-Learner': x_learner_tau,
        'XGBoost (Predictor)': xgb_tau,
        'Random': np.random.uniform(size=len(true_tau))
    }
    
    metrics_rows = []
    plt.figure(figsize=(10, 7))
    
    for name, pred_tau in models.items():
        pehe = calculate_pehe(true_tau, pred_tau)
        curve = get_uplift_curve(y_obs, treatment, pred_tau)
        auuc = calculate_auuc(curve)
        profit = simulate_profit(true_tau, pred_tau, cost_per_treatment=5.0, top_k=0.2)
        
        metrics_rows.append({
            'Model': name,
            'PEHE (Lower is Better)': pehe,
            'AUUC (Higher is Better)': auuc,
            'Profit (Top 20%)': profit
        })
        
        plt.plot(np.linspace(0, 100, len(curve)), curve, label=f"{name} (AUUC: {auuc:.0f})")
        
    # Baseline for random
    plt.axline((0, 0), (100, curve[-1]), color='gray', linestyle='--', label='Theoretical Random')
    
    plt.title('Final Uplift Curves: CDTT vs Baselines')
    plt.xlabel('Population Treated (%)')
    plt.ylabel('Cumulative Uplift (Revenue)')
    plt.legend()
    plt.grid(True)
    plt.savefig('final_uplift_comparison.png')
    
    print("\n=== Table 3: Decision-Aware Performance Table ===")
    print(tabulate(metrics_rows, headers='keys', tablefmt='pipe', showindex=False))
    
    # Save results
    pd.DataFrame(metrics_rows).to_csv('final_performance_metrics.csv', index=False)
    print("\nEvaluation complete. Plot saved to 'final_uplift_comparison.png'.")

if __name__ == "__main__":
    run_final_evaluation()

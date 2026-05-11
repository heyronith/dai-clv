import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from tabulate import tabulate

from src.data.online_retail_dataset import OnlineRetailDataProcessor, PipelineConfig, OnlineRetailDataset, collate_fn
from src.data.causal_benchmark import CausalGenerator
from src.models.baselines import BaselineEvaluator
from src.models.deep_predictive import LSTMRegressor, TCNRegressor, train_deep_model, get_deep_predictions
from src.models.meta_learners import MetaLearnerEvaluator

def calculate_auuc(y_obs, treatment, pred_uplift):
    """Calculates AUUC (Area Under Uplift Curve) normalized by max possible AUUC."""
    df = pd.DataFrame({'y': y_obs, 't': treatment, 'pred': pred_uplift})
    df = df.sort_values('pred', ascending=False).reset_index(drop=True)
    
    n = len(df)
    n_t = df['t'].sum()
    n_c = n - n_t
    
    if n_t == 0 or n_c == 0:
        return 0.0
    
    # Cumulative sums
    y_t = (df['y'] * df['t']).cumsum() / n_t
    y_c = (df['y'] * (1 - df['t'])).cumsum() / n_c
    
    uplift_curve = y_t - y_c
    auuc = uplift_curve.sum() / n
    return auuc

def run_evaluation():
    print("=== Phase 4: Full Baseline Evaluation ===")
    
    # 1. Setup Data
    config = PipelineConfig(slide_step_months=3)
    processor = OnlineRetailDataProcessor('online_retail_II.xlsx', config)
    processor.load_and_clean()
    processor.create_sliding_windows()
    
    generator = CausalGenerator(processor.processed_samples)
    causal_df = generator.generate_causal_labels(kappa=2.0)
    
    # 2. Run Baselines (Task 4.1 & 4.2)
    evaluator = BaselineEvaluator(causal_df, processor.processed_samples)
    base_results = evaluator.evaluate() # DataFrame with 3 models
    
    # Store predictions for AUUC
    bg_gg_preds = evaluator.run_classical_baseline()
    xgb_preds = evaluator.run_xgboost_baseline()
    
    # 3. Run Deep Sequential (Task 4.3)
    print("\nTraining Deep Sequential Models...")
    dataset = OnlineRetailDataset(processor.processed_samples)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    # LSTM
    print("Fitting LSTM...")
    lstm_model = LSTMRegressor(input_dim=8)
    train_deep_model(lstm_model, loader, epochs=5)
    lstm_preds = get_deep_predictions(lstm_model, test_loader)
    
    # TCN
    print("Fitting TCN...")
    tcn_model = TCNRegressor(input_dim=8)
    train_deep_model(tcn_model, loader, epochs=5)
    tcn_preds = get_deep_predictions(tcn_model, test_loader)
    
    # 4. Run Meta-Learners (Task 4.4)
    print("\nTraining Meta-Learners...")
    meta_eval = MetaLearnerEvaluator(evaluator.data)
    meta_preds = meta_eval.train_evaluate() # Dict of 3 models
    
    # 5. Consolidate and Calculate AUUC
    all_models = {
        'BG/NBD + GG': bg_gg_preds,
        'XGBoost': xgb_preds,
        'Naive Uplift': np.full(len(causal_df), causal_df['y_obs'].mean()),
        'LSTM (MSE)': lstm_preds,
        'TCN (MSE)': tcn_preds,
        'S-Learner': meta_preds['S-Learner'],
        'T-Learner': meta_preds['T-Learner'],
        'X-Learner': meta_preds['X-Learner']
    }
    
    y_obs = causal_df['y_obs'].values
    treatment = causal_df['treatment'].values
    true_tau = causal_df['tau'].values
    true_ate = true_tau.mean()
    
    final_rows = []
    for name, preds in all_models.items():
        pred_ate = preds.mean()
        ate_error = np.abs(pred_ate - true_ate)
        rmse = np.sqrt(mean_squared_error(y_obs, preds))
        spearman, _ = spearmanr(preds, true_tau)
        auuc = calculate_auuc(y_obs, treatment, preds)
        
        final_rows.append({
            'Model': name,
            'ATE Error': ate_error,
            'RMSE (Y_obs)': rmse,
            'Tau Spearman': spearman,
            'AUUC': auuc
        })
    
    results_df = pd.DataFrame(final_rows)
    print("\n=== Table 2: Consolidated Baseline & Meta-Learner Performance ===")
    print(tabulate(results_df, headers='keys', tablefmt='pipe', showindex=False))
    results_df.to_csv('consolidated_results.csv', index=False)

if __name__ == "__main__":
    run_evaluation()

import pandas as pd
import numpy as np
import xgboost as xgb
from lifetimes import BetaGeoFitter, GammaGammaFitter
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from src.data.online_retail_dataset import OnlineRetailDataProcessor, PipelineConfig
from src.data.causal_benchmark import CausalGenerator
import matplotlib.pyplot as plt

class BaselineEvaluator:
    def __init__(self, causal_df: pd.DataFrame, processor_samples: list):
        self.df = causal_df
        self.samples = processor_samples
        self._prepare_rfm()

    def _prepare_rfm(self):
        """Prepares RFM features for lifetimes and XGBoost."""
        rfm_rows = []
        for s in self.samples:
            # Reconstruct RFM from sequences
            # Note: lifetimes needs frequency as repeat purchases
            raw_spend = s['raw_sequence'][:, 0]
            # Since daily aggregated, length is number of days with purchases
            frequency = len(raw_spend) - 1 
            # Monetary value is average of transactions
            monetary_value = raw_spend.mean() if len(raw_spend) > 0 else 0
            
            # Recency and T from original dates in sample logic (relative to obs_end)
            # tenure feature is (obs_end - first_purchase)
            T = s['static_features'][0] * 365.0
            # delta_t sum is age of last purchase relative to first
            recency = s['sequence'][:, 2].sum() * 365.0
            
            rfm_rows.append({
                'customer_id': s['customer_id'],
                'frequency': frequency,
                'recency': recency,
                'T': T,
                'monetary_value': monetary_value,
                'tenure': s['static_features'][0],
                'avg_spend': s['static_features'][1],
                'has_returned': s['static_features'][2],
                'velocity': s['sequence'][:, 3].mean()
            })
        self.rfm_df = pd.DataFrame(rfm_rows)
        # Direct assignment to avoid many-to-many merge explosion
        self.data = self.rfm_df.copy()
        self.data['y_obs'] = self.df['y_obs'].values
        self.data['tau'] = self.df['tau'].values
        self.data['treatment'] = self.df['treatment'].values

    def run_classical_baseline(self):
        """Task 4.1: BG/NBD + Gamma-Gamma."""
        print("Fitting BG/NBD + Gamma-Gamma...")
        # 1. BG/NBD for frequency
        bgf = BetaGeoFitter(penalizer_coef=0.0)
        bgf.fit(self.data['frequency'], self.data['recency'], self.data['T'])
        
        # 2. Gamma-Gamma for monetary value
        # Filtering for customers with frequency > 0 and monetary_value > 0
        returning_customers = self.data[(self.data['frequency'] > 0) & (self.data['monetary_value'] > 0)]
        ggf = GammaGammaFitter(penalizer_coef=0.0)
        ggf.fit(returning_customers['frequency'], returning_customers['monetary_value'])
        
        # 3. Predict expected CLV for 6 months (approx 180 days)
        # expected_number_of_purchases_up_to_time
        exp_purchases = bgf.conditional_expected_number_of_purchases_up_to_time(180, self.data['frequency'], self.data['recency'], self.data['T'])
        # conditional_expected_average_profit - only works for positive monetary values
        # We'll use a small epsilon for non-positive values to avoid errors
        safe_monetary = self.data['monetary_value'].clip(lower=0.01)
        exp_monetary = ggf.conditional_expected_average_profit(self.data['frequency'], safe_monetary)
        
        preds = exp_purchases * exp_monetary
        # Set predictions to 0 for those with 0 or negative past spend
        preds[self.data['monetary_value'] <= 0] = 0
        return preds.fillna(0)

    def run_xgboost_baseline(self):
        """Task 4.2: XGBoost Regressor."""
        print("Training XGBoost...")
        X = self.data[['frequency', 'recency', 'T', 'monetary_value', 'tenure', 'avg_spend', 'has_returned', 'velocity']]
        y = self.data['y_obs']
        
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        model.fit(X, y)
        return model.predict(X)

    def evaluate(self):
        # 1. Run Models
        bg_gg_preds = self.run_classical_baseline()
        xgb_preds = self.run_xgboost_baseline()
        
        # 2. Naive Uplift (Heuristic)
        # Global mean difference between treated and untreated
        treated_mean = self.data[self.data['treatment'] == 1]['y_obs'].mean()
        untreated_mean = self.data[self.data['treatment'] == 0]['y_obs'].mean()
        naive_ate = treated_mean - untreated_mean
        naive_preds = np.full(len(self.data), naive_ate)
        
        true_ate = self.data['tau'].mean()
        
        results = []
        for name, preds in zip(['BG/NBD + GG', 'XGBoost', 'Naive Uplift'], [bg_gg_preds, xgb_preds, naive_preds]):
            # ATE Error: |predicted ATE - true ATE|
            # Note: BG/NBD and XGBoost predict Y_obs, not Tau. 
            # For this comparison, we treat their mean prediction as a naive ATE proxy 
            # OR we simply note their predictive error for Y_obs.
            pred_ate = preds.mean()
            ate_error = np.abs(pred_ate - true_ate)
            
            rmse = np.sqrt(mean_squared_error(self.data['y_obs'], preds))
            spearman, _ = spearmanr(preds, self.data['tau'])
            
            results.append({
                'Model': name,
                'ATE Error': ate_error,
                'Predictive RMSE': rmse,
                'Tau Spearman': spearman
            })
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    import os
    # 1. Load Data
    config = PipelineConfig(slide_step_months=3)
    processor = OnlineRetailDataProcessor('online_retail_II.xlsx', config)
    processor.load_and_clean()
    processor.create_sliding_windows()
    
    # 2. Causal Labels
    generator = CausalGenerator(processor.processed_samples)
    causal_df = generator.generate_causal_labels(kappa=2.0)
    
    # 3. Baselines
    evaluator = BaselineEvaluator(causal_df, processor.processed_samples)
    results_table = evaluator.evaluate()
    
    print("\n=== Baseline Performance Table ===")
    print(results_table.to_markdown(index=False))
    
    # Save results
    results_table.to_csv('baseline_results.csv', index=False)

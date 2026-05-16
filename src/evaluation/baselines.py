import pandas as pd
import numpy as np
import xgboost as xgb
from src.data.data_utils import DataSplitter
from typing import Dict, List

class BaselineEvaluator:
    """
    Implements Task 4.1: Research-Grade Baseline Evaluation.
    Includes RFM flattening, T-Learner, and X-Learner.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.features = ['frequency', 'recency', 'T', 'monetary_value', 'tenure', 'avg_spend', 'has_returned', 'velocity']

    def prepare_rfm_features(self, samples: List[Dict], causal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Flattens temporal sequences into static RFM features for baseline models.
        """
        rfm_rows = []
        for s in samples:
            raw_spend = s['raw_sequence'][:, 0]
            frequency = len(raw_spend) - 1 
            monetary_value = raw_spend.mean() if len(raw_spend) > 0 else 0
            T = s['static_features'][0] * 365.0
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
        
        rfm_df = pd.DataFrame(rfm_rows)
        # Merge with treatment and observed outcomes from causal_df
        full_df = rfm_df.merge(causal_df[['customer_id', 'treatment', 'y_obs']], on='customer_id')
        return full_df

    def train_t_learner(self, train_df: pd.DataFrame) -> Dict:
        """
        Standard T-Learner using XGBoost.
        """
        X_t = train_df[train_df['treatment'] == 1][self.features]
        y_t = train_df[train_df['treatment'] == 1]['y_obs']
        
        X_c = train_df[train_df['treatment'] == 0][self.features]
        y_c = train_df[train_df['treatment'] == 0]['y_obs']
        
        m1 = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=self.seed)
        m0 = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=self.seed)
        
        m1.fit(X_t, y_t)
        m0.fit(X_c, y_c)
        
        return {'m1': m1, 'm0': m0}

    def train_x_learner(self, train_df: pd.DataFrame) -> Dict:
        """
        Manual implementation of the X-Learner (Kunzel et al. 2019).
        """
        # Stage 1: Fit mu1 and mu0 (T-Learner)
        t_learner = self.train_t_learner(train_df)
        m1, m0 = t_learner['m1'], t_learner['m0']
        
        # Stage 2: Impute treatment effects
        treated_df = train_df[train_df['treatment'] == 1]
        control_df = train_df[train_df['treatment'] == 0]
        
        # D1 = Y(1) - mu0(X)
        d1 = treated_df['y_obs'] - m0.predict(treated_df[self.features])
        # D0 = mu1(X) - Y(0)
        d0 = m1.predict(control_df[self.features]) - control_df['y_obs']
        
        # Stage 3: Fit tau1 and tau0
        tau1 = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=self.seed)
        tau0 = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=self.seed)
        
        tau1.fit(treated_df[self.features], d1)
        tau0.fit(control_df[self.features], d0)
        
        # Propensity model (simple logistic regression or XGB)
        g = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=self.seed)
        g.fit(train_df[self.features], train_df['treatment'])
        
        return {'tau1': tau1, 'tau0': tau0, 'propensity': g}

    def predict_uplift(self, model_dict: Dict, test_df: pd.DataFrame, method: str = 'T-Learner') -> np.ndarray:
        X_test = test_df[self.features]
        
        if method == 'T-Learner':
            return model_dict['m1'].predict(X_test) - model_dict['m0'].predict(X_test)
        
        elif method == 'X-Learner':
            tau1_pred = model_dict['tau1'].predict(X_test)
            tau0_pred = model_dict['tau0'].predict(X_test)
            # Propensity weighting: tau = g(x)*tau0 + (1-g(x))*tau1
            g_x = model_dict['propensity'].predict_proba(X_test)[:, 1]
            return g_x * tau0_pred + (1 - g_x) * tau1_pred
            
        return np.zeros(len(test_df))

if __name__ == "__main__":
    from src.data.online_retail_dataset import OnlineRetailDataProcessor, PipelineConfig
    from src.data.causal_benchmark import CausalGenerator
    
    # 1. Setup Data
    config = PipelineConfig(slide_step_months=3)
    processor = OnlineRetailDataProcessor('online_retail_II.xlsx', config)
    processor.load_and_clean()
    processor.create_sliding_windows()
    generator = CausalGenerator(processor.processed_samples)
    causal_df = generator.generate_causal_labels(kappa=2.0)
    
    # 2. Split (Strict Isolation)
    splitter = DataSplitter(causal_df)
    train_idx, val_idx, test_idx = splitter.split()
    
    evaluator = BaselineEvaluator()
    full_rfm_df = evaluator.prepare_rfm_features(processor.processed_samples, causal_df)
    
    train_df = full_rfm_df.loc[train_idx]
    test_df = full_rfm_df.loc[test_idx]
    
    # 3. Train Baselines
    print("Training T-Learner (XGBoost)...")
    t_learner = evaluator.train_t_learner(train_df)
    t_preds = evaluator.predict_uplift(t_learner, test_df, method='T-Learner')
    
    print("Training X-Learner (Manual XGBoost)...")
    x_learner = evaluator.train_x_learner(train_df)
    x_preds = evaluator.predict_uplift(x_learner, test_df, method='X-Learner')
    
    # 4. Save Results
    results = pd.DataFrame({
        'customer_id': test_df['customer_id'],
        't_learner_uplift': t_preds,
        'x_learner_uplift': x_preds
    })
    results.to_csv('baseline_preds.csv', index=False)
    print("Baseline predictions saved to baseline_preds.csv")

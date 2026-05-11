import pandas as pd
import numpy as np
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor
from xgboost import XGBRegressor

class MetaLearnerEvaluator:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.feature_cols = ['frequency', 'recency', 'T', 'monetary_value', 'tenure', 'avg_spend', 'has_returned', 'velocity']

    def train_evaluate(self):
        X = self.data[self.feature_cols].values
        y = self.data['y_obs'].values
        treatment = self.data['treatment'].values
        
        # Initialize base learner
        base_learner = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        
        # 1. S-Learner
        print("Training S-Learner...")
        s_learner = BaseSRegressor(learner=base_learner)
        s_learner.fit(X, treatment, y)
        s_ite = s_learner.predict(X).flatten()
        
        # 2. T-Learner
        print("Training T-Learner...")
        t_learner = BaseTRegressor(learner=base_learner)
        t_learner.fit(X, treatment, y)
        t_ite = t_learner.predict(X).flatten()
        
        # 3. X-Learner
        print("Training X-Learner...")
        x_learner = BaseXRegressor(learner=base_learner)
        x_learner.fit(X, treatment, y)
        x_ite = x_learner.predict(X).flatten()
        
        return {
            'S-Learner': s_ite,
            'T-Learner': t_ite,
            'X-Learner': x_ite
        }

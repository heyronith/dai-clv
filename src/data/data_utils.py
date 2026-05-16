import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from typing import List, Tuple, Dict

class DataSplitter:
    """
    Centralized Data Splitter for Causal CLV research.
    Implements Customer-Level Splitting and Cross-Fitting support.
    """
    def __init__(self, df: pd.DataFrame, seed: int = 42):
        self.df = df
        self.seed = seed
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        
    def split(self, train_size: float = 0.6, val_size: float = 0.2, test_size: float = 0.2) -> Tuple[List[int], List[int], List[int]]:
        """
        Splits the dataframe into Train, Val, and Test sets based on unique customer_id.
        Ensures strict isolation to prevent temporal/customer leakage.
        """
        unique_ids = self.df['customer_id'].unique()
        
        # First split: Train vs Temp (Val + Test)
        train_ids, temp_ids = train_test_split(
            unique_ids, 
            train_size=train_size, 
            random_state=self.seed
        )
        
        # Second split: Val vs Test
        val_relative_size = val_size / (val_size + test_size)
        val_ids, test_ids = train_test_split(
            temp_ids, 
            train_size=val_relative_size, 
            random_state=self.seed
        )
        
        # Map IDs back to indices in the original dataframe
        self.train_idx = self.df[self.df['customer_id'].isin(train_ids)].index.tolist()
        self.val_idx = self.df[self.df['customer_id'].isin(val_ids)].index.tolist()
        self.test_idx = self.df[self.df['customer_id'].isin(test_ids)].index.tolist()
        
        return self.train_idx, self.val_idx, self.test_idx

    def get_cross_fit_indices(self, k: int = 5) -> List[Tuple[List[int], List[int]]]:
        """
        Generates K-Fold indices within the Training set for cross-fitting (DML).
        Each fold is also customer-isolated.
        """
        if self.train_idx is None:
            raise ValueError("Split must be called before generating cross-fit indices.")
        
        train_df = self.df.loc[self.train_idx]
        unique_train_ids = train_df['customer_id'].unique()
        
        kf = KFold(n_splits=k, shuffle=True, random_state=self.seed)
        
        folds = []
        for train_fold_ids_idx, val_fold_ids_idx in kf.split(unique_train_ids):
            train_fold_ids = unique_train_ids[train_fold_ids_idx]
            val_fold_ids = unique_train_ids[val_fold_ids_idx]
            
            # Extract indices relative to the original dataframe
            train_fold_idx = train_df[train_df['customer_id'].isin(train_fold_ids)].index.tolist()
            val_fold_idx = train_df[train_df['customer_id'].isin(val_fold_ids)].index.tolist()
            
            folds.append((train_fold_idx, val_fold_idx))
            
        return folds

    def get_split_report(self) -> Dict:
        """Returns basic statistics about the splits."""
        if self.train_idx is None:
            return {"status": "not split"}
            
        return {
            "train": {"n_samples": len(self.train_idx), "n_customers": self.df.loc[self.train_idx, 'customer_id'].nunique()},
            "val": {"n_samples": len(self.val_idx), "n_customers": self.df.loc[self.val_idx, 'customer_id'].nunique()},
            "test": {"n_samples": len(self.test_idx), "n_customers": self.df.loc[self.test_idx, 'customer_id'].nunique()}
        }

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional
import datetime
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class PipelineConfig:
    observation_window_months: int = 12
    target_window_months: int = 6
    slide_step_months: int = 1
    min_transactions: int = 1 
    sequence_padding_value: float = 0.0

class OnlineRetailDataProcessor:
    def __init__(self, file_path: str, config: PipelineConfig):
        self.file_path = file_path
        self.config = config
        self.df = None
        self.processed_samples = []
        self.stats = {}
        self.customer_first_purchase = {}

    def load_and_clean(self):
        print(f"Loading data from {self.file_path}...")
        df1 = pd.read_excel(self.file_path, sheet_name='Year 2009-2010')
        df2 = pd.read_excel(self.file_path, sheet_name='Year 2010-2011')
        self.df = pd.concat([df1, df2], ignore_index=True)
        
        self.df = self.df.dropna(subset=['Customer ID'])
        self.df['Customer ID'] = self.df['Customer ID'].astype(int)
        self.df['Net_Spend'] = self.df['Quantity'] * self.df['Price']
        self.df['Date'] = pd.to_datetime(self.df['InvoiceDate']).dt.normalize()
        
        # Track first ever purchase for each customer for tenure calculation
        self.customer_first_purchase = self.df.groupby('Customer ID')['Date'].min().to_dict()
        
        daily_df = self.df.groupby(['Customer ID', 'Date']).agg(
            total_net_spend=('Net_Spend', 'sum'),
            total_quantity=('Quantity', 'sum'),
            transaction_count=('Invoice', 'nunique')
        ).reset_index()
        
        self.df = daily_df.sort_values(['Customer ID', 'Date'])
        print(f"Data processed: {len(self.df)} daily aggregated records.")

    def _safe_log1p(self, x):
        return np.sign(x) * np.log1p(np.abs(x))

    def create_sliding_windows(self):
        if self.df is None:
            self.load_and_clean()
            
        print("Creating sliding windows with advanced features...")
        min_date = self.df['Date'].min()
        max_date = self.df['Date'].max()
        
        current_obs_start = min_date
        samples = []
        
        while True:
            obs_end = current_obs_start + pd.DateOffset(months=self.config.observation_window_months)
            target_end = obs_end + pd.DateOffset(months=self.config.target_window_months)
            
            if target_end > max_date:
                break
            
            obs_mask = (self.df['Date'] >= current_obs_start) & (self.df['Date'] < obs_end)
            obs_period_data = self.df[obs_mask]
            
            target_mask = (self.df['Date'] >= obs_end) & (self.df['Date'] < target_end)
            target_period_data = self.df[target_mask]
            
            active_customers = obs_period_data['Customer ID'].unique()
            
            for cust_id in active_customers:
                cust_obs = obs_period_data[obs_period_data['Customer ID'] == cust_id].sort_values('Date')
                
                if len(cust_obs) < self.config.min_transactions:
                    continue
                
                cust_target_spend = target_period_data[target_period_data['Customer ID'] == cust_id]['total_net_spend'].sum()
                ziln_label = max(0.0, float(cust_target_spend))
                
                # --- Task 2.2: Temporal Feature Engineering ---
                cust_obs = cust_obs.copy()
                
                # 1. Delta T and Velocity
                cust_obs['delta_t'] = cust_obs['Date'].diff().dt.days.fillna(0)
                cust_obs['velocity'] = cust_obs['delta_t'].diff().fillna(0)
                
                # 2. Time-Cyclic Features
                months = cust_obs['Date'].dt.month
                day_of_week = cust_obs['Date'].dt.dayofweek
                
                cust_obs['sin_month'] = np.sin(2 * np.pi * months / 12)
                cust_obs['cos_month'] = np.cos(2 * np.pi * months / 12)
                cust_obs['sin_dow'] = np.sin(2 * np.pi * day_of_week / 7)
                cust_obs['cos_dow'] = np.cos(2 * np.pi * day_of_week / 7)
                
                # 3. Static Features
                first_purchase = self.customer_first_purchase[cust_id]
                tenure = (obs_end - first_purchase).days / 365.0
                avg_spend = cust_obs['total_net_spend'].mean()
                has_returned = float((cust_obs['total_net_spend'] < 0).any())
                
                static_features = np.array([tenure, self._safe_log1p(avg_spend), has_returned], dtype=np.float32)
                
                # 4. Sequence Tensor (Shape: L x 8)
                # [Spend, Quantity, Delta T, Velocity, sin_month, cos_month, sin_dow, cos_dow]
                features = np.stack([
                    self._safe_log1p(cust_obs['total_net_spend'].values),
                    self._safe_log1p(cust_obs['total_quantity'].values),
                    cust_obs['delta_t'].values / 365.0,
                    cust_obs['velocity'].values / 365.0,
                    cust_obs['sin_month'].values,
                    cust_obs['cos_month'].values,
                    cust_obs['sin_dow'].values,
                    cust_obs['cos_dow'].values
                ], axis=1)
                
                samples.append({
                    'customer_id': cust_id,
                    'sequence': features,
                    'raw_sequence': cust_obs[['total_net_spend', 'total_quantity']].values,
                    'static_features': static_features,
                    'label': ziln_label,
                    'has_returns': has_returned > 0
                })
            
            current_obs_start += pd.DateOffset(months=self.config.slide_step_months)
            print(f"Processed window ending {obs_end.date()}, samples: {len(samples)}")
            
        self.processed_samples = samples
        self._calculate_stats()

    def _calculate_stats(self):
        labels = [s['label'] for s in self.processed_samples]
        seq_lens = [len(s['sequence']) for s in self.processed_samples]
        has_returns = [s['has_returns'] for s in self.processed_samples]
        
        self.stats = {
            'total_samples': len(labels),
            'zero_inflation_ratio': sum(1 for x in labels if x == 0) / len(labels) if labels else 0,
            'avg_seq_len': np.mean(seq_lens) if seq_lens else 0,
            'max_seq_len': np.max(seq_lens) if seq_lens else 0,
            'return_sequence_ratio': sum(has_returns) / len(labels) if labels else 0,
            'label_distribution': labels,
            'seq_len_distribution': seq_lens
        }

    def generate_correlation_report(self):
        print("\n=== Feature Correlation Report (Task 2.2) ===")
        
        # Prepare data for correlation
        rows = []
        feature_names = ['Spend', 'Quantity', 'Delta_T', 'Velocity', 'sin_month', 'cos_month', 'sin_dow', 'cos_dow']
        static_names = ['Tenure', 'Avg_Spend', 'Has_Returned']
        
        for s in self.processed_samples:
            # For sequences, we'll take the mean of features to correlate with target
            seq_mean = s['sequence'].mean(axis=0)
            row = list(seq_mean) + list(s['static_features']) + [s['label']]
            rows.append(row)
            
        corr_df = pd.DataFrame(rows, columns=feature_names + static_names + ['Target_Revenue'])
        correlations = corr_df.corr()['Target_Revenue'].sort_values(ascending=False)
        
        print(correlations)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('feature_correlation_report.png')
        print("\nCorrelation heatmap saved to 'feature_correlation_report.png'.")

    def generate_report(self):
        print("\n=== Data Pipeline Report ===")
        print(f"Total Sequences: {self.stats['total_samples']}")
        print(f"Zero-Inflation: {self.stats['zero_inflation_ratio']:.2%}")
        print(f"Avg Seq Len: {self.stats['avg_seq_len']:.2f}")
        print(f"Return Ratio: {self.stats['return_sequence_ratio']:.2%}")
        self.generate_correlation_report()

class OnlineRetailDataset(Dataset):
    def __init__(self, samples: List[dict]):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        sequence = torch.FloatTensor(sample['sequence'])
        static = torch.FloatTensor(sample['static_features'])
        label = torch.FloatTensor([sample['label']])
        return sequence, static, label

def collate_fn(batch):
    sequences, static_features, labels = zip(*batch)
    lengths = torch.LongTensor([len(seq) for seq in sequences])
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    static_features = torch.stack(static_features)
    labels = torch.stack(labels)
    return padded_sequences, static_features, labels, lengths

if __name__ == "__main__":
    config = PipelineConfig(
        observation_window_months=12,
        target_window_months=6,
        slide_step_months=3 
    )
    
    processor = OnlineRetailDataProcessor('online_retail_II.xlsx', config)
    processor.load_and_clean()
    processor.create_sliding_windows()
    processor.generate_report()
    
    dataset = OnlineRetailDataset(processor.processed_samples)
    if len(dataset) > 0:
        seq, static, label = dataset[0]
        print(f"\nSample 0 Check:")
        print(f"Sequence shape: {seq.shape} (Expected: [L, 8])")
        print(f"Static features shape: {static.shape} (Expected: [3])")
        print(f"Label: {label.item()}")

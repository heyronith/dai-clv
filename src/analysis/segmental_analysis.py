import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from src.data.online_retail_dataset import OnlineRetailDataProcessor, PipelineConfig, OnlineRetailDataset, collate_fn
from src.data.causal_benchmark import CausalGenerator
from src.experiments.ablation_study import AblationModel
from tabulate import tabulate

def get_predictions(model, loader):
    model.eval()
    device = next(model.parameters()).device
    all_tau = []
    with torch.no_grad():
        for seq, static, label, lengths in loader:
            out = model(seq.to(device), static.to(device), lengths.to(device))
            if model.mode != 'no-ziln':
                (logits_c, mu_c, sigma_c), (logits_t, mu_t, sigma_t), _ = out
                y_t = torch.sigmoid(logits_t) * torch.exp(mu_t + 0.5 * sigma_t**2)
                y_c = torch.sigmoid(logits_c) * torch.exp(mu_c + 0.5 * sigma_c**2)
            else:
                y_c, y_t, _ = out
            tau = (y_t - y_c).cpu().numpy().flatten()
            all_tau.extend(tau)
    return np.array(all_tau)

def run_segmental_analysis():
    print("=== Phase 6.3: Segmental Behavioral Analysis ===")
    
    # 1. Setup Data
    config = PipelineConfig(slide_step_months=3)
    processor = OnlineRetailDataProcessor('online_retail_II.xlsx', config)
    processor.load_and_clean()
    processor.create_sliding_windows()
    causal_df = CausalGenerator(processor.processed_samples).generate_causal_labels()
    dataset = OnlineRetailDataset(processor.processed_samples)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    # 2. Get Models (We'll re-train quickly or use the architecture logic)
    # For speed and consistency with Task 6.2, we'll assume the findings
    from src.experiments.ablation_study import train_ablation
    
    print("Training Full CDTT (ZILN)...")
    model_full = train_ablation('full', loader, causal_df, epochs=5)
    print("Training No-ZILN (MSE)...")
    model_mse = train_ablation('no-ziln', loader, causal_df, epochs=5)
    
    cdtt_tau = get_predictions(model_full, loader)
    mse_tau = get_predictions(model_mse, loader)
    
    # 3. Ground Truth Quadrant Mapping
    # Threshold for "Response": Median of Y(0)
    threshold = causal_df['y0'].median()
    
    def map_quadrant(row):
        y0 = row['y0']
        y1 = row['y1']
        if y0 > y1: return 'Sleeping Dog'
        if y0 > threshold and y1 > threshold: return 'Sure Thing'
        if y0 <= threshold and y1 > threshold: return 'Persuadable'
        return 'Lost Cause'
        
    causal_df['quadrant'] = causal_df.apply(map_quadrant, axis=1)
    
    # 4. Behavioral Profiling (Top 10%)
    top_k = 0.1
    n_top = int(len(causal_df) * top_k)
    
    idx_cdtt = np.argsort(cdtt_tau)[::-1][:n_top]
    idx_mse = np.argsort(mse_tau)[::-1][:n_top]
    
    # Whales = Top 1% of Y(0)
    whale_threshold = causal_df['y0'].quantile(0.99)
    causal_df['is_whale'] = causal_df['y0'] > whale_threshold
    
    # Add behavioral features for analysis
    # We can get these from the processor or causal_df
    # causal_df already has tenure and avg_spend
    # Let's add velocity and has_returned from the original samples
    causal_df['velocity'] = [s['sequence'][:, 3].mean() for s in processor.processed_samples]
    causal_df['has_returned'] = [s['has_returns'] for s in processor.processed_samples]
    
    profile_cdtt = causal_df.iloc[idx_cdtt]
    profile_mse = causal_df.iloc[idx_mse]
    
    report = []
    for name, df in [('CDTT (ZILN)', profile_cdtt), ('No-ZILN (MSE)', profile_mse)]:
        report.append({
            'Model': name,
            'Avg Tenure (Years)': df['tenure'].mean(),
            'Avg Velocity': df['velocity'].mean(),
            'Return Rate (%)': df['has_returned'].mean() * 100,
            'Whales Targeted (%)': df['is_whale'].mean() * 100,
            'Persuadables (%)': (df['quadrant'] == 'Persuadable').mean() * 100,
            'Sure Things (%)': (df['quadrant'] == 'Sure Thing').mean() * 100,
            'Sleeping Dogs (%)': (df['quadrant'] == 'Sleeping Dog').mean() * 100
        })
        
    print("\n=== Targeting Profile Report ===")
    print(tabulate(report, headers='keys', tablefmt='pipe', showindex=False))
    
    # Quadrant Distribution Plot
    plt.figure(figsize=(10, 6))
    causal_df['quadrant'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Global Causal Quadrant Distribution (Ground Truth)')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('causal_quadrant_distribution.png')
    
if __name__ == "__main__":
    run_segmental_analysis()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from src.data.online_retail_dataset import OnlineRetailDataProcessor, PipelineConfig, OnlineRetailDataset, collate_fn
from src.data.causal_benchmark import CausalGenerator
from src.models.cdtt_dr import TemporalTransformer, ZILNLoss
from src.evaluation.causal_metrics import calculate_auuc, get_uplift_curve, simulate_profit
from tabulate import tabulate

class AblationModel(nn.Module):
    def __init__(self, mode='full', seq_dim=8, static_dim=3, hidden_dim=64):
        super(AblationModel, self).__init__()
        self.mode = mode
        
        # Components
        self.seq_embedding = nn.Linear(seq_dim, hidden_dim)
        self.static_embedding = nn.Linear(static_dim, hidden_dim)
        
        if mode != 'no-temporal':
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.fc_combined = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.fc_combined = nn.Linear(hidden_dim, hidden_dim) # Only static
            
        # Heads
        out_dim = 3 if mode != 'no-ziln' else 1
        self.head_control = nn.Linear(hidden_dim, out_dim)
        self.head_treated = nn.Linear(hidden_dim, out_dim)
        self.head_propensity = nn.Linear(hidden_dim, 1)

    def forward(self, seq, static, lengths):
        if self.mode != 'no-temporal':
            seq_emb = self.seq_embedding(seq)
            mask = torch.arange(seq.size(1), device=seq.device)[None, :] >= lengths[:, None]
            seq_out = self.transformer(seq_emb, src_key_padding_mask=mask)
            pooled_seq = torch.stack([seq_out[i, :lengths[i]].mean(dim=0) for i in range(seq.size(0))])
            static_emb = F.relu(self.static_embedding(static))
            combined = torch.cat([pooled_seq, static_emb], dim=-1)
        else:
            combined = F.relu(self.static_embedding(static))
            
        hidden = F.relu(self.fc_combined(combined))
        out_c = self.head_control(hidden)
        out_t = self.head_treated(hidden)
        p_score = torch.sigmoid(self.head_propensity(hidden))
        
        if self.mode != 'no-ziln':
            def process(out):
                return out[:, 0:1], out[:, 1:2], F.softplus(out[:, 2:3]) + 1e-4
            return process(out_c), process(out_t), p_score
        else:
            return out_c, out_t, p_score

def train_ablation(model_mode, loader, causal_df, epochs=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AblationModel(mode=model_mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    ziln_criterion = ZILNLoss()
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()
    
    treatment_tensor = torch.FloatTensor(causal_df['treatment'].values).unsqueeze(1).to(device)
    label_tensor = torch.FloatTensor(causal_df['y_obs'].values).unsqueeze(1).to(device)
    
    best_loss = float('inf')
    early_stop_count = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (seq, static, label, lengths) in enumerate(loader):
            batch_indices = range(i * loader.batch_size, min((i+1) * loader.batch_size, len(causal_df)))
            t_batch = treatment_tensor[batch_indices]
            l_batch = label_tensor[batch_indices]
            
            optimizer.zero_grad()
            out = model(seq.to(device), static.to(device), lengths.to(device))
            
            # Loss Calculation
            if model_mode != 'no-ziln':
                (logits_c, mu_c, sigma_c), (logits_t, mu_t, sigma_t), p_score = out
                loss_t = ziln_criterion(logits_t, mu_t, sigma_t, l_batch).mean()
                loss_c = ziln_criterion(logits_c, mu_c, sigma_c, l_batch).mean()
            else:
                y_c, y_t, p_score = out
                loss_t = mse_criterion(y_t, l_batch)
                loss_c = mse_criterion(y_c, l_batch)
                
            p_loss = bce_criterion(p_score, t_batch)
            
            # T-Learner / DR weighting
            mask_t = (t_batch == 1).float()
            mask_c = (t_batch == 0).float()
            
            if model_mode == 'no-dr':
                loss = (mask_t * loss_t + mask_c * loss_c).mean()
            else:
                loss = (mask_t * loss_t + mask_c * loss_c).mean() + p_loss
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_count = 0
        else:
            early_stop_count += 1
            
        if early_stop_count > 10: break # Simple early stopping
        
    return model

def evaluate_model(model, loader, causal_df):
    model.eval()
    device = next(model.parameters()).device
    all_tau = []
    with torch.no_grad():
        for i, (seq, static, label, lengths) in enumerate(loader):
            out = model(seq.to(device), static.to(device), lengths.to(device))
            if model.mode != 'no-ziln':
                (logits_c, mu_c, sigma_c), (logits_t, mu_t, sigma_t), _ = out
                y_t = torch.sigmoid(logits_t) * torch.exp(mu_t + 0.5 * sigma_t**2)
                y_c = torch.sigmoid(logits_c) * torch.exp(mu_c + 0.5 * sigma_c**2)
            else:
                y_c, y_t, _ = out
            tau = (y_t - y_c).cpu().numpy().flatten()
            all_tau.extend(tau)
            
    pred_tau = np.array(all_tau)
    true_tau = causal_df['tau'].values
    y_obs = causal_df['y_obs'].values
    treatment = causal_df['treatment'].values
    
    auuc = calculate_auuc(get_uplift_curve(y_obs, treatment, pred_tau))
    profit = simulate_profit(true_tau, pred_tau, top_k=0.2)
    return auuc, profit

if __name__ == "__main__":
    # Setup
    config = PipelineConfig(slide_step_months=3)
    processor = OnlineRetailDataProcessor('online_retail_II.xlsx', config)
    processor.load_and_clean()
    processor.create_sliding_windows()
    causal_df = CausalGenerator(processor.processed_samples).generate_causal_labels()
    dataset = OnlineRetailDataset(processor.processed_samples)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    results = []
    modes = ['full', 'no-temporal', 'no-ziln', 'no-dr', 'long-run']
    
    # Base for full (from previous task 6.1)
    # We re-train here for consistency
    for mode in modes:
        print(f"\nRunning Ablation: {mode}...")
        epochs = 15 if mode != 'long-run' else 50
        m = train_ablation('full' if mode == 'long-run' else mode, loader, causal_df, epochs=epochs)
        auuc, profit = evaluate_model(m, loader, causal_df)
        results.append({'Mode': mode, 'AUUC': auuc, 'Profit': profit})
        
    # Generate Ablation Matrix
    df = pd.DataFrame(results)
    base_auuc = df[df['Mode'] == 'full']['AUUC'].values[0]
    base_profit = df[df['Mode'] == 'full']['Profit'].values[0]
    
    df['Δ AUUC (%)'] = (df['AUUC'] - base_auuc) / base_auuc * 100
    df['Δ Profit ($)'] = df['Profit'] - base_profit
    
    print("\n=== Table 4: Ablation Study Matrix ===")
    print(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))
    df.to_csv('ablation_matrix.csv', index=False)

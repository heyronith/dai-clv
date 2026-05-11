import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.data.online_retail_dataset import OnlineRetailDataset, collate_fn

class ZILNLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(ZILNLoss, self).__init__()
        self.eps = eps

    def forward(self, logits, mu, sigma, target):
        positive_mask = (target > 0).float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, positive_mask, reduction='none')
        y_pos = target.clamp(min=self.eps)
        log_y = torch.log(y_pos)
        term1 = -torch.log(sigma.clamp(min=self.eps))
        term2 = -0.5 * np.log(2 * np.pi)
        term3 = -((log_y - mu) ** 2) / (2 * (sigma ** 2).clamp(min=self.eps))
        lognormal_log_prob = term1 + term2 + term3
        regression_loss = -log_y - lognormal_log_prob
        loss = bce_loss + positive_mask * regression_loss
        return loss # Return per-sample loss for weighting

class TemporalTransformer(nn.Module):
    def __init__(self, seq_dim=8, static_dim=3, hidden_dim=64, nhead=4, num_layers=2):
        super(TemporalTransformer, self).__init__()
        self.seq_embedding = nn.Linear(seq_dim, hidden_dim)
        self.static_embedding = nn.Linear(static_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_combined = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Heads
        self.head_control = nn.Linear(hidden_dim, 3)
        self.head_treated = nn.Linear(hidden_dim, 3)
        self.head_propensity = nn.Linear(hidden_dim, 1) # Task 5.4: Propensity Head
        
    def forward(self, seq, static, lengths):
        seq_emb = self.seq_embedding(seq)
        mask = torch.arange(seq.size(1), device=seq.device)[None, :] >= lengths[:, None]
        seq_out = self.transformer(seq_emb, src_key_padding_mask=mask)
        pooled_seq = []
        for i in range(seq.size(0)):
            pooled_seq.append(seq_out[i, :lengths[i]].mean(dim=0))
        pooled_seq = torch.stack(pooled_seq)
        static_emb = F.relu(self.static_embedding(static))
        combined = torch.cat([pooled_seq, static_emb], dim=-1)
        hidden = F.relu(self.fc_combined(combined))
        
        out_c = self.head_control(hidden)
        out_t = self.head_treated(hidden)
        out_p = self.head_propensity(hidden)
        
        def process_head_out(out):
            logits = out[:, 0:1]
            mu = out[:, 1:2]
            sigma = F.softplus(out[:, 2:3]) + 1e-4
            return logits, mu, sigma
            
        return process_head_out(out_c), process_head_out(out_t), torch.sigmoid(out_p)

def train_cdtt_dr(model, loader, causal_df, epochs=20, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ziln_criterion = ZILNLoss()
    bce_criterion = nn.BCELoss()
    
    treatment_tensor = torch.FloatTensor(causal_df['treatment'].values).unsqueeze(1)
    
    history = {'total': [], 'ziln': [], 'propensity': [], 'dr_ate': []}
    
    model.train()
    for epoch in range(epochs):
        epoch_losses = {'total': 0, 'ziln': 0, 'prop': 0}
        batch_dr_ates = []
        
        for i, (seq, static, label, lengths) in enumerate(loader):
            batch_indices = range(i * loader.batch_size, min((i+1) * loader.batch_size, len(causal_df)))
            t_batch = treatment_tensor[batch_indices].to(seq.device)
            
            optimizer.zero_grad()
            (logits_c, mu_c, sigma_c), (logits_t, mu_t, sigma_t), p_score = model(seq, static, lengths)
            
            # 1. Propensity Loss
            prop_loss = bce_criterion(p_score, t_batch)
            
            # 2. ZILN Outcome Loss (T-Learner part)
            mask_t = (t_batch == 1).float()
            mask_c = (t_batch == 0).float()
            
            loss_t = ziln_criterion(logits_t, mu_t, sigma_t, label)
            loss_c = ziln_criterion(logits_c, mu_c, sigma_c, label)
            ziln_loss = (mask_t * loss_t + mask_c * loss_c).mean()
            
            # 3. Unified Loss
            total_loss = ziln_loss + prop_loss
            total_loss.backward()
            optimizer.step()
            
            # 4. Doubly Robust Signal (for monitoring/calibration)
            with torch.no_grad():
                # Clipping propensity for stability
                e = p_score.clamp(0.05, 0.95)
                # Predicted outcomes (expected value of ZILN = p * exp(mu + 0.5*sigma^2))
                y_hat_t = torch.sigmoid(logits_t) * torch.exp(mu_t + 0.5 * sigma_t**2)
                y_hat_c = torch.sigmoid(logits_c) * torch.exp(mu_c + 0.5 * sigma_c**2)
                
                # DR formula from prompt
                dr_t = (t_batch * label / e) - ((t_batch - e) * y_hat_t / e)
                dr_c = ((1 - t_batch) * label / (1 - e)) + ((t_batch - e) * y_hat_c / (1 - e))
                tau_dr = (dr_t - dr_c).mean().item()
                batch_dr_ates.append(tau_dr)
            
            epoch_losses['total'] += total_loss.item()
            epoch_losses['ziln'] += ziln_loss.item()
            epoch_losses['prop'] += prop_loss.item()
            
        n_batches = len(loader)
        history['total'].append(epoch_losses['total'] / n_batches)
        history['ziln'].append(epoch_losses['ziln'] / n_batches)
        history['propensity'].append(epoch_losses['prop'] / n_batches)
        history['dr_ate'].append(np.mean(batch_dr_ates))
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {history['total'][-1]:.4f} (Prop: {history['propensity'][-1]:.4f}, DR ATE: {history['dr_ate'][-1]:.2f})")
        
    return history

if __name__ == "__main__":
    from src.data.online_retail_dataset import OnlineRetailDataProcessor, PipelineConfig
    from src.data.causal_benchmark import CausalGenerator
    
    config = PipelineConfig(slide_step_months=3)
    processor = OnlineRetailDataProcessor('online_retail_II.xlsx', config)
    processor.load_and_clean()
    processor.create_sliding_windows()
    generator = CausalGenerator(processor.processed_samples)
    causal_df = generator.generate_causal_labels(kappa=2.0)
    
    dataset = OnlineRetailDataset(processor.processed_samples)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    model = TemporalTransformer()
    print("Starting CDTT Training with DR Correction...")
    history = train_cdtt_dr(model, loader, causal_df, epochs=15)
    
    # Causal Calibration Report
    true_ate = causal_df['tau'].mean()
    predicted_ate = history['dr_ate'][-1]
    
    print("\n=== Causal Calibration Report ===")
    print(f"Ground Truth ATE: {true_ate:.2f}")
    print(f"CDTT Predicted (DR) ATE: {predicted_ate:.2f}")
    print(f"ATE Bias: {predicted_ate - true_ate:.2f}")
    print(f"ATE Error Ratio: {abs(predicted_ate - true_ate) / true_ate:.2%}")
    print("================================\n")
    
    plt.figure(figsize=(10, 6))
    plt.axhline(y=true_ate, color='r', linestyle='--', label='Ground Truth ATE')
    plt.plot(history['dr_ate'], label='CDTT DR-ATE Estimate')
    plt.title('Causal Calibration: DR-ATE Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('ATE')
    plt.legend()
    plt.grid(True)
    plt.savefig('causal_calibration_report.png')
    print("Calibration report saved to 'causal_calibration_report.png'.")
    
    torch.save(model.state_dict(), 'cdtt_dr_model.pth')

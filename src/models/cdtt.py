import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.data.online_retail_dataset import OnlineRetailDataset, collate_fn

class ZILNLoss(nn.Module):
    """
    Zero-Inflated Lognormal Loss.
    logits: log-odds of having a positive outcome (y > 0).
    mu, sigma: parameters of the lognormal distribution for y > 0.
    """
    def __init__(self, eps=1e-6):
        super(ZILNLoss, self).__init__()
        self.eps = eps

    def forward(self, logits, mu, sigma, target):
        # target shape: [batch, 1]
        positive_mask = (target > 0).float()
        
        # 1. Classification part: cross-entropy for y > 0 vs y = 0
        # logits are for p = Prob(y > 0)
        bce_loss = F.binary_cross_entropy_with_logits(logits, positive_mask, reduction='none')
        
        # 2. Regression part: log-likelihood of lognormal for y > 0
        # PDF: 1/(y * sigma * sqrt(2pi)) * exp(-(log(y)-mu)^2 / 2sigma^2)
        # Log-PDF: -log(y) - log(sigma) - 0.5*log(2pi) - (log(y)-mu)^2 / (2*sigma^2)
        # Negative Log-Likelihood (NLL): log(y) + log(sigma) + 0.5*log(2pi) + (log(y)-mu)^2 / (2*sigma^2)
        
        y_pos = target.clamp(min=self.eps)
        log_y = torch.log(y_pos)
        
        # Log-likelihood terms
        term1 = torch.log(sigma.clamp(min=self.eps)) # +log(sigma)
        term2 = 0.5 * np.log(2 * np.pi)              # +0.5*log(2pi)
        term3 = ((log_y - mu) ** 2) / (2 * (sigma ** 2).clamp(min=self.eps)) # +quad
        
        # Total regression loss (NLL)
        regression_loss = log_y + term1 + term2 + term3
        
        loss = bce_loss + positive_mask * regression_loss
        
        return loss.mean(), bce_loss.mean(), (positive_mask * regression_loss).sum() / (positive_mask.sum() + self.eps)

class TemporalTransformer(nn.Module):
    def __init__(self, seq_dim=8, static_dim=3, hidden_dim=64, nhead=4, num_layers=2):
        super(TemporalTransformer, self).__init__()
        
        self.seq_embedding = nn.Linear(seq_dim, hidden_dim)
        self.static_embedding = nn.Linear(static_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_combined = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Two ZILN heads: one for T=0 (Control) and one for T=1 (Treated)
        # Each head outputs [logits, mu, sigma]
        self.head_control = nn.Linear(hidden_dim, 3)
        self.head_treated = nn.Linear(hidden_dim, 3)
        
    def forward(self, seq, static, lengths):
        # seq: [batch, seq_len, 8], static: [batch, 3]
        
        # 1. Embed sequence and apply transformer
        seq_emb = self.seq_embedding(seq) # [batch, seq_len, hidden]
        
        # Transformer mask for padding
        mask = torch.arange(seq.size(1), device=seq.device)[None, :] >= lengths[:, None]
        
        seq_out = self.transformer(seq_emb, src_key_padding_mask=mask)
        
        # Pool sequence output (mean pooling over active tokens)
        pooled_seq = []
        for i in range(seq.size(0)):
            pooled_seq.append(seq_out[i, :lengths[i]].mean(dim=0))
        pooled_seq = torch.stack(pooled_seq) # [batch, hidden]
        
        # 2. Embed static features
        static_emb = F.relu(self.static_embedding(static)) # [batch, hidden]
        
        # 3. Combine
        combined = torch.cat([pooled_seq, static_emb], dim=-1)
        hidden = F.relu(self.fc_combined(combined))
        
        # 4. Heads
        out_c = self.head_control(hidden)
        out_t = self.head_treated(hidden)
        
        # Process outputs: sigma must be positive
        def process_head_out(out):
            logits = out[:, 0:1]
            mu = out[:, 1:2]
            sigma = F.softplus(out[:, 2:3]) + 1e-4
            return logits, mu, sigma
            
        return process_head_out(out_c), process_head_out(out_t)

def train_cdtt(model, loader, causal_df, epochs=20, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = ZILNLoss()
    
    treatment_tensor = torch.FloatTensor(causal_df['treatment'].values).unsqueeze(1)
    
    history = {'total': [], 'cls': [], 'reg': []}
    
    model.train()
    for epoch in range(epochs):
        epoch_losses = {'total': 0, 'cls': 0, 'reg': 0}
        
        # We iterate over the loader. We need to make sure treatment matches the indices.
        # OnlineRetailDataset returns (seq, static, label). 
        # But our causal labels are in causal_df.
        # The loader preserves order if shuffle=False.
        
        for i, (seq, static, label, lengths) in enumerate(loader):
            # For this training, we need the treatment for each sample in the batch
            # If we used a custom dataset that includes treatment, it would be easier.
            # Let's assume the loader matches the causal_df order (shuffle=False for now or handle indexing)
            batch_indices = range(i * loader.batch_size, min((i+1) * loader.batch_size, len(causal_df)))
            t_batch = treatment_tensor[batch_indices].to(seq.device)
            
            optimizer.zero_grad()
            (logits_c, mu_c, sigma_c), (logits_t, mu_t, sigma_t) = model(seq, static, lengths)
            
            # Deep T-Learner logic:
            # If T=1, use treated head and treated label
            # If T=0, use control head and control label
            
            # Masking
            mask_t = (t_batch == 1).float()
            mask_c = (t_batch == 0).float()
            
            loss_t, cls_t, reg_t = criterion(logits_t, mu_t, sigma_t, label)
            loss_c, cls_c, reg_c = criterion(logits_c, mu_c, sigma_c, label)
            
            # Combine losses based on treatment
            total_loss = (mask_t * loss_t + mask_c * loss_c).mean()
            
            total_loss.backward()
            optimizer.step()
            
            epoch_losses['total'] += total_loss.item()
            epoch_losses['cls'] += (mask_t * cls_t + mask_c * cls_c).mean().item()
            epoch_losses['reg'] += (mask_t * reg_t + mask_c * reg_c).mean().item()
            
        n_batches = len(loader)
        history['total'].append(epoch_losses['total'] / n_batches)
        history['cls'].append(epoch_losses['cls'] / n_batches)
        history['reg'].append(epoch_losses['reg'] / n_batches)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {history['total'][-1]:.4f} (Cls: {history['cls'][-1]:.4f}, Reg: {history['reg'][-1]:.4f})")
        
    return history

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
    
    dataset = OnlineRetailDataset(processor.processed_samples)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    # 2. Initialize Model
    model = TemporalTransformer()
    
    # 3. Train
    print("Starting CDTT Training...")
    history = train_cdtt(model, loader, causal_df, epochs=15)
    
    # 4. Convergence Report
    plt.figure(figsize=(10, 6))
    plt.plot(history['cls'], label='Classification Loss (Zero-Inflation)')
    plt.plot(history['reg'], label='Regression Loss (Lognormal)')
    plt.title('CDTT Convergence Report: ZILN Head Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('cdtt_convergence_report.png')
    print("Convergence report saved to 'cdtt_convergence_report.png'.")
    
    # Save model
    torch.save(model.state_dict(), 'cdtt_model.pth')

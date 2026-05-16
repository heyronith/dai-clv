import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from src.models.cdtt_dr import TemporalTransformer, ZILNLoss
from src.data.online_retail_dataset import CausalDataset, causal_collate_fn


class DRTrainer:
    """
    Hybrid DR-Learner Training (Step 3).
    
    Uses CausalDataset so DR targets and treatment travel with each sample
    through shuffle, eliminating the batch misalignment bug (Audit Fix C1).
    """
    def __init__(self, lmbda: float = 0.8, lr: float = 0.001, epochs: int = 15):
        self.lmbda = lmbda
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_mean = 0.0
        self.target_std = 1.0

    def train_dr_model(self, model: TemporalTransformer, causal_dataset: CausalDataset,
                       train_idx: list, val_idx: list):
        """
        Train with hybrid loss: lambda * DR-MSE + (1-lambda) * Nuisance.
        
        causal_dataset must already have dr_target populated.
        """
        train_loader = DataLoader(Subset(causal_dataset, train_idx),
                                  batch_size=64, shuffle=True, collate_fn=causal_collate_fn)
        val_loader = DataLoader(Subset(causal_dataset, val_idx),
                                batch_size=64, shuffle=False, collate_fn=causal_collate_fn)

        # Compute target normalization stats from training split only
        train_dr_targets = causal_dataset.dr_target[train_idx]
        self.target_mean = float(np.mean(train_dr_targets))
        self.target_std = float(np.std(train_dr_targets)) + 1e-6

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        mse_criterion = nn.MSELoss()
        ziln_criterion = ZILNLoss()
        bce_criterion = nn.BCELoss()

        history = {'train_loss': [], 'val_dr_mse': [], 'val_pred_ate': []}

        print(f"Starting DR Training with Lambda={self.lmbda} (Target Std: {self.target_std:.2f})...")

        for epoch in range(self.epochs):
            model.train()
            epoch_total_loss = 0

            # causal_collate_fn returns: (seq, static, label, lengths, treatment, dr_target)
            for seq, static, label, lengths, t_batch, dr_target_batch in train_loader:
                seq = seq.to(self.device)
                static = static.to(self.device)
                label = label.to(self.device)
                lengths = lengths.to(self.device)
                t_batch = t_batch.to(self.device)
                dr_target_batch = dr_target_batch.to(self.device)

                # Normalize DR targets
                dr_target_norm = (dr_target_batch - self.target_mean) / self.target_std

                optimizer.zero_grad()
                (logits_c, mu_c, sigma_c), (logits_t, mu_t, sigma_t), p_score, tau_hat_norm = model(seq, static, lengths)

                # 1. DR MSE Loss (on normalized targets)
                loss_dr = mse_criterion(tau_hat_norm, dr_target_norm)

                # 2. Nuisance Loss (regularizer)
                mask_t = (t_batch == 1).float()
                mask_c = (t_batch == 0).float()
                loss_p = bce_criterion(p_score, t_batch)
                loss_ziln = (mask_t * ziln_criterion(logits_t, mu_t, sigma_t, label) +
                             mask_c * ziln_criterion(logits_c, mu_c, sigma_c, label)).mean()

                loss_nuisance = loss_p + loss_ziln

                # 3. Hybrid loss
                total_loss = self.lmbda * loss_dr + (1 - self.lmbda) * (loss_nuisance / 10.0)

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_total_loss += total_loss.item()

            # === Validation ===
            model.eval()
            val_dr_mses = []
            val_tau_preds = []

            with torch.no_grad():
                for seq, static, label, lengths, t_batch, dr_target_batch in val_loader:
                    seq = seq.to(self.device)
                    static = static.to(self.device)
                    lengths = lengths.to(self.device)
                    dr_target_batch = dr_target_batch.to(self.device)

                    _, _, _, tau_hat_norm = model(seq, static, lengths)
                    # Un-scale for evaluation
                    tau_hat = tau_hat_norm * self.target_std + self.target_mean

                    val_dr_mses.append(mse_criterion(tau_hat, dr_target_batch).item())
                    val_tau_preds.append(tau_hat.cpu().numpy())

            avg_val_mse = np.mean(val_dr_mses)
            avg_val_tau = np.concatenate(val_tau_preds).mean()

            history['train_loss'].append(epoch_total_loss / len(train_loader))
            history['val_dr_mse'].append(avg_val_mse)
            history['val_pred_ate'].append(avg_val_tau)

            print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {history['train_loss'][-1]:.4f} | "
                  f"Val DR-MSE: {avg_val_mse:.4f} | Pred ATE: {avg_val_tau:.4f}")

        return history

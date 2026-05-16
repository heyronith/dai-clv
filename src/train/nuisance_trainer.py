import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from src.models.cdtt_dr import TemporalTransformer, ZILNLoss
from src.data.online_retail_dataset import CausalDataset, causal_collate_fn
from sklearn.metrics import brier_score_loss


class NuisanceTrainer:
    """
    Cross-Fitting for Nuisance Estimation (Step 2).
    
    Uses CausalDataset so treatment labels are bundled per-sample.
    Shuffling the DataLoader cannot misalign labels (Audit Fix C1).
    """
    def __init__(self, model_params: dict = None, lr: float = 0.001, epochs: int = 10):
        self.model_params = model_params or {}
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_expectation(self, logits, mu, sigma):
        """E[Y] = p * exp(mu + sigma^2/2) for ZILN distribution.
        
        Clamps the exponent to prevent exp() overflow that creates
        astronomical nuisance predictions and DR target explosion.
        """
        p = torch.sigmoid(logits)
        # Clamp exponent: exp(15) ≈ 3.3M which is already extreme for CLV
        exponent = torch.clamp(mu + 0.5 * torch.pow(sigma, 2), max=15.0)
        return p * torch.exp(exponent)

    def train_cross_fit(self, causal_dataset: CausalDataset, causal_df: pd.DataFrame, folds: list):
        """
        Main cross-fitting loop.
        
        Returns a DataFrame with columns: propensity_pred, mu1_pred, mu0_pred
        aligned to causal_df.index.
        """
        oof_results = pd.DataFrame(index=causal_df.index)
        oof_results['propensity_pred'] = 0.0
        oof_results['mu1_pred'] = 0.0
        oof_results['mu0_pred'] = 0.0

        # Identify val/test indices (not in any OOF fold)
        all_oof_idx = set()
        for _, v_idx in folds:
            all_oof_idx.update(v_idx)
        val_test_idx = sorted(set(causal_df.index) - all_oof_idx)

        # Storage for ensemble predictions on val/test set
        vt_preds_p = np.zeros((len(val_test_idx), len(folds)))
        vt_preds_mu1 = np.zeros((len(val_test_idx), len(folds)))
        vt_preds_mu0 = np.zeros((len(val_test_idx), len(folds)))

        for f, (train_idx, val_idx) in enumerate(folds):
            print(f"\n--- Training Fold {f+1}/{len(folds)} ---")

            # causal_collate_fn returns: (seq, static, label, lengths, treatment, dr_target)
            train_loader = DataLoader(Subset(causal_dataset, train_idx),
                                      batch_size=64, shuffle=True, collate_fn=causal_collate_fn)
            val_loader = DataLoader(Subset(causal_dataset, val_idx),
                                    batch_size=64, shuffle=False, collate_fn=causal_collate_fn)

            # Fresh model per fold
            model = TemporalTransformer(**self.model_params).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
            ziln_criterion = ZILNLoss()
            bce_criterion = nn.BCELoss()

            # === Training ===
            model.train()
            for epoch in range(self.epochs):
                epoch_loss = 0
                for seq, static, label, lengths, t_batch, _ in train_loader:
                    seq = seq.to(self.device)
                    static = static.to(self.device)
                    label = label.to(self.device)
                    lengths = lengths.to(self.device)
                    t_batch = t_batch.to(self.device)

                    optimizer.zero_grad()
                    (logits_c, mu_c, sigma_c), (logits_t, mu_t, sigma_t), p_score, _ = model(seq, static, lengths)

                    # Propensity loss
                    loss_p = bce_criterion(p_score, t_batch)

                    # Masked outcome loss
                    mask_t = (t_batch == 1).float()
                    mask_c = (t_batch == 0).float()
                    loss_t = ziln_criterion(logits_t, mu_t, sigma_t, label)
                    loss_c = ziln_criterion(logits_c, mu_c, sigma_c, label)
                    loss_y = (mask_t * loss_t + mask_c * loss_c).mean()

                    total_loss = loss_p + loss_y
                    total_loss.backward()
                    optimizer.step()
                    epoch_loss += total_loss.item()

                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} | Avg Loss: {epoch_loss/len(train_loader):.4f}")

            # === Out-of-Fold Prediction ===
            model.eval()
            val_preds_p, val_preds_mu1, val_preds_mu0 = [], [], []
            val_t_all = []

            with torch.no_grad():
                for seq, static, label, lengths, t_batch, _ in val_loader:
                    seq = seq.to(self.device)
                    static = static.to(self.device)
                    lengths = lengths.to(self.device)

                    (logits_c, mu_c, sigma_c), (logits_t, mu_t, sigma_t), p_score, _ = model(seq, static, lengths)

                    e_hat = torch.clamp(p_score, 0.05, 0.95)
                    y_hat_t = self._get_expectation(logits_t, mu_t, sigma_t)
                    y_hat_c = self._get_expectation(logits_c, mu_c, sigma_c)

                    val_preds_p.append(e_hat.cpu().numpy())
                    val_preds_mu1.append(y_hat_t.cpu().numpy())
                    val_preds_mu0.append(y_hat_c.cpu().numpy())
                    val_t_all.append(t_batch.cpu().numpy())

            val_preds_p = np.concatenate(val_preds_p).flatten()
            val_preds_mu1 = np.concatenate(val_preds_mu1).flatten()
            val_preds_mu0 = np.concatenate(val_preds_mu0).flatten()

            oof_results.loc[val_idx, 'propensity_pred'] = val_preds_p
            oof_results.loc[val_idx, 'mu1_pred'] = val_preds_mu1
            oof_results.loc[val_idx, 'mu0_pred'] = val_preds_mu0

            # === Ensemble prediction for Val/Test set ===
            vt_loader = DataLoader(Subset(causal_dataset, val_test_idx),
                                   batch_size=64, shuffle=False, collate_fn=causal_collate_fn)
            curr_vt_p, curr_vt_mu1, curr_vt_mu0 = [], [], []
            with torch.no_grad():
                for seq, static, label, lengths, t_batch, _ in vt_loader:
                    seq = seq.to(self.device)
                    static = static.to(self.device)
                    lengths = lengths.to(self.device)

                    (logits_c, mu_c, sigma_c), (logits_t, mu_t, sigma_t), p_score, _ = model(seq, static, lengths)

                    e_hat = torch.clamp(p_score, 0.05, 0.95)
                    y_hat_t = self._get_expectation(logits_t, mu_t, sigma_t)
                    y_hat_c = self._get_expectation(logits_c, mu_c, sigma_c)

                    curr_vt_p.append(e_hat.cpu().numpy())
                    curr_vt_mu1.append(y_hat_t.cpu().numpy())
                    curr_vt_mu0.append(y_hat_c.cpu().numpy())

            vt_preds_p[:, f] = np.concatenate(curr_vt_p).flatten()
            vt_preds_mu1[:, f] = np.concatenate(curr_vt_mu1).flatten()
            vt_preds_mu0[:, f] = np.concatenate(curr_vt_mu0).flatten()

            # Validation metrics
            val_t_all = np.concatenate(val_t_all).flatten()
            brier = brier_score_loss(val_t_all, val_preds_p)
            print(f"Fold {f+1} Validation | Propensity Brier Score: {brier:.4f}")

        # Average ensemble predictions for val/test set
        oof_results.loc[val_test_idx, 'propensity_pred'] = vt_preds_p.mean(axis=1)
        oof_results.loc[val_test_idx, 'mu1_pred'] = vt_preds_mu1.mean(axis=1)
        oof_results.loc[val_test_idx, 'mu0_pred'] = vt_preds_mu0.mean(axis=1)

        return oof_results

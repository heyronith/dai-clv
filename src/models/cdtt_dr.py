import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ZILNLoss(nn.Module):
    """
    Zero-Inflated Lognormal Loss.
    NLL = log(y) + log(sigma) + 0.5*log(2pi) + (log(y)-mu)^2 / (2*sigma^2)
    """
    def __init__(self, eps=1e-6):
        super(ZILNLoss, self).__init__()
        self.eps = eps

    def forward(self, logits, mu, sigma, target):
        positive_mask = (target > 0).float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, positive_mask, reduction='none')

        y_pos = target.clamp(min=self.eps)
        log_y = torch.log(y_pos)
        term1 = torch.log(sigma.clamp(min=self.eps))
        term2 = 0.5 * np.log(2 * np.pi)
        term3 = ((log_y - mu) ** 2) / (2 * (sigma ** 2).clamp(min=self.eps))
        regression_loss = log_y + term1 + term2 + term3

        loss = bce_loss + positive_mask * regression_loss
        return loss  # Per-sample loss for weighting


class TemporalTransformer(nn.Module):
    def __init__(self, seq_dim=8, static_dim=3, hidden_dim=64, nhead=4, num_layers=2):
        super(TemporalTransformer, self).__init__()
        self.seq_embedding = nn.Linear(seq_dim, hidden_dim)
        self.static_embedding = nn.Linear(static_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_combined = nn.Linear(hidden_dim * 2, hidden_dim)

        # Outcome and propensity heads
        self.head_control = nn.Linear(hidden_dim, 3)
        self.head_treated = nn.Linear(hidden_dim, 3)
        self.head_propensity = nn.Linear(hidden_dim, 1)

        # Uplift Head (3-layer MLP)
        self.head_uplift = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """
        Kaiming initialization for custom heads only (Audit Fix C2).
        Does NOT touch the Transformer encoder, which uses its own defaults.
        """
        for head in [self.head_control, self.head_treated, self.head_propensity]:
            nn.init.kaiming_normal_(head.weight)
            nn.init.constant_(head.bias, 0)
        for m in self.head_uplift.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

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
        out_u = self.head_uplift(hidden)

        def process_head_out(out):
            logits = out[:, 0:1]
            mu = out[:, 1:2]
            sigma = F.softplus(out[:, 2:3]) + 1e-4
            return logits, mu, sigma

        return process_head_out(out_c), process_head_out(out_t), torch.sigmoid(out_p), out_u

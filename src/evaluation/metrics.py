import numpy as np
import pandas as pd
import json
from sklearn.utils import resample
from typing import Dict, Tuple


class CausalMetrics:
    """
    Research-Grade Causal Evaluation.
    Includes PEHE, IPW-AUUC, Policy Value, and Uplift Curves.
    """

    @staticmethod
    def calculate_pehe(true_tau: np.ndarray, pred_tau: np.ndarray) -> float:
        """Precision in Estimation of Heterogeneous Effects (Root MSE of tau)."""
        return np.sqrt(np.mean((true_tau - pred_tau) ** 2))

    @staticmethod
    def compute_uplift_curve(y_obs: np.ndarray, treatment: np.ndarray,
                             e_hat: np.ndarray, pred_tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the IPW-adjusted cumulative uplift curve (Audit Fix S4).
        
        Returns (percentiles, uplift_values) for plotting.
        """
        df = pd.DataFrame({'y': y_obs, 't': treatment, 'e': e_hat, 'pred': pred_tau})
        df = df.sort_values('pred', ascending=False).reset_index(drop=True)
        n = len(df)

        cum_treated = (df['y'] * df['t'] / df['e']).cumsum()
        cum_control = (df['y'] * (1 - df['t']) / (1 - df['e'])).cumsum()
        cum_n = np.arange(1, n + 1)

        uplift = (cum_treated.values - cum_control.values) / cum_n
        percentiles = np.linspace(0, 100, n)
        return percentiles, uplift

    @staticmethod
    def calculate_ipw_auuc(y_obs: np.ndarray, treatment: np.ndarray,
                           e_hat: np.ndarray, pred_tau: np.ndarray) -> float:
        """Area Under the IPW-adjusted Uplift Curve."""
        _, uplift = CausalMetrics.compute_uplift_curve(y_obs, treatment, e_hat, pred_tau)
        return float(np.mean(uplift))

    @staticmethod
    def calculate_policy_value(y_obs: np.ndarray, treatment: np.ndarray,
                               e_hat: np.ndarray, pred_tau: np.ndarray, k: float = 0.2) -> float:
        """
        DR-style Policy Value for targeting top K% (Audit Fix S3).
        
        V(π) = (1/N) Σ [ π_i * T_i * Y_i / e_i  -  π_i * (1-T_i) * Y_i / (1-e_i) ]
             + (1/N) Σ [ (1-π_i) * (1-T_i) * Y_i / (1-e_i) ]
        
        Simplified: for targeted group, estimate E[Y(1)]; for non-targeted, estimate E[Y(0)].
        """
        threshold = np.percentile(pred_tau, 100 * (1 - k))
        targeted = (pred_tau >= threshold).astype(float)

        # IPW estimate of E[Y(1)] for targeted group
        val_target = np.sum(targeted * treatment * y_obs / e_hat) / max(np.sum(targeted), 1)
        # IPW estimate of E[Y(0)] for non-targeted group
        val_no_target = np.sum((1 - targeted) * (1 - treatment) * y_obs / (1 - e_hat)) / max(np.sum(1 - targeted), 1)

        # Total policy value = targeted * E[Y(1)] + non-targeted * E[Y(0)]
        frac_targeted = np.mean(targeted)
        return frac_targeted * val_target + (1 - frac_targeted) * val_no_target

    def bootstrap_metrics(self, test_df: pd.DataFrame, pred_dict: Dict[str, np.ndarray],
                          n_iterations: int = 100) -> Dict:
        """
        Bootstrap 95% CIs for PEHE, AUUC, and Profit at multiple K (Audit Fix M1).
        """
        k_values = [0.1, 0.2, 0.5]
        metric_keys = ['pehe', 'auuc'] + [f'profit_{int(k*100)}pct' for k in k_values]
        results = {name: {mk: [] for mk in metric_keys} for name in pred_dict.keys()}

        # Also store uplift curves for plotting (Audit Fix S4)
        curve_storage = {name: [] for name in pred_dict.keys()}

        for iteration in range(n_iterations):
            boot_idx = resample(np.arange(len(test_df)), random_state=iteration)

            y_obs = test_df['y_obs'].values[boot_idx]
            treatment = test_df['treatment'].values[boot_idx]
            true_tau = test_df['tau'].values[boot_idx]
            e_hat = test_df['propensity_pred'].values[boot_idx]

            for name, all_preds in pred_dict.items():
                preds = all_preds[boot_idx]

                results[name]['pehe'].append(self.calculate_pehe(true_tau, preds))
                results[name]['auuc'].append(self.calculate_ipw_auuc(y_obs, treatment, e_hat, preds))

                for k in k_values:
                    key = f'profit_{int(k*100)}pct'
                    results[name][key].append(self.calculate_policy_value(y_obs, treatment, e_hat, preds, k=k))

                # Store uplift curve for this bootstrap iteration
                _, uplift = self.compute_uplift_curve(y_obs, treatment, e_hat, preds)
                curve_storage[name].append(uplift)

        # Format results with CIs
        final_report = {}
        for name, metrics in results.items():
            final_report[name] = {}
            for m_name, vals in metrics.items():
                final_report[name][m_name] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'ci_low': float(np.percentile(vals, 2.5)),
                    'ci_high': float(np.percentile(vals, 97.5))
                }

        return final_report, curve_storage

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict


class VisualizationSuite:
    """
    Manuscript-quality visualizations (300 DPI).
    """
    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use('seaborn-v0_8-whitegrid')

    def plot_uplift_curves(self, curve_storage: Dict[str, list],
                           output_path: str = 'v2_uplift_curves.png'):
        """
        Figure 1: Real IPW-adjusted cumulative uplift curves with bootstrapped CI bands.
        curve_storage: {model_name: [list of uplift arrays from bootstrap]}
        """
        plt.figure(figsize=(10, 6))

        for name, curves in curve_storage.items():
            # Interpolate all curves to a common x-axis (100 points)
            n_points = 100
            interpolated = []
            for curve in curves:
                x_orig = np.linspace(0, 100, len(curve))
                x_new = np.linspace(0, 100, n_points)
                interpolated.append(np.interp(x_new, x_orig, curve))

            interpolated = np.array(interpolated)
            mean_curve = np.mean(interpolated, axis=0)
            low_ci = np.percentile(interpolated, 2.5, axis=0)
            high_ci = np.percentile(interpolated, 97.5, axis=0)

            x = np.linspace(0, 100, n_points)
            plt.plot(x, mean_curve, label=name, linewidth=2)
            plt.fill_between(x, low_ci, high_ci, alpha=0.15)

        # Random baseline (flat line at overall mean uplift)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Random')
        plt.title('Cumulative IPW-Adjusted Uplift Curve (95% CI)', fontsize=14)
        plt.xlabel('Percentile Targeted (%)', fontsize=12)
        plt.ylabel('Mean IPW Uplift per Customer', fontsize=12)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()

    def plot_calibration(self, true_tau: np.ndarray, pred_tau: np.ndarray,
                         name: str, output_path: str = 'v2_calibration.png'):
        """
        Figure 2: Causal Calibration Plot (Predicted vs. True).
        """
        plt.figure(figsize=(8, 8))

        rho, _ = spearmanr(pred_tau, true_tau)
        pehe = np.sqrt(np.mean((true_tau - pred_tau) ** 2))

        plt.hexbin(pred_tau, true_tau, gridsize=30, cmap='Blues', mincnt=1, alpha=0.8)

        lims = [
            min(np.min(pred_tau), np.min(true_tau)),
            max(np.max(pred_tau), np.max(true_tau)),
        ]
        plt.plot(lims, lims, 'r--', alpha=0.75, zorder=3, label='Perfect Calibration')

        plt.text(0.05, 0.95, f'Spearman ρ: {rho:.3f}\nPEHE: {pehe:.3f}',
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        plt.title(f'Causal Calibration: {name}', fontsize=14)
        plt.xlabel('Predicted Uplift (τ-hat)', fontsize=12)
        plt.ylabel('Ground Truth Uplift (τ-true)', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()

    def plot_policy_profit(self, final_report: Dict, output_path: str = 'v2_policy_profit.png'):
        """
        Figure 3: Policy Value at K% thresholds — all from real bootstrap CIs.
        """
        models = list(final_report.keys())
        k_labels = ['10%', '20%', '50%']
        k_keys = ['profit_10pct', 'profit_20pct', 'profit_50pct']

        x = np.arange(len(k_labels))
        width = 0.25
        colors = ['#2196F3', '#FF9800', '#4CAF50']

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, model_name in enumerate(models):
            means = [final_report[model_name][k]['mean'] for k in k_keys]
            ci_lows = [final_report[model_name][k]['ci_low'] for k in k_keys]
            ci_highs = [final_report[model_name][k]['ci_high'] for k in k_keys]
            errors = [[m - lo for m, lo in zip(means, ci_lows)],
                      [hi - m for m, hi in zip(means, ci_highs)]]

            ax.bar(x + i * width, means, width, label=model_name,
                   color=colors[i % len(colors)], alpha=0.85, yerr=errors, capsize=4)

        ax.set_xlabel('Top K% Targeted', fontsize=12)
        ax.set_ylabel('Policy Value (IPW Estimator)', fontsize=12)
        ax.set_title('Policy Value at Targeting Thresholds (95% CI)', fontsize=14)
        ax.set_xticks(x + width)
        ax.set_xticklabels(k_labels)
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()

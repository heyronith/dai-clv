import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.online_retail_dataset import OnlineRetailDataProcessor, PipelineConfig
from typing import List, Dict, Tuple

class CausalGenerator:
    def __init__(self, samples: List[Dict]):
        self.samples = samples
        self.df = self._prepare_dataframe()
        
    def _prepare_dataframe(self) -> pd.DataFrame:
        """Extracts key features and baseline outcomes into a DataFrame."""
        rows = []
        for s in self.samples:
            row = {
                'customer_id': s['customer_id'],
                'y0': s['label'], # Raw future revenue from dataset
                'velocity': s['sequence'][:, 3].mean(), # Mean velocity in obs window
                'tenure': s['static_features'][0],
                'avg_spend': s['static_features'][1],
                'has_returns': s['static_features'][2]
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def generate_causal_labels(self, 
                               alpha: float = 2.0, 
                               gamma: float = 50.0, 
                               kappa: float = 1.5,
                               moderate_tenure: float = 0.5,
                               tenure_sigma: float = 0.2) -> pd.DataFrame:
        """
        Implements Task 3.1 & 3.2: Semi-Synthetic Causal Generator.
        
        Args:
            alpha: Sensitivity to purchase velocity.
            gamma: Peak lift for moderate tenure customers.
            kappa: Selection bias strength (higher = more confounding).
            moderate_tenure: The tenure (in years) considered 'moderate'.
        """
        # 1. Heterogeneous Treatment Effect (tau_i)
        # Higher for high velocity and moderate tenure (Gaussian peak)
        tenure_effect = gamma * np.exp(-((self.df['tenure'] - moderate_tenure)**2) / (2 * tenure_sigma**2))
        velocity_effect = alpha * np.maximum(0, self.df['velocity'] * 365) # Velocity back to days scale
        
        noise = np.random.normal(0, 5, size=len(self.df))
        self.df['tau'] = np.maximum(0, velocity_effect + tenure_effect + noise)
        
        # 2. Treated Outcome (Y(1))
        self.df['y1'] = self.df['y0'] + self.df['tau']
        
        # 3. Treatment Assignment (T) - Confounded Assignment
        # Higher P(T=1) for high Avg_Spend (Selection Bias)
        logit_score = kappa * (self.df['avg_spend'] - self.df['avg_spend'].median())
        prob_t = 1 / (1 + np.exp(-logit_score))
        self.df['treatment'] = np.random.binomial(1, prob_t)
        
        # 4. Observed Outcome (Y_obs)
        self.df['y_obs'] = self.df['treatment'] * self.df['y1'] + (1 - self.df['treatment']) * self.df['y0']
        
        return self.df

    def generate_selection_bias_report(self):
        """Task 3.3: Measurable Selection Bias Analysis."""
        treated = self.df[self.df['treatment'] == 1]
        untreated = self.df[self.df['treatment'] == 0]
        
        avg_y0_treated = treated['y0'].mean()
        avg_y0_untreated = untreated['y0'].mean()
        selection_bias = avg_y0_treated - avg_y0_untreated
        
        print("\n=== Selection Bias Report ===")
        print(f"Total Treated: {len(treated)} ({len(treated)/len(self.df):.2%})")
        print(f"Total Untreated: {len(untreated)} ({len(untreated)/len(self.df):.2%})")
        print(f"Average Y(0) (Baseline) for Treated: {avg_y0_treated:.2f}")
        print(f"Average Y(0) (Baseline) for Untreated: {avg_y0_untreated:.2f}")
        print(f"Selection Bias (ΔY0): {selection_bias:.2f}")
        print(f"Selection Bias Ratio: {selection_bias / avg_y0_untreated:.2%}")
        print("============================\n")
        
        # Visualization
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.kdeplot(treated['y0'], label='Treated (T=1)', fill=True)
        sns.kdeplot(untreated['y0'], label='Untreated (T=0)', fill=True)
        plt.title('Baseline Outcome Y(0) Distribution by Treatment')
        plt.xlabel('Revenue')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        sns.scatterplot(data=self.df, x='tenure', y='tau', hue='treatment', alpha=0.5)
        plt.title('Heterogeneous Treatment Effect (τ) vs Tenure')
        
        plt.tight_layout()
        plt.savefig('selection_bias_report.png')
        print("Selection bias report saved to 'selection_bias_report.png'.")

if __name__ == "__main__":
    # 1. Load Real Dataset
    config = PipelineConfig(slide_step_months=3)
    processor = OnlineRetailDataProcessor('online_retail_II.xlsx', config)
    processor.load_and_clean()
    processor.create_sliding_windows()
    
    # 2. Generate Causal Labels
    generator = CausalGenerator(processor.processed_samples)
    causal_df = generator.generate_causal_labels(kappa=2.0) # Strong selection bias
    
    # 3. Report
    generator.generate_selection_bias_report()
    
    # Show some examples
    print("\nCausal Label Samples:")
    print(causal_df[['customer_id', 'y0', 'tau', 'T', 'y_obs']].head())

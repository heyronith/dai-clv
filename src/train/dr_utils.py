import pandas as pd
import numpy as np

def generate_dr_targets(causal_df: pd.DataFrame, nuisance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Implements Task 3.1: DR Pseudo-Outcome Generation (AIPW).
    
    Formula: 
    psi = (mu1 - mu0) + T*(Y - mu1)/e - (1-T)*(Y - mu0)/(1-e)
    """
    # Merge on index to ensure alignment
    df = causal_df.merge(nuisance_df, left_index=True, right_index=True)
    
    T = df['treatment'].values
    Y = df['y_obs'].values
    
    # Nuisance predictions (already clipped in Step 2)
    e = df['propensity_pred'].values
    mu1 = df['mu1_pred'].values
    mu0 = df['mu0_pred'].values
    
    # Doubly Robust Formula Components
    # 1. Prediction part (T-Learner style)
    tau_pred = mu1 - mu0
    
    # 2. Correction part (Weighting residual by inverse propensity)
    # Note: e is already clipped to [0.05, 0.95]
    correction_t = T * (Y - mu1) / e
    correction_c = (1 - T) * (Y - mu0) / (1 - e)
    
    # 3. Combine to get DR target (pseudo-outcome)
    dr_raw = tau_pred + correction_t - correction_c
    
    # Stability: Clamp at the 95th percentile of the DR targets themselves.
    # Using 95th instead of 99th to be more aggressive against IPW outliers.
    limit = np.percentile(np.abs(dr_raw), 95)
    dr_clipped = np.clip(dr_raw, -limit, limit)
    
    print(f"[DR Targets] Raw range: [{dr_raw.min():.1f}, {dr_raw.max():.1f}] | "
          f"Clipped at ±{limit:.1f} | Std: {np.std(dr_clipped):.1f}")
    
    df['dr_target'] = dr_clipped


    
    return df

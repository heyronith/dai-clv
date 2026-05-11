import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_pehe(true_tau, pred_tau):
    return np.sqrt(np.mean((true_tau - pred_tau)**2))

def get_uplift_curve(y_obs, treatment, pred_tau):
    """
    Returns the uplift curve: cumulative uplift across population.
    """
    df = pd.DataFrame({'y': y_obs, 't': treatment, 'pred': pred_tau})
    df = df.sort_values('pred', ascending=False).reset_index(drop=True)
    
    n = len(df)
    uplift = []
    
    # Cumulative stats
    n_t = 0
    n_c = 0
    y_t_sum = 0
    y_c_sum = 0
    
    for i in range(n):
        if df.loc[i, 't'] == 1:
            n_t += 1
            y_t_sum += df.loc[i, 'y']
        else:
            n_c += 1
            y_c_sum += df.loc[i, 'y']
        
        # Calculate uplift at this point
        if n_t > 0 and n_c > 0:
            u = (y_t_sum / n_t - y_c_sum / n_c) * (n_t + n_c)
        else:
            u = 0
        uplift.append(u)
        
    return np.array(uplift)

def calculate_auuc(uplift_curve):
    return np.mean(uplift_curve)

def get_qini_curve(y_obs, treatment, pred_tau):
    """
    Qini Curve = Cumulative Gain in Treated - (Cumulative Gain in Control * Ratio of Treated/Control)
    """
    df = pd.DataFrame({'y': y_obs, 't': treatment, 'pred': pred_tau})
    df = df.sort_values('pred', ascending=False).reset_index(drop=True)
    
    n_t_total = df['t'].sum()
    n_c_total = len(df) - n_t_total
    
    qini = []
    y_t_sum = 0
    y_c_sum = 0
    n_t = 0
    
    for i in range(len(df)):
        if df.loc[i, 't'] == 1:
            y_t_sum += df.loc[i, 'y']
            n_t += 1
        else:
            y_c_sum += df.loc[i, 'y']
            
        q = y_t_sum - (y_c_sum * n_t / (i + 1 - n_t + 1e-6))
        qini.append(q)
        
    return np.array(qini)

def simulate_profit(true_tau, pred_tau, cost_per_treatment=5.0, top_k=0.1):
    """
    Simulates profit if we treat the top k percent of customers.
    Profit = sum(true_tau - cost) for those treated.
    """
    n = len(true_tau)
    k = int(n * top_k)
    
    # Sort indices by predicted tau
    idx = np.argsort(pred_tau)[::-1][:k]
    
    # Calculate profit
    total_tau = true_tau[idx].sum()
    total_cost = k * cost_per_treatment
    return total_tau - total_cost

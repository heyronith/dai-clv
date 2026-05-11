# Results and Discussion: Decision-Aware Incremental CLV

## 1. The Predictive-Causal Paradox
Our experiments reveal a fundamental disconnect between predictive accuracy and causal validity in high-confounding environments. In our baseline evaluation, **XGBoost** achieved the lowest Predictive RMSE (972.28), yet exhibited a **negative Spearman correlation (-0.09)** with the true treatment effect ($\tau$). 

This phenomenon is explained by the **187% selection bias** inherent in our semi-synthetic generator, where treatment assignment was positively correlated with historical spend. The non-causal predictor correctly identified high-spend individuals (predictive success) but fundamentally failed to distinguish between their baseline spend and the incremental lift provided by the intervention (causal failure). This confirms that optimizing for absolute revenue without causal correction can lead to suboptimal targeting policies that prioritize customers who would have purchased regardless.

## 2. Temporal Dynamics and the CDTT Contribution
The introduction of the **Temporal Transformer** architecture provided a significant boost in ranking quality. In our ablation study, the Transformer-based CDTT improved the **Area Under the Uplift Curve (AUUC)** by **6.29%** compared to a static Multi-Layer Perceptron (MLP) using identical features.

This validates our hypothesis that **purchase "rhythms"**—captured through sequential velocity and inter-purchase delta-T—are critical indicators of treatment responsiveness. While static features provide a reliable baseline, the attention mechanism allows the model to identify "momentum" in the customer journey, distinguishing between stable long-term loyalty and transient, treatment-sensitive spikes in activity.

## 3. The ZILN-MSE Paradox: Persuadables vs. Sure Things
A critical finding of this research is the behavioral trade-off between the **Zero-Inflated Lognormal (ZILN)** head and the standard **Mean Squared Error (MSE)** head.

Our segmental analysis showed that:
- **CDTT (ZILN)** successfully targeted **75% more Persuadables** (incremental responders) than the MSE model.
- **MSE**, however, appeared more "profitable" in absolute terms by gravitating toward **Sure Things** (high-baseline spenders).

This represents a classic **"Causal Paradox"**: the ZILN head is a technically superior causal instrument because it models the heavy-tailed distribution of spend volume, allowing it to identify the "hidden" persuadables who are not yet high-volume spenders. Conversely, the MSE head acts as a conservative "safe bet" by defaulting to high-volume customers (Whales) whose absolute revenue compensates for the model's inability to find true incremental lift. For business applications, this suggests that the choice of loss head must be aligned with the specific objective: **Ranking for Efficiency (ZILN)** vs. **Targeting for Absolute Volume (MSE)**.

## 4. Scientific Limitations: Scale Bias and Overfitting
In the spirit of scientific skepticism, we identify two primary limitations in the current CDTT implementation:

1.  **Scale Bias**: The high **Precision in Estimation of Heterogeneous Effects (PEHE)** observed in our deep models indicates a calibration gap. While the models excel at *ranking* responders (AUUC), their absolute estimates of $\tau$ are often biased by the large scale of predicted revenue. Future work should investigate post-hoc calibration techniques for ZILN-based uplift.
2.  **Overfitting Risks**: Our "Long-Run" ablation experiment showed a catastrophic **56% drop in AUUC** when training beyond 15 epochs. The Transformer's complexity, while powerful for capturing temporal dynamics, makes it highly sensitive to the sparse and noisy nature of transactional data, necessitating aggressive regularization and early stopping.

## 5. Policy Implications for Decision-Aware ML
Our findings have direct implications for the design of "Decision-Aware" machine learning systems. True incremental CLV requires moving beyond the maximization of absolute observed revenue toward the estimation of **conditional average treatment effects (CATE)** through distributional modeling.

The success of the **Doubly Robust (DR)** correction in improving AUUC by **18.5%** over a standard T-Learner demonstrates that propensity-weighting is a non-negotiable requirement for longitudinal CLV modeling. We conclude that future AI systems in CRM and marketing must be built on **causal-distributional foundations** to ensure they drive true incremental growth rather than merely harvesting baseline demand.

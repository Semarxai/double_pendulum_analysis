# double_pendulum_analysis
Bipredictability (P) analysis for double pendulum trajectories. Companion code for Hafez et al. (2026).


Analysis code for computing bi-predictability (P) from double pendulum trajectories, as described in:

> Hafez et al. (2026), "A Mathematical Theory of Agency and Intelligence"

## Method

**State definition**: S = (θ₁, θ₂, ω₁, ω₂)

**Pipeline**:
1. Unwrap angles (handle 2π discontinuities)
2. Z-score normalize per variable
3. Discretize into 16 equal-width bins
4. Window: 300 steps, stride 75
5. Compute per window: H(S), H(S'), MI(S;S'), P, ΔH

**Metrics**:
- P = MI(S;S') / [H(S) + H(S')]
- ΔH = H(S'|S) − H(S|S')

## Files

- `main.py` — Analysis script
- `DP_Sample.csv` — Sample trajectory (20k steps)
- `dp_metrics_output.csv` — Output metrics

## Usage

```bash
pip install pandas numpy
python main.py
```

## Trajectory Data

Trajectories generated using MATLAB ode45 with parameters described in Methods 4.4.1. Simulation code by R. Pena.

## Related Repositories

- RL analysis: [github.com/semarxai/idt_app](https://github.com/semarxai/idt_app)
- LLM analysis: [github.com/semarxai/llm-monitoring](https://github.com/semarxai/llm-monitoring)


import numpy as np
import pandas as pd

# Step 1: Load data
df = pd.read_csv("DP_Sample.csv")
print(f"Loaded: {len(df):,} rows")
print(f"Columns: {list(df.columns)}")
print(df.head())

# Step 2: Unwrap angles
df["theta1"] = np.unwrap(df["theta1"].values)
df["theta2"] = np.unwrap(df["theta2"].values)

print("\nAngles unwrapped.")
print(df[["theta1", "theta2"]].head(10))

# Step 3: Z-score normalize
STATE = ["theta1", "theta2", "omega1", "omega2"]

for col in STATE:
    mean = df[col].mean()
    std = df[col].std()
    df[f"z_{col}"] = (df[col] - mean) / (std + 1e-9)

print("\nZ-score normalized.")
print(df[["z_theta1", "z_theta2", "z_omega1", "z_omega2"]].head(10))


# Step 4: Bin into 16 equal-width bins
N_BINS = 16

for col in STATE:
    z_col = f"z_{col}"
    df[f"bin_{col}"] = pd.cut(df[z_col], bins=N_BINS, labels=False)

print("\nBinned into 16 bins.")
print(df[["bin_theta1", "bin_theta2", "bin_omega1", "bin_omega2"]].head(10))



# Step 5: Create state labels and windowing
df["state"] = (df["bin_theta1"].astype(str) + "_" +
               df["bin_theta2"].astype(str) + "_" +
               df["bin_omega1"].astype(str) + "_" +
               df["bin_omega2"].astype(str))

df["next_state"] = df["state"].shift(-1)

# Window parameters
W = 300  # window size
STRIDE = 75  # stride

windows = []
for start in range(0, len(df) - W, STRIDE):
    windows.append({
        "window_id": len(windows) + 1,
        "start": start,
        "end": start + W
    })

print(f"\nCreated {len(windows)} windows (W={W}, stride={STRIDE})")

# Step 6: Compute P, entropy metrics, and Delta H per window
from collections import Counter

results = []

for w in windows:
    subset = df.iloc[w["start"]:w["end"]].dropna(subset=["state", "next_state"])

    states = subset["state"].tolist()
    next_states = subset["next_state"].tolist()
    pairs = list(zip(states, next_states))

    n = len(pairs)
    if n < 50:
        continue

    # Count frequencies
    p_s = Counter(states)
    p_sp = Counter(next_states)
    p_joint = Counter(pairs)

    # Compute entropies
    H_S = -sum((c / n) * np.log2(c / n) for c in p_s.values())
    H_Sp = -sum((c / n) * np.log2(c / n) for c in p_sp.values())
    H_joint = -sum((c / n) * np.log2(c / n) for c in p_joint.values())

    # Mutual information and P
    MI = H_S + H_Sp - H_joint
    P = MI / (H_S + H_Sp) if (H_S + H_Sp) > 0 else 0

    # Conditional entropies and Delta H
    H_f = H_joint - H_S  # H(S'|S) - forward uncertainty
    H_b = H_joint - H_Sp  # H(S|S') - backward uncertainty
    Delta_H = H_f - H_b  # Predictive asymmetry

    results.append({
        "window_id": w["window_id"],
        "H_S": H_S,
        "H_Sp": H_Sp,
        "MI": MI,
        "P": P,
        "H_f": H_f,
        "H_b": H_b,
        "Delta_H": Delta_H
    })

df_results = pd.DataFrame(results)
print(f"\nComputed metrics for {len(df_results)} windows")
print(df_results.head(10))
print(f"\nMean P: {df_results['P'].mean():.4f}")
print(f"Mean Delta_H: {df_results['Delta_H'].mean():.4f}")


# Step 7: Save results
df_results.to_csv("dp_metrics_output.csv", index=False)

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Input rows: {len(df):,}")
print(f"Windows processed: {len(df_results)}")
print(f"Window size: {W} steps, Stride: {STRIDE}")
print(f"Bins: {N_BINS}")
print(f"\nResults:")
print(f"  Mean P:       {df_results['P'].mean():.4f}")
print(f"  Mean Delta_H: {df_results['Delta_H'].mean():.4f}")
print(f"  Mean H(S):    {df_results['H_S'].mean():.4f}")
print(f"  Mean H(S'):   {df_results['H_Sp'].mean():.4f}")
print(f"\nSaved to: dp_metrics_output.csv")

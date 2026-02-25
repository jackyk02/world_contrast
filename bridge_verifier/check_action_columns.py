"""
Script to inspect DROID parquet files and check what the 'action' column corresponds to.
"""
import pandas as pd
import numpy as np

# Load a sample episode
parquet_path = "/root/droid_1.0.1/data/chunk-000/episode_000000.parquet"
df = pd.read_parquet(parquet_path)

print("=" * 80)
print("All columns in the parquet file:")
print("=" * 80)
for col in sorted(df.columns):
    print(f"  - {col}")

print("\n" + "=" * 80)
print("Column dtypes:")
print("=" * 80)
print(df.dtypes)

print("\n" + "=" * 80)
print("First 3 rows of action-related columns:")
print("=" * 80)

# Find all action-related columns
action_cols = [c for c in df.columns if 'action' in c.lower()]
print(f"\nAction-related columns found: {action_cols}")

for col in action_cols:
    print(f"\n--- {col} ---")
    print(f"Shape of first element: {np.array(df[col].iloc[0]).shape}")
    print(f"First 3 values:")
    for i in range(min(3, len(df))):
        val = np.array(df[col].iloc[i])
        print(f"  [{i}]: {val}")

# Compare 'action' with other action columns if they exist
if 'action' in df.columns:
    print("\n" + "=" * 80)
    print("Comparing 'action' with other action columns:")
    print("=" * 80)
    
    action_main = np.array(df['action'].iloc[0])
    
    for col in action_cols:
        if col != 'action':
            other = np.array(df[col].iloc[0])
            if action_main.shape == other.shape:
                is_equal = np.allclose(action_main, other, rtol=1e-5, atol=1e-8)
                max_diff = np.max(np.abs(action_main - other))
                print(f"\n'action' vs '{col}':")
                print(f"  Same shape: {action_main.shape} == {other.shape}")
                print(f"  Values equal (allclose): {is_equal}")
                print(f"  Max absolute difference: {max_diff}")
                print(f"  action[0]:     {action_main}")
                print(f"  {col}[0]: {other}")
            else:
                print(f"\n'action' vs '{col}':")
                print(f"  Different shapes: {action_main.shape} vs {other.shape}")

print("\n" + "=" * 80)
print("Sample data info:")
print("=" * 80)
print(f"Number of timesteps in episode: {len(df)}")
print(f"Episode columns with 'state' or 'observation': {[c for c in df.columns if 'state' in c.lower() or 'observation' in c.lower()]}")


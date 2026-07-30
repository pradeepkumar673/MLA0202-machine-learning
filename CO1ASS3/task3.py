"""
Experiment 3: Information Theory
Dataset: Play Tennis Dataset
Task: Calculate the entropy and information gain for all input attributes and
      identify the attribute with the highest information gain.
"""
import math
import pandas as pd

df = pd.read_csv("data_play_tennis.csv")
print("Play Tennis dataset:")
print(df)

target_col = "PlayTennis"
attributes = [c for c in df.columns if c not in ("Day", target_col)]

def entropy(series):
    counts = series.value_counts()
    probs = counts / len(series)
    return -sum(p * math.log2(p) for p in probs if p > 0)

target_entropy = entropy(df[target_col])
print(f"\nTarget attribute: '{target_col}'")
print("Class distribution:")
print(df[target_col].value_counts())
print(f"\nEntropy of the target attribute H({target_col}) = {target_entropy:.4f} bits")


def information_gain(df, attribute, target_col, base_entropy):
    weighted_entropy = 0.0
    print(f"\n  Attribute: {attribute}")
    for value, subset in df.groupby(attribute):
        weight = len(subset) / len(df)
        subset_entropy = entropy(subset[target_col])
        weighted_entropy += weight * subset_entropy
        print(f"    {attribute} = {value:<9} | count = {len(subset)} | "
              f"H(S_{value}) = {subset_entropy:.4f} | weight = {weight:.4f}")
    gain = base_entropy - weighted_entropy
    print(f"    Weighted entropy after split = {weighted_entropy:.4f}")
    print(f"    Information Gain IG({target_col}, {attribute}) = "
          f"{base_entropy:.4f} - {weighted_entropy:.4f} = {gain:.4f}")
    return gain

print("\n--- Calculating Information Gain for every attribute ---")
gains = {}
for attr in attributes:
    gains[attr] = information_gain(df, attr, target_col, target_entropy)
gain_series = pd.Series(gains).sort_values(ascending=False)
print("\n--- Information Gain summary (sorted, highest first) ---")
print(gain_series.to_string())

best_attribute = gain_series.idxmax()
print(f"\nAttribute with the HIGHEST Information Gain: '{best_attribute}' "
      f"(IG = {gain_series.max():.4f} bits)")
print(f"=> '{best_attribute}' should be selected as the root/splitting node "
      f"of a decision tree built on this dataset.")

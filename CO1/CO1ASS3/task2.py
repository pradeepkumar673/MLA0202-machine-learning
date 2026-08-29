"""
Experiment 2: Bayes Decision Theory
Dataset: SMS Spam Collection Dataset
Task: Implement Bayes' theorem to calculate the posterior probability for a
      given message and classify it as Spam or Ham (Not Spam).
"""
import re
import math
import pandas as pd
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split

df = pd.read_csv("data_sms.tsv", sep="\t", header=None, names=["label", "message"])
print("Dataset shape:", df.shape)
print("Class distribution:")
print(df["label"].value_counts())

def tokenize(text):
    text = text.lower()
    return re.findall(r"[a-z']+", text)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])


class_counts = train_df["label"].value_counts()
n_train = len(train_df)
priors = {cls: count / n_train for cls, count in class_counts.items()}
print("\nPrior probabilities computed from the training set:")
for cls, p in priors.items():
    print(f"  P({cls}) = {p:.4f}")

word_counts = {cls: Counter() for cls in class_counts.index}
total_words = {cls: 0 for cls in class_counts.index}

for _, row in train_df.iterrows():
    tokens = tokenize(row["message"])
    word_counts[row["label"]].update(tokens)
    total_words[row["label"]] += len(tokens)

vocabulary = set()
for cls in word_counts:
    vocabulary.update(word_counts[cls].keys())
V = len(vocabulary)
print(f"Vocabulary size (training set): {V}")

def word_likelihood(word, cls):
    """P(word | class) with Laplace (add-1) smoothing."""
    count = word_counts[cls].get(word, 0)
    return (count + 1) / (total_words[cls] + V)


def classify_message(message):
    tokens = tokenize(message)
    log_posterior = {}
    for cls in priors:
        log_prob = math.log(priors[cls])
        for tok in tokens:
            log_prob += math.log(word_likelihood(tok, cls))
        log_posterior[cls] = log_prob
    # convert back from log-space to normalized posterior probabilities
    max_log = max(log_posterior.values())
    exp_vals = {c: math.exp(v - max_log) for c, v in log_posterior.items()}
    total = sum(exp_vals.values())
    posterior = {c: v / total for c, v in exp_vals.items()}
    predicted = max(posterior, key=posterior.get)
    return predicted, posterior

new_message = "Congratulations! You have WON a free ticket to Bahamas, call now to claim your prize!!!"
predicted_class, posterior_probs = classify_message(new_message)

print("\n--- Classifying a new message using Bayes' theorem ---")
print(f"Message: \"{new_message}\"")
print("Posterior probabilities:")
for cls, p in posterior_probs.items():
    print(f"  P({cls} | message) = {p:.10f}")
print(f"\nPredicted class: {predicted_class.upper()}")

# A second example - an everyday, non-spam style message
new_message_2 = "Hey, are we still meeting for lunch tomorrow at 1pm?"
predicted_class_2, posterior_probs_2 = classify_message(new_message_2)
print("\n--- Classifying a second message ---")
print(f"Message: \"{new_message_2}\"")
for cls, p in posterior_probs_2.items():
    print(f"  P({cls} | message) = {p:.10f}")
print(f"\nPredicted class: {predicted_class_2.upper()}")


correct = 0
for _, row in test_df.iterrows():
    pred, _ = classify_message(row["message"])
    if pred == row["label"]:
        correct += 1
accuracy = correct / len(test_df)
print(f"\nAccuracy of the Bayes' theorem classifier on {len(test_df)} test messages: {accuracy:.4f}")

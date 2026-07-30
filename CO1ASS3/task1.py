"""
Experiment 1: Probability Theory
Dataset: Breast Cancer Wisconsin Diagnostic Dataset
Task: Calculate the probability of each class (Malignant / Benign) and use the
      computed probabilities to predict the class of a new data instance.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")          # 0 = Malignant, 1 = Benign
class_names = {0: "Malignant", 1: "Benign"}

print("Dataset shape:", X.shape)
print("Class distribution (raw counts):")
print(y.value_counts().rename(index=class_names))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

priors = y_train.value_counts(normalize=True).sort_index()
print("\nPrior probabilities computed from the training set:")
for cls, p in priors.items():
    print(f"  P({class_names[cls]}) = {p:.4f}")

means = X_train.groupby(y_train).mean()
stds = X_train.groupby(y_train).std()

def gaussian_pdf(x, mean, std):
    eps = 1e-9
    coeff = 1.0 / (np.sqrt(2 * np.pi) * (std + eps))
    exponent = np.exp(-((x - mean) ** 2) / (2 * (std + eps) ** 2))
    return coeff * exponent

def predict_instance(x_row):
    """Combine prior * likelihood (Bayes numerator) for each class."""
    log_posterior = {}
    for cls in priors.index:
        likelihoods = gaussian_pdf(x_row, means.loc[cls], stds.loc[cls])
        log_likelihood = np.sum(np.log(likelihoods + 1e-300))
        log_posterior[cls] = np.log(priors[cls]) + log_likelihood
    # normalize back from log-space for interpretable probabilities
    max_log = max(log_posterior.values())
    exp_vals = {c: np.exp(v - max_log) for c, v in log_posterior.items()}
    total = sum(exp_vals.values())
    posterior = {c: v / total for c, v in exp_vals.items()}
    predicted = max(posterior, key=posterior.get)
    return predicted, posterior


new_instance = X_test.iloc[0]
true_label = y_test.iloc[0]

predicted_class, posterior_probs = predict_instance(new_instance)

print("\n--- Predicting the class of a new data instance (test sample #0) ---")
print("Posterior probabilities computed for the new instance:")
for cls, p in posterior_probs.items():
    print(f"  P({class_names[cls]} | data) = {p:.6f}")

print(f"\nPredicted class : {class_names[predicted_class]}")
print(f"Actual class    : {class_names[true_label]}")
print("Prediction correct:", predicted_class == true_label)


correct = 0
for i in range(len(X_test)):
    pred, _ = predict_instance(X_test.iloc[i])
    if pred == y_test.iloc[i]:
        correct += 1
accuracy = correct / len(X_test)
print(f"\nAccuracy of the probability-based classifier on {len(X_test)} test instances: {accuracy:.4f}")

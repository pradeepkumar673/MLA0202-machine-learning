import numpy as np
import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import metrics

states = ['LaneKeep', 'LaneChange', 'Braking']
obs_seq = [('HighSpeed', 'Straight', 'ZeroAcc'),
           ('MediumSpeed', 'SteeringLeft', 'NegAcc'),
           ('LowSpeed', 'Straight', 'NegAcc')]

prior = np.array([0.6, 0.2, 0.2])
trans = np.array([[0.8, 0.1, 0.1],
                  [0.2, 0.7, 0.1],
                  [0.3, 0.1, 0.6]])
emission = np.array([[0.7, 0.2, 0.1],
                     [0.1, 0.8, 0.1],
                     [0.1, 0.1, 0.8]])

seq_len = len(obs_seq)
viterbi = np.zeros((len(states), seq_len))
viterbi[:, 0] = prior * emission[:, 0]

for t in range(1, seq_len):
    for s in range(len(states)):
        viterbi[s, t] = np.max(viterbi[:, t-1] * trans[:, s]) * emission[s, t]

best_path = [states[np.argmax(viterbi[:, t])] for t in range(seq_len)]

train_data = [
    [("HighSpeed", "LaneKeep"), ("SteeringLeft", "LaneChange"), ("NegAcc", "Braking")],
    [("HighSpeed", "LaneKeep"), ("HighSpeed", "LaneKeep"), ("NegAcc", "Braking")],
    [("SteeringRight", "LaneChange"), ("MediumSpeed", "LaneChange"), ("NegAcc", "Braking")]
]

test_data = [
    [("HighSpeed", "LaneKeep"), ("SteeringLeft", "LaneChange"), ("NegAcc", "Braking")]
]

def extract_features(seq):
    feat_seq = []
    for i in range(len(seq)):
        word = seq[i][0]
        feat = {
            'bias': 1.0,
            'feature': word,
            'is_speed': 'Speed' in word,
            'is_steering': 'Steering' in word,
            'is_acc': 'Acc' in word,
            'prev_feat': '' if i == 0 else seq[i-1][0],
            'next_feat': '' if i == len(seq)-1 else seq[i+1][0]
        }
        feat_seq.append(feat)
    return feat_seq

X_train = [extract_features(s) for s in train_data]
y_train = [[label for token, label in s] for s in train_data]

X_test = [extract_features(s) for s in test_data]
y_test = [[label for token, label in s] for s in test_data]

crf = sklearn_crfsuite.CRF(algorithm='lbfgs', max_iterations=100)
crf.fit(X_train, y_train)

y_pred = crf.predict(X_test)
crf_acc = metrics.flat_accuracy_score(y_test, y_pred)

print("--- QUESTION 2: HMM VS CRF FOR AUTONOMOUS VEHICLE ACTIVITY RECOGNITION ---")
print("\n[HMM Viterbi Predicted Sequence]")
print(best_path)

print("\n[CRF Predicted Sequence]")
print(y_pred[0])
print("CRF Sequence Accuracy:", crf_acc)

comp_data = {
    "Evaluation Parameter": [
        "State Representation",
        "Observation Modeling",
        "Learning Approach",
        "Prediction Accuracy",
        "Contextual Features",
        "Computational Efficiency",
        "Scalability"
    ],
    "Hidden Markov Model (HMM)": [
        "Generative (Directed sequence)",
        "Assumes observations independent given state",
        "Maximum Likelihood (Baum-Welch / EM)",
        "Moderate (Suffer from independence assumption)",
        "Poor (Strict independence assumption)",
        "High O(T * N^2) fast inference",
        "Scales well for simple discrete sequences"
    ],
    "Conditional Random Field (CRF)": [
        "Discriminative (Undirected sequence)",
        "Models conditional P(Y|X) directly",
        "Convex optimization (L-BFGS / Gradient)",
        "High (Handles overlapping features)",
        "Excellent (Rich sliding window context)",
        "Moderate (Slower training due to normalization)",
        "Highly scalable for complex multi-sensor data"
    ]
}

df = pd.DataFrame(comp_data)
print("\n[HMM vs CRF Evaluation Table]")
print(df.to_string(index=False))

print("\n[Recommendation & Justification]")
print("Conditional Random Field (CRF) is recommended for autonomous driving activity recognition because continuous vehicle sensor streams (speed, acceleration, steering angle) exhibit complex temporal correlations and overlapping features. CRF models P(Y|X) directly without forcing strong independence assumptions, avoiding label bias and yielding significantly higher accuracy.")

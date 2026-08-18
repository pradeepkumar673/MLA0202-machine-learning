import numpy as np

hidden_states = ['Rainy', 'Sunny']
observations = ['Sunny', 'Cloudy', 'Rainy']

start_prob = np.array([0.6, 0.4])

transition_prob = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

emission_prob = np.array([
    [0.1, 0.4, 0.5],
    [0.6, 0.3, 0.1]
])

obs_sequence = ['Cloudy', 'Sunny', 'Rainy', 'Rainy']
obs_indices = [observations.index(o) for o in obs_sequence]

n_states = len(hidden_states)
n_steps = len(obs_sequence)

viterbi_matrix = np.zeros((n_states, n_steps))
backpointer_matrix = np.zeros((n_states, n_steps), dtype=int)

viterbi_matrix[:, 0] = start_prob * emission_prob[:, obs_indices[0]]

for t in range(1, n_steps):
    for s in range(n_states):
        probabilities = viterbi_matrix[:, t-1] * transition_prob[:, s]
        backpointer_matrix[s, t] = np.argmax(probabilities)
        viterbi_matrix[s, t] = probabilities[backpointer_matrix[s, t]] * emission_prob[s, obs_indices[t]]

best_path = np.zeros(n_steps, dtype=int)
best_path[-1] = np.argmax(viterbi_matrix[:, -1])

for t in range(n_steps - 2, -1, -1):
    best_path[t] = backpointer_matrix[best_path[t+1], t+1]

predicted_states = [hidden_states[i] for i in best_path]

print("Observations:", obs_sequence)
print("Predicted Hidden States:", predicted_states)
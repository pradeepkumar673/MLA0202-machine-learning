states = ["Healthy", "Warning", "Failure"]

observations = ["Normal", "High", "Critical", "High", "Critical"]

start_probability = {
    "Healthy": 0.7,
    "Warning": 0.2,
    "Failure": 0.1
}

transition_probability = {
    "Healthy": {
        "Healthy": 0.7,
        "Warning": 0.25,
        "Failure": 0.05
    },
    "Warning": {
        "Healthy": 0.2,
        "Warning": 0.6,
        "Failure": 0.2
    },
    "Failure": {
        "Healthy": 0.05,
        "Warning": 0.25,
        "Failure": 0.7
    }
}

emission_probability = {
    "Healthy": {
        "Normal": 0.8,
        "High": 0.18,
        "Critical": 0.02
    },
    "Warning": {
        "Normal": 0.2,
        "High": 0.6,
        "Critical": 0.2
    },
    "Failure": {
        "Normal": 0.02,
        "High": 0.18,
        "Critical": 0.8
    }
}

viterbi = {}
paths = {}

for state in states:
    viterbi[state] = (
        start_probability[state]
        * emission_probability[state][observations[0]]
    )
    paths[state] = [state]

for observation in observations[1:]:
    new_viterbi = {}
    new_paths = {}

    for current_state in states:
        best_probability = 0
        best_previous_state = ""

        for previous_state in states:
            probability = (
                viterbi[previous_state]
                * transition_probability[previous_state][current_state]
                * emission_probability[current_state][observation]
            )

            if probability > best_probability:
                best_probability = probability
                best_previous_state = previous_state

        new_viterbi[current_state] = best_probability
        new_paths[current_state] = (
            paths[best_previous_state] + [current_state]
        )

    viterbi = new_viterbi
    paths = new_paths

final_state = max(viterbi, key=viterbi.get)
best_path = paths[final_state]

print("Hidden Markov Model")
print("-------------------")

print("Sensor Observations:")
print(observations)

print("\nPredicted Machine States:")

for i in range(len(observations)):
    print(observations[i], "->", best_path[i])

print("\nMost Likely Final State:", final_state)

if final_state == "Healthy":
    print("Maintenance Status: Machine is operating normally")
elif final_state == "Warning":
    print("Maintenance Status: Maintenance should be planned")
else:
    print("Maintenance Status: Immediate maintenance required")
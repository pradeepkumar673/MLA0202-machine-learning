states = ["Healthy", "Warning", "Failure"]

observations = ["Normal", "High", "Critical", "High"]

emission = {
    "Healthy": [0.8, 0.2, 0.0],
    "Warning": [0.2, 0.6, 0.2],
    "Failure": [0.0, 0.2, 0.8]
}

result = []

for observation in observations:
    if observation == "Normal":
        state = "Healthy"
    elif observation == "High":
        state = "Warning"
    else:
        state = "Failure"

    result.append(state)

print("Observations:", observations)
print("Predicted States:", result)
print("Final Machine State:", result[-1])
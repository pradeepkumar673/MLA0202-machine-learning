from itertools import product

amounts = ["Low", "Medium", "High"]
locations = ["Local", "Foreign"]
devices = ["Known", "Unknown"]
histories = ["No", "Yes"]

p_amount = {
    "Low": 0.6,
    "Medium": 0.3,
    "High": 0.1
}

p_location = {
    "Local": 0.8,
    "Foreign": 0.2
}

p_device = {
    "Known": 0.85,
    "Unknown": 0.15
}

p_history = {
    "No": 0.9,
    "Yes": 0.1
}

p_fraud = {
    ("Low", "Local", "Known", "No"): 0.01,
    ("Low", "Local", "Known", "Yes"): 0.10,
    ("Low", "Local", "Unknown", "No"): 0.05,
    ("Low", "Local", "Unknown", "Yes"): 0.30,
    ("Low", "Foreign", "Known", "No"): 0.05,
    ("Low", "Foreign", "Known", "Yes"): 0.30,
    ("Low", "Foreign", "Unknown", "No"): 0.15,
    ("Low", "Foreign", "Unknown", "Yes"): 0.50,

    ("Medium", "Local", "Known", "No"): 0.03,
    ("Medium", "Local", "Known", "Yes"): 0.15,
    ("Medium", "Local", "Unknown", "No"): 0.10,
    ("Medium", "Local", "Unknown", "Yes"): 0.40,
    ("Medium", "Foreign", "Known", "No"): 0.10,
    ("Medium", "Foreign", "Known", "Yes"): 0.40,
    ("Medium", "Foreign", "Unknown", "No"): 0.25,
    ("Medium", "Foreign", "Unknown", "Yes"): 0.60,

    ("High", "Local", "Known", "No"): 0.05,
    ("High", "Local", "Known", "Yes"): 0.25,
    ("High", "Local", "Unknown", "No"): 0.20,
    ("High", "Local", "Unknown", "Yes"): 0.60,
    ("High", "Foreign", "Known", "No"): 0.20,
    ("High", "Foreign", "Known", "Yes"): 0.60,
    ("High", "Foreign", "Unknown", "No"): 0.40,
    ("High", "Foreign", "Unknown", "Yes"): 0.80
}

amount = "High"
location = "Foreign"
device = "Unknown"
history = "Yes"

def calculate_probability(fraud):
    key = (amount, location, device, history)

    probability = (
        p_amount[amount]
        * p_location[location]
        * p_device[device]
        * p_history[history]
    )

    if fraud:
        probability *= p_fraud[key]
    else:
        probability *= 1 - p_fraud[key]

    return probability

fraud = calculate_probability(True)
not_fraud = calculate_probability(False)

total = fraud + not_fraud

fraud_probability = fraud / total
not_fraud_probability = not_fraud / total

print("Bayesian Network Fraud Detection")
print("--------------------------------")
print("Transaction Amount:", amount)
print("Transaction Location:", location)
print("Login Device:", device)
print("Previous Fraud History:", history)

print("\nFraud Probability:", round(fraud_probability * 100, 2), "%")
print("Not Fraud Probability:", round(not_fraud_probability * 100, 2), "%")

if fraud_probability > not_fraud_probability:
    print("Prediction: FRAUDULENT")
else:
    print("Prediction: NOT FRAUDULENT")
from itertools import product

p = {
    "Low": 0.02, "Medium": 0.10, "High": 0.25
}

amount = "High"
location = "Foreign"
device = "Unknown"
history = "Yes"

fraud = p[amount]

if location == "Foreign":
    fraud += 0.15
if device == "Unknown":
    fraud += 0.20
if history == "Yes":
    fraud += 0.25

fraud = min(fraud, 0.95)

print("Transaction:", amount, location, device, history)
print("Fraud Probability:", round(fraud * 100, 2), "%")

if fraud > 0.5:
    print("Prediction: Fraudulent")
else:
    print("Prediction: Not Fraudulent")
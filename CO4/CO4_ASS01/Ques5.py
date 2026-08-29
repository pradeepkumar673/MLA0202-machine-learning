import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

np.random.seed(42)
n = 1000

age = np.random.choice(['Young', 'Middle', 'Senior'], n)
income = np.random.choice(['Low', 'Medium', 'High'], n)
vehicle = np.random.choice(['Sedan', 'SUV', 'Truck'], n)

claim_prob = []
for a, i, v in zip(age, income, vehicle):
    p = 0.1
    if a == 'Senior': p += 0.2
    if i == 'Low': p += 0.2
    if v == 'Truck': p += 0.1
    claim_prob.append(min(p, 0.9))

claim = ['Yes' if np.random.rand() < p else 'No' for p in claim_prob]

data = pd.DataFrame({'Age': age, 'Income': income, 'VehicleType': vehicle, 'Claim': claim})

hc = HillClimbSearch(data)
best_model = hc.estimate(scoring_method=BicScore(data))

network = BayesianNetwork(best_model.edges())
network.fit(data)

inference = VariableElimination(network)

result = inference.query(variables=['Claim'], evidence={'Age': 'Senior', 'Income': 'Low', 'VehicleType': 'Truck'})

print("Learned Network Edges:", list(network.edges()))
print("\nPrediction Result for Senior, Low Income, Truck:")
print(result)
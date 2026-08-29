import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

bn_model = BayesianNetwork([
    ('MedicalHistory', 'Disease'),
    ('Disease', 'Fever'),
    ('Disease', 'Cough')
])

cpd_history = TabularCPD(variable='MedicalHistory', variable_card=2, values=[[0.7], [0.3]])

cpd_disease = TabularCPD(
    variable='Disease',
    variable_card=2,
    values=[[0.9, 0.4],
            [0.1, 0.6]],
    evidence=['MedicalHistory'],
    evidence_card=[2]
)

cpd_fever = TabularCPD(
    variable='Fever',
    variable_card=2,
    values=[[0.85, 0.15],
            [0.15, 0.85]],
    evidence=['Disease'],
    evidence_card=[2]
)

cpd_cough = TabularCPD(
    variable='Cough',
    variable_card=2,
    values=[[0.8, 0.2],
            [0.2, 0.8]],
    evidence=['Disease'],
    evidence_card=[2]
)

bn_model.add_cpds(cpd_history, cpd_disease, cpd_fever, cpd_cough)

bn_infer = VariableElimination(bn_model)
bn_result = bn_infer.query(variables=['Disease'], evidence={'Fever': 1, 'Cough': 1, 'MedicalHistory': 1})

psi_history_disease = np.array([[0.63, 0.07], [0.12, 0.18]])
psi_disease_fever = np.array([[0.85, 0.15], [0.15, 0.85]])
psi_disease_cough = np.array([[0.80, 0.20], [0.20, 0.80]])

joint_mrf = psi_history_disease[1, :] * psi_disease_fever[:, 1] * psi_disease_cough[:, 1]
partition_z = np.sum(joint_mrf)
mrf_prob = joint_mrf / partition_z

print("--- QUESTION 1: BAYESIAN NETWORK VS MARKOV RANDOM FIELD FOR DISEASE DIAGNOSIS ---")
print("\n[Bayesian Network Inference Result]")
print(bn_result)

print("\n[Markov Random Field Normalized Joint Probabilities]")
print("P(Disease=No | Symptoms) =", round(mrf_prob[0], 4))
print("P(Disease=Yes | Symptoms) =", round(mrf_prob[1], 4))

comparison_data = {
    "Parameter": [
        "Graph Structure",
        "Conditional Independence",
        "Inference Process",
        "Uncertainty Handling",
        "Computational Complexity",
        "Real-World Applicability"
    ],
    "Bayesian Network (BN)": [
        "Directed Acyclic Graph (DAG)",
        "d-Separation (Causal direction)",
        "Exact Variable Elimination / Junction Tree",
        "Conditional Probability Tables (CPDs)",
        "Linear in CPD parameters O(K^N)",
        "High (Causal Symptom-Disease Modeling)"
    ],
    "Markov Random Field (MRF)": [
        "Undirected Graph",
        "Global & Local Markov Blanket",
        "Loopy Belief Propagation / MCMC",
        "Factor Potentials & Energy Functions",
        "NP-hard due to Partition Function Z",
        "Moderate (Spatial/Symmetric Relations)"
    ]
}

df_comp = pd.DataFrame(comparison_data)
print("\n[Comparative Analysis Table]")
print(df_comp.to_string(index=False))

print("\n[Justification & Recommendation]")
print("Bayesian Network is more appropriate for healthcare disease diagnosis because medical relationships are naturally directed (Disease causes Symptoms). Prior probabilities and CPDs are easily interpretable by physicians, and normalized local factors eliminate the heavy partition function computation.")

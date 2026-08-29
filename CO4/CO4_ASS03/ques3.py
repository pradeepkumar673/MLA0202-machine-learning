import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

bn_risk = BayesianNetwork([
    ('Income', 'DebtRatio'),
    ('CreditScore', 'DebtRatio'),
    ('DebtRatio', 'DefaultRisk')
])

cpd_inc = TabularCPD('Income', 2, [[0.6], [0.4]])
cpd_score = TabularCPD('CreditScore', 2, [[0.7], [0.3]])

cpd_debt = TabularCPD(
    'DebtRatio', 2,
    [[0.9, 0.6, 0.4, 0.1],
     [0.1, 0.4, 0.6, 0.9]],
    evidence=['Income', 'CreditScore'],
    evidence_card=[2, 2]
)

cpd_default = TabularCPD(
    'DefaultRisk', 2,
    [[0.95, 0.30],
     [0.05, 0.70]],
    evidence=['DebtRatio'],
    evidence_card=[2]
)

bn_risk.add_cpds(cpd_inc, cpd_score, cpd_debt, cpd_default)

infer = VariableElimination(bn_risk)
res = infer.query(variables=['DefaultRisk'], evidence={'Income': 0, 'CreditScore': 0})

mrf_psi_inc_score_debt = np.array([0.9, 0.6, 0.4, 0.1, 0.1, 0.4, 0.6, 0.9])
mrf_psi_debt_default = np.array([0.95, 0.30, 0.05, 0.70])

unnorm_prob_high_risk = mrf_psi_inc_score_debt[3] * mrf_psi_debt_default[3]
unnorm_prob_low_risk = mrf_psi_inc_score_debt[0] * mrf_psi_debt_default[0]
z_partition = unnorm_prob_high_risk + unnorm_prob_low_risk

mrf_prob_default = unnorm_prob_high_risk / z_partition

print("--- QUESTION 3: DIRECTED VS UNDIRECTED GRAPHICAL MODELS FOR FINANCIAL RISK PREDICTION ---")
print("\n[Bayesian Network Default Risk Inference]")
print(res)

print("\n[Markov Random Field Calculated Default Probability]")
print("P(DefaultRisk=High | Low Income & Low Credit Score) =", round(mrf_prob_default, 4))

eval_metrics = {
    "Evaluation Criterion": [
        "Graph Representation",
        "Dependency Modeling",
        "Inference Techniques",
        "Learning Algorithms",
        "Generalization Capability",
        "Computational Requirements",
        "Suitability for Financial Risk"
    ],
    "Directed Models (Bayesian Networks)": [
        "Directed Acyclic Graph (DAG)",
        "Causal & asymmetric relationships",
        "Variable Elimination / Belief Propagation",
        "Maximum Likelihood / Bayesian Estimation",
        "High (Prevents over-fitting via local CPDs)",
        "Efficient (No global partition function Z)",
        "Excellent (Interpretable causal credit chains)"
    ],
    "Undirected Models (MRFs)": [
        "Undirected Graph",
        "Symmetric & cyclic mutual dependencies",
        "Junction Tree / MCMC sampling",
        "Iterative Proportional Fitting (IPF) / Gradient",
        "Moderate (Prone to local energy minima)",
        "High (Requires computing partition function Z)",
        "Limited (Lacks explicit causal directionality)"
    ]
}

df_eval = pd.DataFrame(eval_metrics)
print("\n[Evaluation Framework Table]")
print(df_eval.to_string(index=False))

print("\n[Conclusion & Recommendation]")
print("Directed Graphical Models (Bayesian Networks) are more effective for financial risk prediction. Financial risk features follow clear hierarchical causality (Income and Credit Score directly influence Debt-to-Income Ratio, which determines Default Risk). Directed models provide conditional probability tables essential for banking regulatory auditability, avoiding intractable partition functions.")

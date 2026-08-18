from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = BayesianNetwork([('Obesity', 'Diabetes'), ('HighBloodSugar', 'Diabetes')])

cpd_obesity = TabularCPD(variable='Obesity', variable_card=2, values=[[0.7], [0.3]])

cpd_hbs = TabularCPD(variable='HighBloodSugar', variable_card=2, values=[[0.8], [0.2]])

cpd_diabetes = TabularCPD(
    variable='Diabetes',
    variable_card=2,
    values=[
        [0.95, 0.80, 0.40, 0.05],
        [0.05, 0.20, 0.60, 0.95]
    ],
    evidence=['Obesity', 'HighBloodSugar'],
    evidence_card=[2, 2]
)

model.add_cpds(cpd_obesity, cpd_hbs, cpd_diabetes)
model.check_model()

inference = VariableElimination(model)
result = inference.query(variables=['Diabetes'], evidence={'Obesity': 1, 'HighBloodSugar': 1})

print(result)
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination, BeliefPropagation,CausalInference, ApproxInference
from pgmpy.sampling import BayesianModelSampling

data = pd.read_csv("./data/natural_disasters_discretized_en.csv")

# Definir la estructura de la red bow-tie con al menos 10 nodos
model = BayesianNetwork() # se crea una instancia de la clase BayesianNetwork de pgmpy. Esta clase representa el modelo de red bayesiana que estamos construyendo.


model.add_edges_from([ #cada tupla representa una conexion entre dos nodos
    #el nodo central es DisasterType
    #Esto son las causas o factores que influyen en el tipo de desastre
    ("PreparednessTraining", "DisasterType"),
    ("CommunityAwareness", "DisasterType"),
    ("AgenciesInvolved", "DisasterType"),
    ("CoordinationScore", "DisasterType"),
    ("CommunicationScore", "DisasterType"),
    #Consecuencias o impactos que tienen diferentes aspectos del desastre
    ("DisasterType", "Personnel"),
    ("DisasterType", "MedicalSupplies"),
    ("DisasterType", "Shelters"),
    ("DisasterType", "FoodWater"),
    ("DisasterType", "ResponseTime"),
    ("DisasterType", "ResponseDuration"),
    ("DisasterType", "PeopleAffected"),
    ("DisasterType", "PeopleAssisted"),
    ("DisasterType", "InfrastructureCondition"),
    ("DisasterType", "EconomicDamage"),
    ("DisasterType", "Casualties"),
    ("DisasterType", "LongTermImpact")
])

# Estimar las probabilidades condicionales a partir de los datos
estimator = MaximumLikelihoodEstimator(model, data)
cpd_preparedness_training = estimator.estimate_cpd('PreparednessTraining')
cpd_community_awareness = estimator.estimate_cpd('CommunityAwareness')
cpd_agencies_involved = estimator.estimate_cpd('AgenciesInvolved')
cpd_coordination_score = estimator.estimate_cpd('CoordinationScore')
cpd_communication_score = estimator.estimate_cpd('CommunicationScore')
cpd_disaster_type = estimator.estimate_cpd('DisasterType')
cpd_personnel = estimator.estimate_cpd('Personnel')
cpd_medical_supplies = estimator.estimate_cpd('MedicalSupplies')
cpd_shelters = estimator.estimate_cpd('Shelters')
cpd_food_water = estimator.estimate_cpd('FoodWater')
cpd_response_time = estimator.estimate_cpd('ResponseTime')
cpd_response_duration = estimator.estimate_cpd('ResponseDuration')
cpd_people_affected = estimator.estimate_cpd('PeopleAffected')
cpd_people_assisted = estimator.estimate_cpd('PeopleAssisted')
cpd_infrastructure_condition = estimator.estimate_cpd('InfrastructureCondition')
cpd_economic_damage = estimator.estimate_cpd('EconomicDamage')
cpd_casualties = estimator.estimate_cpd('Casualties')
cpd_long_term_impact = estimator.estimate_cpd('LongTermImpact')


# Agregar CPDs al modelo
model.add_cpds(
    cpd_preparedness_training, cpd_community_awareness, cpd_agencies_involved, cpd_coordination_score,
    cpd_communication_score, cpd_disaster_type, cpd_personnel, cpd_medical_supplies, cpd_shelters,
    cpd_food_water, cpd_response_time, cpd_response_duration, cpd_people_affected, cpd_people_assisted,
    cpd_infrastructure_condition, cpd_economic_damage, cpd_casualties, cpd_long_term_impact
)

# Comprobar si el modelo está correctamente definido
assert model.check_model() #no muestra ningun mensaje de error por lo que esta correctamente definido

#Algoritmos de inferencia exacta
#¿Cuál es la probabilidad del riesgo del sistema si PreparednessTraining ocurre?
# Variable Elimination
ve_inference = VariableElimination(model)
marginal_probability_ve = ve_inference.query(variables=['DisasterType'], evidence={'PreparednessTraining': 'Yes'})

# Belief Propagation
bp_inference = BeliefPropagation(model)
marginal_probability_bp = bp_inference.query(variables=['DisasterType'], evidence={'PreparednessTraining': 'Yes'})

# Causal inference
inference = CausalInference(model)
marginal_probability_ca = inference.query(variables=['DisasterType'], evidence={'PreparednessTraining': 'Yes'})

print("Variable Elimination:\n", marginal_probability_ve)
print("Belief Propagation:\n", marginal_probability_bp)
print("Causal Inference:\n", marginal_probability_ca)

#Algoritmos de inferencia aproximada
# Sampling Importance Resampling (SIR)
sampling = BayesianModelSampling(model)
samples = sampling.likelihood_weighted_sample(size=1000, evidence=[('PreparednessTraining', 'Yes')])
approximate_probability_sir = samples['DisasterType'].value_counts(normalize=True)

print("Sampling Importance Resampling (SIR):\n", approximate_probability_sir)

#ApproxInference
aprox_inference = ApproxInference(model)
marginal_probability_approx = aprox_inference.query(variables=['DisasterType'], evidence={'PreparednessTraining': 'Yes'})

print("Approximate Inference:\n", marginal_probability_approx)

#¿Cuál es la probabilidad del riesgo del sistema si CommunityAwareness ocurre?
# Variable Elimination
ve_inference2 = VariableElimination(model)
marginal_probability_ve2 = ve_inference2.query(variables=['DisasterType'], evidence={'CommunityAwareness': 'Yes'})

# Belief Propagation
bp_inference2 = BeliefPropagation(model)
marginal_probability_bp2 = bp_inference2.query(variables=['DisasterType'], evidence={'CommunityAwareness': 'Yes'})

# Causal inference
inference2 = CausalInference(model)
marginal_probability_ca2 = inference2.query(variables=['DisasterType'], evidence={'CommunityAwareness': 'Yes'})

print("Variable Elimination:\n", marginal_probability_ve2)
print("Belief Propagation:\n", marginal_probability_bp2)
print("Causal Inference:\n", marginal_probability_ca2)

#Algoritmos de inferencia aproximada
# Sampling Importance Resampling (SIR)
sampling2 = BayesianModelSampling(model)
samples2 = sampling2.likelihood_weighted_sample(size=1000, evidence=[('CommunityAwareness', 'Yes')])
approximate_probability_sir2 = samples2['DisasterType'].value_counts(normalize=True)

print("Sampling Importance Resampling (SIR):\n", approximate_probability_sir2)

#ApproxInference
aprox_inference2 = ApproxInference(model)
marginal_probability_approx2 = aprox_inference2.query(variables=['DisasterType'], evidence={'CommunityAwareness': 'Yes'})

print("Approximate Inference:\n", marginal_probability_approx2)


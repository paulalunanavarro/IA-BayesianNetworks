import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

data = pd.read_csv("./data/natural_disasters_extended.csv")


# Definir la estructura de la red bow-tie con al menos 10 nodos
model = BayesianNetwork([
    ('CentralNode', 'ResponseTime'),
    ('CentralNode', 'ResponseDuration'),
    ('CentralNode', 'PeopleAffected'),
    ('CentralNode', 'PeopleAssisted'),
    ('CentralNode', 'InfrastructureCondition'),
    ('CentralNode', 'PreparednessTraining'),
    ('CentralNode', 'CommunityAwareness'),
    ('CentralNode', 'AgenciesInvolved'),
    ('CentralNode', 'CoordinationScore'),
    ('CentralNode', 'CommunicationScore'),
    ('CentralNode', 'EconomicDamage'),
    ('CentralNode', 'Casualties'),
    ('CentralNode', 'LongTermImpact')
])

# Define CPDs para los nodos restantes en función de sus padres en el grafo

# CPDs para los nodos principales

# Rellenar los CPDs restantes con los valores adecuados
num_central_nodes = 4

cpd_central_node = TabularCPD(variable='CentralNode', variable_card=num_central_nodes,
                               values=[[0.25], [0.25], [0.25], [0.25]])

cpd_response_time = TabularCPD(variable='ResponseTime', variable_card=2,
                                values=[[0.7, 0.3],
                                        [0.6, 0.4]],
                                evidence=['CentralNode'], evidence_card=[num_central_nodes])

cpd_response_duration = TabularCPD(variable='ResponseDuration', variable_card=2,
                                    values=[[0.8, 0.2],
                                            [0.3, 0.7]],
                                    evidence=['CentralNode'], evidence_card=[num_central_nodes])
 
cpd_people_affected = TabularCPD(variable='PeopleAffected', variable_card=2,
                                  values=[[0.7, 0.3], [0.3, 0.7]],
                                  evidence=['CentralNode'], evidence_card=[2])

cpd_people_assisted = TabularCPD(variable='PeopleAssisted', variable_card=2,
                                  values=[[0.6, 0.7, 0.4, 0.3],
                                          [0.4, 0.3, 0.6, 0.7]],
                                  evidence=['CentralNode', 'ResponseTime'], evidence_card=[2, 2])

cpd_infrastructure_condition = TabularCPD(variable='InfrastructureCondition', variable_card=2,
                                          values=[[0.4, 0.3, 0.3],
                                                  [0.3, 0.4, 0.3]],
                                          evidence=['CentralNode'], evidence_card=[2])

cpd_preparedness_training = TabularCPD(variable='PreparednessTraining', variable_card=2,
                                       values=[[0.5, 0.5], [0.2, 0.8]],
                                       evidence=['CentralNode'], evidence_card=[2])

cpd_community_awareness = TabularCPD(variable='CommunityAwareness', variable_card=2,
                                     values=[[0.4, 0.6], [0.3, 0.7]],
                                     evidence=['CentralNode'], evidence_card=[2])

cpd_agencies_involved = TabularCPD(variable='AgenciesInvolved', variable_card=2,
                                   values=[[0.6, 0.4], [0.4, 0.6]],
                                   evidence=['CentralNode'], evidence_card=[2])

cpd_coordination_score = TabularCPD(variable='CoordinationScore', variable_card=2,
                                    values=[[0.7, 0.3], [0.4, 0.6]],
                                    evidence=['CentralNode'], evidence_card=[2])

cpd_communication_score = TabularCPD(variable='CommunicationScore', variable_card=2,
                                     values=[[0.8, 0.2], [0.3, 0.7]],
                                     evidence=['CentralNode'], evidence_card=[2])

cpd_economic_damage = TabularCPD(variable='EconomicDamage', variable_card=2,
                                  values=[[0.8, 0.2], [0.3, 0.7]],
                                  evidence=['CentralNode'], evidence_card=[2])

cpd_casualties = TabularCPD(variable='Casualties', variable_card=2,
                             values=[[0.7, 0.3], [0.3, 0.7]],
                             evidence=['CentralNode'], evidence_card=[2])

cpd_long_term_impact = TabularCPD(variable='LongTermImpact', variable_card=2,
                                   values=[[0.8, 0.2], [0.3, 0.7]],
                                   evidence=['CentralNode'], evidence_card=[2])

# Agregar todas las CPDs al modelo
model.add_cpds(cpd_central_node, cpd_response_duration, cpd_response_time,cpd_people_affected, cpd_people_assisted, cpd_infrastructure_condition,
                cpd_preparedness_training, cpd_community_awareness, cpd_agencies_involved,
                cpd_coordination_score, cpd_communication_score, cpd_economic_damage,
                cpd_casualties, cpd_long_term_impact)


# Comprobar si el modelo es válido
assert model.check_model()







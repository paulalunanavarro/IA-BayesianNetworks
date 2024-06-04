import pandas as pd
import random
import numpy as np

# Generar datos de ejemplo para simular 1000 líneas
n = 1000
disaster_types = ["Flood", "Hurricane", "Tornado", "Earthquake"]
regions = ["Houston", "New Orleans", "Phoenix", "Austin", "California", "Alaska"]
dates = pd.date_range(start='2010-01-01', end='2024-12-31', periods=n)
latitudes = np.random.uniform(25, 50, n)
longitudes = np.random.uniform(-125, -70, n)
personnel = np.random.randint(50, 500, n)
medical_supplies = np.random.randint(100, 1000, n)
shelters = np.random.randint(10, 100, n)
food_water = np.random.randint(200, 2000, n)
response_times = np.random.uniform(1, 6, n)
response_durations = np.random.uniform(12, 120, n)
people_affected = np.random.randint(1000, 20000, n)
people_assisted = np.random.randint(500, 15000, n)
infrastructure_conditions = ["Poor", "Fair", "Good"]
preparedness_training = ["Yes", "No"]
community_awareness = ["Yes", "No"]
agencies_involved = np.random.randint(3, 10, n)
coordination_scores = np.random.randint(5, 10, n)
communication_scores = np.random.randint(5, 10, n)
economic_damage = np.random.randint(1000000, 1000000000, n)
casualties = np.random.randint(50, 500, n)
long_term_impact = np.random.randint(5, 10, n)

# Crear el DataFrame
data = {
    "EventID": range(1, n+1),
    "DisasterType": [random.choice(disaster_types) for _ in range(n)],
    "Date": dates.strftime('%Y-%m-%d'),
    "Latitude": latitudes,
    "Longitude": longitudes,
    "Region": [random.choice(regions) for _ in range(n)],
    "Personnel": personnel,
    "MedicalSupplies": medical_supplies,
    "Shelters": shelters,
    "FoodWater": food_water,
    "ResponseTime": response_times,
    "ResponseDuration": response_durations,
    "PeopleAffected": people_affected,
    "PeopleAssisted": people_assisted,
    "InfrastructureCondition": [random.choice(infrastructure_conditions) for _ in range(n)],
    "PreparednessTraining": [random.choice(preparedness_training) for _ in range(n)],
    "CommunityAwareness": [random.choice(community_awareness) for _ in range(n)],
    "AgenciesInvolved": agencies_involved,
    "CoordinationScore": coordination_scores,
    "CommunicationScore": communication_scores,
    "EconomicDamage": economic_damage,
    "Casualties": casualties,
    "LongTermImpact": long_term_impact
}

df = pd.DataFrame(data)

# Guardar el DataFrame como archivo CSV
csv_path = "./data/natural_disasters_extended.csv"
df.to_csv(csv_path, index=False)

print("Archivo CSV generado con éxito.")

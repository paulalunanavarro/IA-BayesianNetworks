import pandas as pd

file = "./data/natural_disasters_extended.csv"

pd.set_option('display.max_columns', None)

data = pd.read_csv(file)
print(data)

# Discretizar ResponseTime en categorías 'fast', 'medium' y 'slow'
bins_response_time = [0, 2, 4, float('inf')]  
labels_response_time_en = ['fast', 'medium', 'slow']  
data['ResponseTime_Category'] = pd.cut(data['ResponseTime'], bins=bins_response_time, labels=labels_response_time_en, right=False)

# Discretizar ResponseDuration en categorías 'short', 'medium' y 'long'
bins_response_duration = [0, 30, 60, float('inf')]  
labels_response_duration_en = ['short', 'medium', 'long']  
data['ResponseDuration_Category'] = pd.cut(data['ResponseDuration'], bins=bins_response_duration, labels=labels_response_duration_en, right=False)

# Discretizar PeopleAffected en categorías 'low', 'medium' y 'high'
bins_people_affected = [0, 5000, 10000, float('inf')]  
labels_people_affected_en = ['low', 'medium', 'high'] 
data['PeopleAffected_Category'] = pd.cut(data['PeopleAffected'], bins=bins_people_affected, labels=labels_people_affected_en, right=False)

# Discretizar PeopleAssisted en categorías 'low', 'medium' y 'high'
bins_people_assisted = [0, 3000, 8000, float('inf')]  
labels_people_assisted_en = ['low', 'medium', 'high']  
data['PeopleAssisted_Category'] = pd.cut(data['PeopleAssisted'], bins=bins_people_assisted, labels=labels_people_assisted_en, right=False)

# Discretizar AgenciesInvolved en categorías 'low', 'medium' y 'high'
bins_agencies_involved = [0, 4, 7, float('inf')]
labels_agencies_involved_en = ['low', 'medium', 'high']
data['AgenciesInvolved_Category'] = pd.cut(data['AgenciesInvolved'], bins=bins_agencies_involved, labels=labels_agencies_involved_en, right=False)

# Discretizar CoordinationScore en categorías 'low', 'medium' y 'high'
bins_coordination_score = [0, 6, 8, float('inf')]
labels_coordination_score_en = ['low', 'medium', 'high']
data['CoordinationScore_Category'] = pd.cut(data['CoordinationScore'], bins=bins_coordination_score, labels=labels_coordination_score_en, right=False)

# Discretizar CommunicationScore en categorías 'low', 'medium' y 'high'
bins_communication_score = [0, 6, 8, float('inf')]
labels_communication_score_en = ['low', 'medium', 'high']
data['CommunicationScore_Category'] = pd.cut(data['CommunicationScore'], bins=bins_communication_score, labels=labels_communication_score_en, right=False)

# Discretizar EconomicDamage en categorías 'low', 'medium' y 'high'
bins_economic_damage = [0, 3e8, 6e8, float('inf')]
labels_economic_damage_en = ['low', 'medium', 'high']
data['EconomicDamage_Category'] = pd.cut(data['EconomicDamage'], bins=bins_economic_damage, labels=labels_economic_damage_en, right=False)

# Discretizar Casualties en categorías 'low', 'medium' y 'high'
bins_casualties = [0, 200, 300, float('inf')]
labels_casualties_en = ['low', 'medium', 'high']
data['Casualties_Category'] = pd.cut(data['Casualties'], bins=bins_casualties, labels=labels_casualties_en, right=False)


output_file_path_en = './data/natural_disasters_discretized_en.csv'
data.to_csv(output_file_path_en, index=False)

print(f"Archivo CSV discretizado guardado en: {output_file_path_en}")

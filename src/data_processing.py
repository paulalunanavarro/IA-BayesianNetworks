import pandas as pd

file = "C:/Users/Paula Luna Navarro/Documents/3 ing informatica/2 cuatrimestre/IA/IA-BayesianNetworks/data/natural_disasters_2000_2024.csv"

data = pd.read_csv(file,delimiter=";")
print(data)
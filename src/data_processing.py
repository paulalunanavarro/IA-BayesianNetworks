import pandas as pd

file = "./data/natural_disasters.csv"

pd.set_option('display.max_columns', None)

data = pd.read_csv(file)
print(data)
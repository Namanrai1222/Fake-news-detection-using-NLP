import pandas as pd

fake = pd.read_csv("data/raw/Fake.csv")
true = pd.read_csv("data/raw/True.csv")

print("Fake shape:", fake.shape)
print("True shape:", true.shape)
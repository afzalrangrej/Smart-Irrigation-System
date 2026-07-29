import pandas as pd

# Load dataset
df = pd.read_csv("data/irrigation_prediction.csv")

# First 5 rows
print(df.head())

# Shape
print("\nShape:")
print(df.shape)

# Information
print("\nInfo:")
print(df.info())

# Statistics
print("\nSummary:")
print(df.describe())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Target variable
print("\nTarget Variable:")
print(df["Irrigation_Need"].value_counts())
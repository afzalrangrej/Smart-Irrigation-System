import pandas as pd

df = pd.read_csv("data/irrigation_prediction.csv")

categorical_columns = [
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region"
]

for col in categorical_columns:
    print(f"\n{col}")
    print(df[col].unique())
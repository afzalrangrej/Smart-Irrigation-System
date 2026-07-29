import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

df = pd.read_csv("data/irrigation_prediction.csv")

print(df.shape)

# Distribution of Irrigation Need
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Irrigation_Need")
plt.title("Distribution of Irrigation Need")
plt.show()

# Crop Type Distribution
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Crop_Type")
plt.xticks(rotation=45)
plt.title("Crop Type Distribution")
plt.show()

# Soil Type Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Soil_Type")
plt.title("Soil Type Distribution")
plt.show()

# Temperature Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Temperature_C"], bins=30, kde=True)
plt.title("Temperature Distribution")
plt.show()

# Humidity Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Humidity"], bins=30, kde=True)
plt.title("Humidity Distribution")
plt.show()

# Soil Moisture Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Soil_Moisture"], bins=30, kde=True)
plt.title("Soil Moisture Distribution")
plt.show()
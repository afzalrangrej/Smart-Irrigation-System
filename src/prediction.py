import joblib
import pandas as pd

# Load Model
model = joblib.load("models/irrigation_model.pkl")

# Load Dataset
df = pd.read_csv("data/irrigation_prediction.csv")

# Prepare the Data
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
categorical_columns = df.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Take One Sample
sample = df.drop("Irrigation_Need", axis=1).iloc[[0]]

# Predict
prediction = model.predict(sample)
print("Prediction:", prediction)

# Decode the Prediction
target_encoder = label_encoders["Irrigation_Need"]
print("Predicted Irrigation Need:",
      target_encoder.inverse_transform(prediction))

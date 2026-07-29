from fastapi import FastAPI
from schemas import IrrigationInput
import pandas as pd
import joblib

# ----------------------------
# Load Model & Label Encoders
# ----------------------------
model = joblib.load("../models/irrigation_model.pkl")
label_encoders = joblib.load("../models/label_encoders.pkl")

# ----------------------------
# Create FastAPI App
# ----------------------------
app = FastAPI(
    title="Smart Irrigation API",
    description="Predict irrigation requirement using Machine Learning",
    version="1.0"
)

# ----------------------------
# Home Route
# ----------------------------
@app.get("/")
def home():
    return {
        "message": "Smart Irrigation API is Running!"
    }

# ----------------------------
# Prediction Route
# ----------------------------

@app.post("/predict")
def predict(data: IrrigationInput):

    try:
        print("Received Data:")
        print(data)

        input_data = pd.DataFrame([data.dict()])
        print(input_data)

        categorical_columns = [
            "Soil_Type",
            "Crop_Type",
            "Crop_Growth_Stage",
            "Season",
            "Irrigation_Type",
            "Water_Source",
            "Mulching_Used",
            "Region"
        ]

        for col in categorical_columns:
            print(f"Encoding {col}: {input_data[col].iloc[0]}")
            input_data[col] = label_encoders[col].transform(input_data[col])

        print(input_data)

        prediction = model.predict(input_data)[0]

        prediction = label_encoders["Irrigation_Need"].inverse_transform([prediction])[0]

        print("Prediction:", prediction)

        return {"Prediction": prediction}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
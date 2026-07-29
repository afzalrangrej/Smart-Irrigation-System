import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Load Dataset
df = pd.read_csv("data/irrigation_prediction.csv")

# Encode Categorical Columns
label_encoders = {}
categorical_columns = df.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split Features & Target
X = df.drop("Irrigation_Need", axis=1)
y = df["Irrigation_Need"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Value": [accuracy, precision, recall, f1]
})

metrics_df.to_csv("reports/model_metrics.csv", index=False)

disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.savefig("reports/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# Save the Model
joblib.dump(model, "models/irrigation_model.pkl")
print("Model saved successfully.")

# Create reports folder
os.makedirs("reports", exist_ok=True)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    "reports/feature_importance.csv",
    index=False
)

categorical_columns = df.select_dtypes(include=["object"]).columns

plt.figure(figsize=(8,5))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.gca().invert_yaxis()

plt.title("Feature Importance")

plt.xlabel("Importance")

plt.tight_layout()

plt.savefig(
    "reports/feature_importance.png",
    dpi=300
)

plt.close()

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

pd.DataFrame(report).transpose().to_csv(
    "reports/classification_report.csv"
)

joblib.dump(label_encoders, "models/label_encoders.pkl")

print("Model saved successfully.")
print("Label encoders saved successfully.")
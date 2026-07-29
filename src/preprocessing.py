import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/irrigation_prediction.csv")

# Check data types
print(df.dtypes)

# Encode Categorical Columns
label_encoders = {}
categorical_columns = df.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
print(df.head())

# Separate Features and Target
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

# Verify Shapes
print("Training Features:", X_train.shape)
print("Testing Features :", X_test.shape)
print("Training Labels :", y_train.shape)
print("Testing Labels  :", y_test.shape)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

df = pd.read_csv("Telco Customer Churn.csv")

# Keep ONLY columns used in UI
df = df[[
    "tenure",
    "MonthlyCharges",
    "Contract",
    "PaymentMethod",
    "Churn"
]]

# Drop missing values 
df = df.dropna()

# Encode target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})


# Features & target
X = df.drop("Churn", axis=1)
y = df["Churn"]

num_cols = ["tenure", "MonthlyCharges"]
cat_cols = ["Contract", "PaymentMethod"]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="passthrough"
)

# Pipeline
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])


# Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)


# Save model
with open("churn_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model pipeline saved successfully!")

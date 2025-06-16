import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
df = pd.read_csv("Shoe prices.csv")

# Clean and preprocess
df = df.dropna(subset=["Brand", "Size", "Type", "Price (USD)"])

# Clean Size column
df["Size"] = df["Size"].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

# ✅ Clean Price column — this line is critical
df["Price (USD)"] = df["Price (USD)"].astype(str).str.replace(r"[$,]", "", regex=True).str.strip().astype(float)

# Features and target
X = df[["Brand", "Type", "Gender", "Size", "Material"]]
y = df["Price (USD)"]

# Preprocessing pipeline
numeric_features = ["Size"]
numeric_transformer = StandardScaler()

categorical_features = ["Brand", "Type", "Gender", "Material"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train and save model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
print("Test Accuracy (R²):", model.score(X_test, y_test))

joblib.dump(model, "shoe_price_model.pkl")

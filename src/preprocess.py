import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib


# ==========================================================
# 1. LOAD DATA
# ==========================================================

file_path = "data/transactions.csv"

data = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Original dataset shape:", data.shape)


# ==========================================================
# 2. CONVERT DATE INTO USEFUL FEATURES
# ==========================================================

data["date"] = pd.to_datetime(data["date"])

data["day"] = data["date"].dt.day
data["month"] = data["date"].dt.month
data["day_of_week"] = data["date"].dt.dayofweek

# Remove original date because ML model cannot directly use date text
data = data.drop(columns=["date"])


# ==========================================================
# 3. REMOVE ID COLUMNS
# ==========================================================

# IDs identify transactions/customers but do not provide
# useful numerical information for this small ML dataset.

data = data.drop(
    columns=["transaction_id", "customer_id"]
)


# ==========================================================
# 4. SEPARATE FEATURES AND TARGET
# ==========================================================

X = data.drop(columns=["is_fraud"])

y = data["is_fraud"]


print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("is_fraud")


# ==========================================================
# 5. IDENTIFY COLUMN TYPES
# ==========================================================

categorical_features = [
    "transaction_type",
    "payment_method",
    "location",
    "previous_location",
    "device_type"
]

numerical_features = [
    "hour",
    "amount",
    "device_changed",
    "is_international",
    "previous_amount",
    "transaction_frequency",
    "distance_from_previous",
    "failed_attempts",
    "account_age_days",
    "day",
    "month",
    "day_of_week"
]


# ==========================================================
# 6. NUMERICAL PIPELINE
# ==========================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ==========================================================
# 7. CATEGORICAL PIPELINE
# ==========================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ==========================================================
# 8. COMBINE PREPROCESSING
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ==========================================================
# 9. TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================================
# 10. FIT PREPROCESSOR
# ==========================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


print(
    "\nOriginal number of features:",
    X.shape[1]
)

print(
    "Processed number of features:",
    X_train_processed.shape[1]
)


# ==========================================================
# 11. CREATE REQUIRED FOLDERS
# ==========================================================

os.makedirs("models", exist_ok=True)


# ==========================================================
# 12. SAVE PREPROCESSOR
# ==========================================================

joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)


print(
    "\nPreprocessor saved successfully!"
)

print(
    "models/preprocessor.pkl"
)


# ==========================================================
# 13. SAVE PROCESSED DATA
# ==========================================================

joblib.dump(
    X_train_processed,
    "models/X_train.pkl"
)

joblib.dump(
    X_test_processed,
    "models/X_test.pkl"
)

joblib.dump(
    y_train,
    "models/y_train.pkl"
)

joblib.dump(
    y_test,
    "models/y_test.pkl"
)


print("\nPreprocessing completed successfully!")
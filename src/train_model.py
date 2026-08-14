import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# ==========================================================
# 1. LOAD DATA
# ==========================================================

data = pd.read_csv("data/transactions.csv")

print("Dataset loaded successfully!")
print("Total transactions:", len(data))


# ==========================================================
# 2. DATE FEATURE ENGINEERING
# ==========================================================

data["date"] = pd.to_datetime(data["date"])

data["day"] = data["date"].dt.day
data["month"] = data["date"].dt.month
data["day_of_week"] = data["date"].dt.dayofweek

data = data.drop(columns=["date"])


# ==========================================================
# 3. REMOVE IDENTIFICATION COLUMNS
# ==========================================================

data = data.drop(
    columns=["transaction_id", "customer_id"]
)


# ==========================================================
# 4. FEATURES AND TARGET
# ==========================================================

X = data.drop(columns=["is_fraud"])

y = data["is_fraud"]


# ==========================================================
# 5. LOAD PREPROCESSOR
# ==========================================================

preprocessor = joblib.load(
    "models/preprocessor.pkl"
)


# ==========================================================
# 6. TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining transactions:", len(X_train))
print("Testing transactions:", len(X_test))


# ==========================================================
# 7. PREPROCESS DATA
# ==========================================================

X_train_processed = preprocessor.transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


print(
    "\nProcessed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)


# ==========================================================
# 8. CREATE RANDOM FOREST MODEL
# ==========================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight="balanced"
)


# ==========================================================
# 9. TRAIN MODEL
# ==========================================================

print("\nTraining fraud detection model...")

model.fit(
    X_train_processed,
    y_train
)

print("Model training completed!")


# ==========================================================
# 10. MAKE PREDICTIONS
# ==========================================================

y_pred = model.predict(
    X_test_processed
)


# ==========================================================
# 11. MODEL EVALUATION
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========== MODEL RESULTS ==========")

print(
    "Accuracy:",
    round(accuracy, 4)
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==========================================================
# 12. SAVE MODEL
# ==========================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/fraud_model.pkl"
)

print(
    "\nModel saved successfully!"
)

print(
    "models/fraud_model.pkl"
)
import pandas as pd
import joblib
import time

# Load trained model
model = joblib.load("models/fraud_model.pkl")

# Load preprocessor
preprocessor = joblib.load("models/preprocessor.pkl")

print("======================================")
print(" REAL-TIME FRAUD DETECTION SYSTEM")
print("======================================")
print("Model loaded successfully!")
print("Starting pseudo-streaming...\n")


# Load transactions
data = pd.read_csv("data/transactions.csv")


# Process transactions one by one
for index, transaction in data.iterrows():

    # Convert current transaction into DataFrame
    transaction_df = pd.DataFrame([transaction])

    # Save transaction ID
    transaction_id = transaction["transaction_id"]

    # Save amount
    amount = transaction["amount"]

    # Remove columns that model doesn't need
    transaction_input = transaction_df.drop(
        columns=["transaction_id", "is_fraud"]
    )

    # Preprocess transaction
    processed_transaction = preprocessor.transform(
        transaction_input
    )

    # Predict
    prediction = model.predict(
        processed_transaction
    )[0]

    # Get fraud probability
    probability = model.predict_proba(
        processed_transaction
    )[0][1]

    print("--------------------------------------")
    print("Transaction ID :", transaction_id)
    print("Amount         : ₹", amount)

    if prediction == 1:
        print("Status         : 🚨 FRAUD")
    else:
        print("Status         : ✅ GENUINE")

    print(
        "Fraud Probability:",
        round(probability * 100, 2),
        "%"
    )

    print("--------------------------------------")

    # Wait 2 seconds
    time.sleep(2)


print("\n======================================")
print(" STREAMING COMPLETED")
print("======================================")
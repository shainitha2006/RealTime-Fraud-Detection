import pandas as pd

file_path = "data/transactions.csv"

data = pd.read_csv(file_path)

print("\n========== FIRST 5 ROWS ==========")
print(data.head())

print("\n========== DATASET SIZE ==========")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])

print("\n========== COLUMN NAMES ==========")
print(data.columns.tolist())

print("\n========== DATA TYPES ==========")
print(data.dtypes)

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())

print("\n========== DUPLICATES ==========")
print("Duplicate rows:", data.duplicated().sum())
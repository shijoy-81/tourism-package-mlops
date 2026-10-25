import sys
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]

df = pd.read_csv(DATA_PATH)

missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    sys.exit(f"Validation FAILED. Missing columns: {missing}")
print("Validation passed: all expected columns present.")

print(f"\nShape: {df.shape}")
print(f"\nTarget distribution (ProdTaken):\n{df['ProdTaken'].value_counts(normalize=True).round(3)}")
print(f"\nMissing values per column:\n{df.isna().sum()[df.isna().sum() > 0]}")
print(f"\nDtypes:\n{df.dtypes}")

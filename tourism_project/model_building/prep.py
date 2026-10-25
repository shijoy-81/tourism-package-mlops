import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

# 1. Load from the repository data folder
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape}")

# 2. Remove unnecessary columns (index column and identifier)
df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")

# 3. Clean inconsistent category labels
df["Gender"] = df["Gender"].str.strip().replace({"Fe Male": "Female"})
df["MaritalStatus"] = df["MaritalStatus"].str.strip().replace({"Unmarried": "Single"})
df["Occupation"] = df["Occupation"].str.strip().replace({"Free Lancer": "Freelancer"})
df["TypeofContact"] = df["TypeofContact"].str.strip()

# 4. Drop duplicates and any rows missing the target
df = df.drop_duplicates().dropna(subset=[TARGET])
print(f"After cleaning: {df.shape}")

# 5. Split features / target, then stratified train-test split
X = df.drop(columns=[TARGET])
y = df[TARGET]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Save locally (repo root, so the workflow can upload them as an artifact)
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"Train: {Xtrain.shape}, Test: {Xtest.shape}")
print(f"Train buyer rate: {ytrain.mean():.3f} | Test buyer rate: {ytest.mean():.3f}")
print("Saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")

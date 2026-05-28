import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

# ── 1. Load dataset from HF data space ──
DATASET_PATH = "hf://datasets/divyabhangre/tourism-pkg-prediction-data/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print(f"✅ Dataset loaded! Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ── 2. Data cleaning - remove unnecessary columns ──
df.drop(columns=['CustomerID'], inplace=True)
print("✅ Dropped CustomerID column!")

# ── 3. Encode categorical columns ──
label_encoder = LabelEncoder()
categorical_cols = ['TypeofContact', 'CityTier', 'Occupation',
                    'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])
print("✅ Categorical columns encoded!")

# ── 4. Split into train and test ──
target_col = 'ProdTaken'
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Split done! Train: {Xtrain.shape}, Test: {Xtest.shape}")

# ── 5. Save locally ──
Xtrain.to_csv("tourism_project/data/Xtrain.csv", index=False)
Xtest.to_csv("tourism_project/data/Xtest.csv",   index=False)
ytrain.to_csv("tourism_project/data/ytrain.csv",  index=False)
ytest.to_csv("tourism_project/data/ytest.csv",    index=False)
print("✅ Train/test files saved locally!")

# ── 6. Upload train/test back to HF ──
files = [
    "tourism_project/data/Xtrain.csv",
    "tourism_project/data/Xtest.csv",
    "tourism_project/data/ytrain.csv",
    "tourism_project/data/ytest.csv"
]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="divyabhangre/tourism-pkg-prediction-data",
        repo_type="dataset",
    )
    print(f"✅ Uploaded {file_path.split('/')[-1]} to HF!")

print("\n🎉 Data preparation complete!")

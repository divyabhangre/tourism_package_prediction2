from datasets import load_dataset
import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
from huggingface_hub import HfApi
import joblib
from itertools import product

# ── MLflow tracking ──
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("MLOps_CICD_experiment")

# ── 1. Load data from HF ──
api = HfApi(token=os.getenv("HF_TOKEN"))

train_X = load_dataset("divyabhangre/tourism-pkg-prediction-data", data_files="Xtrain.csv", split="train").to_pandas()
test_X  = load_dataset("divyabhangre/tourism-pkg-prediction-data", data_files="Xtest.csv",  split="train").to_pandas()
train_y = load_dataset("divyabhangre/tourism-pkg-prediction-data", data_files="ytrain.csv", split="train").to_pandas()
test_y  = load_dataset("divyabhangre/tourism-pkg-prediction-data", data_files="ytest.csv",  split="train").to_pandas()

print("✅ Data loaded from HF!")

# ── 2. Define ONE model and parameters ──
models = {
    "DecisionTree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": [3, 5],
            "min_samples_split": [2, 5]
        }
    }
}

# ── 3. Tune, Log, Evaluate ──
best_model      = None
best_score      = 0
best_model_name = ""

for model_name, model_info in models.items():
    model        = model_info["model"]
    params       = model_info["params"]
    param_keys   = list(params.keys())
    param_values = list(params.values())

    for combo in product(*param_values):
        param_combo = dict(zip(param_keys, combo))

        with mlflow.start_run(run_name=f"{model_name}_{param_combo}"):

            model.set_params(**param_combo)
            model.fit(train_X, train_y.values.ravel())
            predictions = model.predict(test_X)

            acc  = accuracy_score(test_y, predictions)
            prec = precision_score(test_y, predictions, average="weighted")
            rec  = recall_score(test_y, predictions, average="weighted")
            f1   = f1_score(test_y, predictions, average="weighted")

            mlflow.log_param("model_name", model_name)
            for k, v in param_combo.items():
                mlflow.log_param(k, v)
            mlflow.log_metric("accuracy",  acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall",    rec)
            mlflow.log_metric("f1_score",  f1)
            mlflow.sklearn.log_model(model, model_name)

            print(f"✅ {model_name} | Params: {param_combo} | Accuracy: {acc:.4f}")

            if acc > best_score:
                best_score      = acc
                best_model      = model
                best_model_name = model_name

print(f"\n🏆 Best Model: {best_model_name} | Accuracy: {best_score:.4f}")

# ── 4. Save and upload best model ──
os.makedirs("tourism_project/model_building/model", exist_ok=True)
model_path = "tourism_project/model_building/model/best_model.pkl"
joblib.dump(best_model, model_path)
print(f"✅ Best model saved locally!")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="best_model.pkl",
    repo_id="divyabhangre/tourism-package-predict",
    repo_type="space",
)
print(f"✅ Best model uploaded to HF Space!")

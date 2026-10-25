import os
import joblib
import pandas as pd
import mlflow
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)

# ---- 1. Load splits produced by the data-prep job ----
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

# ---- 2. Column groups ----
categorical_features = Xtrain.select_dtypes(include="object").columns.tolist()
numeric_features = [c for c in Xtrain.columns if c not in categorical_features]
print("Numeric:", numeric_features)
print("Categorical:", categorical_features)

# ---- 3. Preprocessing + model ----
preprocessor = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_features),
    (make_pipeline(SimpleImputer(strategy="most_frequent"),
                   OneHotEncoder(handle_unknown="ignore")), categorical_features),
)

# Class imbalance: weight buyers (class 1) more heavily
class_weight = (ytrain == 0).sum() / (ytrain == 1).sum()
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42,
                              eval_metric="logloss")

pipeline = make_pipeline(preprocessor, xgb_model)

param_grid = {
    "xgbclassifier__n_estimators": [50, 100, 150],
    "xgbclassifier__max_depth": [3, 4, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.5, 0.7],
    "xgbclassifier__reg_lambda": [0.5, 1.0],
}

# ---- 4. MLflow setup ----
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("tourism-package-prediction")

with mlflow.start_run(run_name="xgboost_gridsearch"):
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(Xtrain, ytrain)

    # Log every tuned parameter combination as a nested run
    results = grid.cv_results_
    for i, params in enumerate(results["params"]):
        with mlflow.start_run(run_name=f"combo_{i}", nested=True):
            mlflow.log_params({k.replace("xgbclassifier__", ""): v for k, v in params.items()})
            mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])
            mlflow.log_metric("std_cv_f1", results["std_test_score"][i])

    # Best model
    best_model = grid.best_estimator_
    mlflow.log_params({f"best_{k.replace('xgbclassifier__', '')}": v
                       for k, v in grid.best_params_.items()})
    mlflow.log_metric("best_cv_f1", grid.best_score_)

    # ---- 5. Evaluate ----
    for name, X, y in [("train", Xtrain, ytrain), ("test", Xtest, ytest)]:
        pred = best_model.predict(X)
        proba = best_model.predict_proba(X)[:, 1]
        mlflow.log_metrics({
            f"{name}_accuracy": accuracy_score(y, pred),
            f"{name}_precision": precision_score(y, pred),
            f"{name}_recall": recall_score(y, pred),
            f"{name}_f1": f1_score(y, pred),
            f"{name}_roc_auc": roc_auc_score(y, proba),
        })

    print("Best params:", grid.best_params_)
    print("\nTest set report:\n", classification_report(ytest, best_model.predict(Xtest)))

    # ---- 6. Save the model into the deployment folder ----
    os.makedirs("tourism_project/deployment", exist_ok=True)
    model_path = "tourism_project/deployment/best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path)
    print(f"Model saved to {model_path}")

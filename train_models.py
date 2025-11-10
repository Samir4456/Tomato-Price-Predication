import os
import warnings
warnings.filterwarnings("ignore")
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_validate, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

sns.set(style="whitegrid")
np.random.seed(42)

# ------------------ Paths & config ------------------
folder_tag = "_lag"
INPUT_CSV = "artifacts/data/clean_data.csv"
TARGET = "Average_Price"

model_plot_path = f"artifacts/model_plots{folder_tag}/"
model_path = f"artifacts/models{folder_tag}/"
model_results_path = f"artifacts/model_results{folder_tag}/"
preprocessor_path = f"artifacts/preprocessor/preprocessor_lag_fixed.pkl"

os.makedirs(model_plot_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)
os.makedirs(model_results_path, exist_ok=True)
os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

RUN_MODELS = ["Linear", "RandomForest", "XGBoost", "LightGBM"]
N_SPLITS = 5
RANDOM_SEARCH_ITER = 20
GRID_SEARCH_SMALL = True

# ------------------ Metrics ------------------
def regression_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MSE": mse, "MAE": mae, "MAPE": mape, "R2": r2}

sk_rmse = make_scorer(lambda y, yhat: -np.sqrt(mean_squared_error(y, yhat)))
sk_mse = make_scorer(lambda y, yhat: -mean_squared_error(y, yhat))
sk_mae = make_scorer(lambda y, yhat: -mean_absolute_error(y, yhat))
def sk_mape(y, yhat):
    mask = y != 0
    return -np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100 if mask.sum() > 0 else 0
sk_mape_scorer = make_scorer(sk_mape)
SCORING = {"neg_rmse": sk_rmse, "neg_mse": sk_mse, "neg_mae": sk_mae, "neg_mape": sk_mape_scorer, "r2": "r2"}

# ------------------ Load data ------------------
df = pd.read_csv(INPUT_CSV, parse_dates=["Date"], infer_datetime_format=True)
df = df.dropna(subset=[TARGET]).reset_index(drop=True)
if "Date" in df.columns:
    df = df.sort_values("Date").reset_index(drop=True)

bool_cols = df.select_dtypes(include=["bool"]).columns
df[bool_cols] = df[bool_cols].astype(int)

cols_all = [c for c in df.columns if c not in [TARGET, "Date"]]
numeric_cols = df[cols_all].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in cols_all if c not in numeric_cols]

# ------------------ Leakage-Free Lag & Roll Transformer ------------------
class LagRollTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lag_config=None, roll_config=None, use_std=False):
        self.lag_config = lag_config or {}
        self.roll_config = roll_config or {}
        self.use_std = use_std

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()
        
        # Lags
        for col, lags in self.lag_config.items():
            if col not in X_new.columns:
                raise KeyError(f"{col} not found for lagging")
            for lag in lags:
                X_new[f"{col}_lag{lag}"] = X_new[col].shift(lag)
        
        # Rolling stats (shifted by 1 to avoid leakage)
        for col, windows in self.roll_config.items():
            if col not in X_new.columns:
                raise KeyError(f"{col} not found for rolling")
            for w in windows:
                X_new[f"{col}_rollmean_{w}"] = X_new[col].shift(1).rolling(window=w, min_periods=1).mean()
                if self.use_std:
                    X_new[f"{col}_rollstd_{w}"] = X_new[col].shift(1).rolling(window=w, min_periods=1).std()
        
        return X_new

# ------------------ Lag/Roll configuration ------------------
custom_lag_config = {
    "Average_Price": [1, 3, 7],
    "Sarlahi_Temperature": [1,3,7],
    "Sarlahi_Rainfall_MM": [1,3,7],
    "Hilly_Temperature": [1,3,7],
    "Hilly_Rainfall_MM": [1,3,7],
    "Kathmandu_Rainfall_MM": [1,2]
}

custom_roll_config = {
    "Average_Price": [1,7],
    "Supply_Volume": [3]
}

lag_transformer = LagRollTransformer(
    lag_config=custom_lag_config,
    roll_config=custom_roll_config,
    use_std=True
)

# ------------------ Apply transformer ------------------
X_lagged = lag_transformer.fit_transform(df[cols_all + [TARGET]])

# Drop initial rows with NaNs caused by lag/roll
max_lag = max([max(v) for v in custom_lag_config.values()])
max_roll = max([max(v) for v in custom_roll_config.values()])
drop_rows = max(max_lag, max_roll)

X_lagged = X_lagged.iloc[drop_rows:].reset_index(drop=True)
y_aligned = df[TARGET].iloc[drop_rows:].reset_index(drop=True)

# Drop original target from features
X_lagged = X_lagged.drop(columns=[TARGET])

# ------------------ Preprocessor ------------------
def build_and_save_preprocessor(df, cols_all, save_path=preprocessor_path):
    numeric_cols = df[cols_all].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in cols_all if c not in numeric_cols]
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, cat_cols)
    ], remainder="drop", sparse_threshold=0)
    preprocessor.fit(df[cols_all])
    joblib.dump(preprocessor, save_path)
    joblib.dump({
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "all_cols": cols_all
    }, os.path.join(os.path.dirname(save_path), "feature_schema.pkl"))
    print(f"Preprocessor saved to {save_path}")
    return preprocessor

preprocessor = build_and_save_preprocessor(pd.concat([X_lagged, y_aligned], axis=1), X_lagged.columns.tolist())

# ------------------ Pipelines ------------------
def make_pipeline_with_lag(estimator):
    return Pipeline([("preproc", preprocessor), ("est", estimator)])

models = {}
param_grids = {}

models["Linear"] = make_pipeline_with_lag(LinearRegression())
param_grids["Linear"] = {}

models["RandomForest"] = make_pipeline_with_lag(RandomForestRegressor(random_state=42))
param_grids["RandomForest"] = {
    "est__n_estimators": [100, 200],
    "est__max_depth": [5, 10, None],
    "est__min_samples_leaf": [1,2,4]
}

if XGBRegressor is not None:
    models["XGBoost"] = make_pipeline_with_lag(XGBRegressor(objective="reg:squarederror", random_state=42))
    param_grids["XGBoost"] = {
        "est__n_estimators": [100,200],
        "est__max_depth": [3,5],
        "est__learning_rate": [0.01,0.05,0.1]
    }

if LGBMRegressor is not None:
    models["LightGBM"] = make_pipeline_with_lag(LGBMRegressor(random_state=42))
    param_grids["LightGBM"] = {
        "est__n_estimators": [100,200],
        "est__num_leaves": [31,64],
        "est__learning_rate": [0.01,0.05]
    }

# ------------------ Train/Test split ------------------
holdout_frac = 0.2
n_holdout = int(len(X_lagged) * holdout_frac)
train_idx = slice(0, len(X_lagged)-n_holdout)
test_idx = slice(len(X_lagged)-n_holdout, len(X_lagged))

X_train_final = X_lagged.iloc[train_idx].reset_index(drop=True)
y_train_final = y_aligned.iloc[train_idx].reset_index(drop=True)
X_test_final = X_lagged.iloc[test_idx].reset_index(drop=True)
y_test_final = y_aligned.iloc[test_idx].reset_index(drop=True)

pd.concat([X_train_final, y_train_final], axis=1).to_csv(f"artifacts/data/train_data{folder_tag}.csv", index=False)
pd.concat([X_test_final, y_test_final], axis=1).to_csv(f"artifacts/data/test_data{folder_tag}.csv", index=False)

# ------------------ Model training ------------------
tscv = TimeSeriesSplit(n_splits=N_SPLITS)
best_models = {}
cv_summary = []

for name, pipeline in models.items():
    if name not in RUN_MODELS:
        continue
    print(f"\nTraining model: {name}")
    grid = param_grids.get(name, None)
    
    if not grid:
        cv_res = cross_validate(pipeline, X_lagged, y_aligned, cv=tscv, scoring=SCORING, return_train_score=True)
        results = {
            "Model": name,
            "RMSE_mean": -np.mean(cv_res["test_neg_rmse"]) if "test_neg_rmse" in cv_res else np.nan,
            "MSE_mean": -np.mean(cv_res["test_neg_mse"]) if "test_neg_mse" in cv_res else np.nan,
            "MAE_mean": -np.mean(cv_res["test_neg_mae"]) if "test_neg_mae" in cv_res else np.nan,
            "MAPE_mean": -np.mean(cv_res["test_neg_mape"]) if "test_neg_mape" in cv_res else np.nan,
            "R2_mean": np.mean(cv_res["test_r2"]) if "test_r2" in cv_res else np.nan
        }
        cv_summary.append(results)
        fitted = pipeline.fit(X_lagged, y_aligned)
        best_models[name] = fitted
        joblib.dump(fitted, f"{model_path}{name}_best.joblib")
        continue
    
    rnd = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=grid,
        n_iter=RANDOM_SEARCH_ITER,
        cv=tscv,
        scoring="neg_mean_squared_error",
        random_state=42,
        verbose=1
    )
    rnd.fit(X_lagged, y_aligned)
    
    best_est = rnd.best_estimator_
    if GRID_SEARCH_SMALL:
        small_grid = {}
        for k, v in grid.items():
            small_grid[k] = v if len(v) <= 3 else v[:3]
        gscv = GridSearchCV(best_est, param_grid=small_grid, cv=tscv, scoring="neg_mean_squared_error", verbose=1)
        gscv.fit(X_lagged, y_aligned)
        best_est = gscv.best_estimator_
    
    best_models[name] = best_est
    joblib.dump(best_est, f"{model_path}{name}_best.joblib")
    
    cv_res = cross_validate(best_est, X_lagged, y_aligned, cv=tscv, scoring=SCORING, return_train_score=False)
    results = {
        "Model": name,
        "RMSE_mean": -np.mean(cv_res["test_neg_rmse"]),
        "MSE_mean": -np.mean(cv_res["test_neg_mse"]),
        "MAE_mean": -np.mean(cv_res["test_neg_mae"]),
        "MAPE_mean": -np.mean(cv_res["test_neg_mape"]),
        "R2_mean": np.mean(cv_res["test_r2"])
    }
    cv_summary.append(results)

# ------------------ Evaluate on holdout ------------------
eval_records = []
for name, model in best_models.items():
    print(f"Evaluating on holdout: {name}")
    y_pred = model.predict(X_test_final)
    metrics = regression_metrics(y_test_final, y_pred)
    metrics["Model"] = name
    eval_records.append(metrics)
    
    fig, ax = plt.subplots(figsize=(10,3))
    if "Date" in df.columns:
        test_dates = df["Date"].iloc[test_idx].reset_index(drop=True)
        ax.plot(test_dates, y_test_final, label="Actual")
        ax.plot(test_dates, y_pred, linestyle="--", label="Predicted")
        fig.autofmt_xdate()
    else:
        ax.plot(y_test_final.index, y_test_final, label="Actual")
        ax.plot(y_test_final.index, y_pred, linestyle="--", label="Predicted")
    ax.set_title(f"{name} — Actual vs Predicted")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{model_plot_path}actual_vs_pred_{name}.png")
    plt.close(fig)

eval_df = pd.DataFrame(eval_records).sort_values("RMSE")
eval_df.to_csv(f"{model_results_path}holdout_performance.csv", index=False)
cv_df = pd.DataFrame(cv_summary)
cv_df.to_csv(f"{model_results_path}cv_summary.csv", index=False)

# ------------------ Feature importances ------------------
if "RandomForest" in best_models:
    rf = best_models["RandomForest"]
    pre = rf.named_steps["preproc"]
    try:
        ohe_names = list(pre.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols))
    except:
        ohe_names = []
    feat_names = numeric_cols + ohe_names
    rf_est = rf.named_steps["est"]
    importances = getattr(rf_est, "feature_importances_", None)
    if importances is not None:
        imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False).head(30)
        imp_df.to_csv(f"{model_results_path}rf_feature_importances.csv", index=False)
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(x="importance", y="feature", data=imp_df, ax=ax)
        ax.set_title("RandomForest: Top feature importances")
        fig.savefig(f"{model_plot_path}rf_top_feature_importances.png")
        plt.close(fig)

# ------------------ Manifest ------------------
manifest = {
    "timestamp": datetime.utcnow().isoformat(),
    "input_csv": INPUT_CSV,
    "target": TARGET,
    "features_used": list(X_lagged.columns),
    "numeric_cols": numeric_cols,
    "cat_cols": cat_cols,
    "models_trained": list(best_models.keys()),
    "holdout_rows": n_holdout
}
with open(f"{model_results_path}manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("\nAll done. Models, plots, and artifacts saved.")

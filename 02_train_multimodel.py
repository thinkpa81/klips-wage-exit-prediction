from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


PROJECT_DIR = Path(r"G:\내 드라이브\project")
OUTPUT_DIR = PROJECT_DIR / "outputs_klips_sr"
PROCESSED_DIR = OUTPUT_DIR / "processed"
LOG_DIR = OUTPUT_DIR / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "klips_stage2_multimodel.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.1) -> float:
    n = len(y_true)
    top_n = max(1, int(np.ceil(n * k)))
    idx = np.argsort(-y_prob)[:top_n]
    positives_total = y_true.sum()
    if positives_total == 0:
        return np.nan
    return float(y_true[idx].sum() / positives_total)


def lift_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.1) -> float:
    base_rate = y_true.mean()
    if base_rate == 0:
        return np.nan
    n = len(y_true)
    top_n = max(1, int(np.ceil(n * k)))
    idx = np.argsort(-y_prob)[:top_n]
    precision_top = y_true[idx].mean()
    return float(precision_top / base_rate)


def evaluate_binary_classifier(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
        "recall_at_10": recall_at_k(y_true, y_prob, 0.10),
        "lift_at_10": lift_at_k(y_true, y_prob, 0.10),
        "recall_at_20": recall_at_k(y_true, y_prob, 0.20),
        "lift_at_20": lift_at_k(y_true, y_prob, 0.20),
    }


def timewise_split(df: pd.DataFrame, train_end: int, valid_end: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["wave"] <= train_end].copy()
    valid = df[(df["wave"] > train_end) & (df["wave"] <= valid_end)].copy()
    test = df[df["wave"] > valid_end].copy()
    return train, valid, test


def select_model_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    candidate_features = [
        "gender",
        "age_final",
        "education_level",
        "marital_status",
        "region",
        "industry_major",
        "occupation_major",
        "tenure_years",
        "weekly_hours",
        "weekly_hours_missing",
        "gt48",
        "gt52",
        "ge55",
        "monthly_wage",
        "monthly_wage_missing",
        "log_monthly_wage",
        "one_person_firm_flag",
        "household_size_raw",
        "housing_tenure_type",
        "housing_type",
        "housing_cost_burden",
    ]
    candidate_features = [c for c in candidate_features if c in df.columns]

    numeric_features = [
        c for c in candidate_features
        if c in {
            "age_final",
            "tenure_years",
            "weekly_hours",
            "weekly_hours_missing",
            "gt48",
            "gt52",
            "ge55",
            "monthly_wage",
            "monthly_wage_missing",
            "log_monthly_wage",
            "one_person_firm_flag",
            "household_size_raw",
            "housing_cost_burden",
        }
    ]
    categorical_features = [c for c in candidate_features if c not in numeric_features]
    return candidate_features, numeric_features, categorical_features


def sanitize_model_input(X: pd.DataFrame, numeric_features: List[str], categorical_features: List[str]) -> pd.DataFrame:
    X = X.copy().replace({pd.NA: np.nan})

    for col in numeric_features:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].astype(object)
            X.loc[pd.isna(X[col]), col] = np.nan

    return X


def make_sklearn_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def fit_sklearn_model(
    model_name: str,
    estimator,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    numeric_features: List[str],
    categorical_features: List[str],
    target_col: str = "exit_label_t1",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train = sanitize_model_input(train_df[feature_cols], numeric_features, categorical_features)
    y_train = train_df[target_col].astype(int)

    X_valid = sanitize_model_input(valid_df[feature_cols], numeric_features, categorical_features)
    y_valid = valid_df[target_col].astype(int)

    X_test = sanitize_model_input(test_df[feature_cols], numeric_features, categorical_features)
    y_test = test_df[target_col].astype(int)

    preprocessor = make_sklearn_preprocessor(numeric_features, categorical_features)

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
    clf.fit(X_train, y_train)

    valid_prob = clf.predict_proba(X_valid)[:, 1]
    test_prob = clf.predict_proba(X_test)[:, 1]

    metric_rows = []
    for split_name, y_true, y_prob in [
        ("valid", y_valid.to_numpy(), valid_prob),
        ("test", y_test.to_numpy(), test_prob),
    ]:
        row = {"model": model_name, "split": split_name}
        row.update(evaluate_binary_classifier(y_true, y_prob))
        metric_rows.append(row)

    pred_cols = [c for c in ["pid", "wave", target_col] if c in test_df.columns]
    pred_df = test_df[pred_cols].copy() if pred_cols else pd.DataFrame(index=test_df.index)
    pred_df[f"proba_{model_name}"] = test_prob

    return pd.DataFrame(metric_rows), pred_df


def fit_catboost_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    numeric_features: List[str],
    categorical_features: List[str],
    target_col: str = "exit_label_t1",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train = train_df[feature_cols].copy().replace({pd.NA: np.nan})
    X_valid = valid_df[feature_cols].copy().replace({pd.NA: np.nan})
    X_test = test_df[feature_cols].copy().replace({pd.NA: np.nan})

    y_train = train_df[target_col].astype(int)
    y_valid = valid_df[target_col].astype(int)
    y_test = test_df[target_col].astype(int)

    for col in numeric_features:
        if col in X_train.columns:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_valid[col] = pd.to_numeric(X_valid[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    # 핵심 수정: categorical을 모두 문자열화
    for col in categorical_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str).fillna("__MISSING__")
            X_valid[col] = X_valid[col].astype(str).fillna("__MISSING__")
            X_test[col] = X_test[col].astype(str).fillna("__MISSING__")

    cat_features_idx = [X_train.columns.get_loc(c) for c in categorical_features if c in X_train.columns]

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        random_seed=42,
    )

    model.fit(X_train, y_train, cat_features=cat_features_idx)

    valid_prob = model.predict_proba(X_valid)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    metric_rows = []
    for split_name, y_true, y_prob in [
        ("valid", y_valid.to_numpy(), valid_prob),
        ("test", y_test.to_numpy(), test_prob),
    ]:
        row = {"model": "catboost", "split": split_name}
        row.update(evaluate_binary_classifier(y_true, y_prob))
        metric_rows.append(row)

    pred_cols = [c for c in ["pid", "wave", target_col] if c in test_df.columns]
    pred_df = test_df[pred_cols].copy() if pred_cols else pd.DataFrame(index=test_df.index)
    pred_df["proba_catboost"] = test_prob

    return pd.DataFrame(metric_rows), pred_df


def main() -> None:
    logger.info("==== Stage2 multi-model start ====")

    input_path = PROCESSED_DIR / "analysis_base_with_label.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path)
    logger.info("Loaded analysis base: %s", df.shape)

    train_df, valid_df, test_df = timewise_split(df, train_end=20, valid_end=23)
    logger.info("Train=%s, Valid=%s, Test=%s", train_df.shape, valid_df.shape, test_df.shape)

    feature_cols, numeric_features, categorical_features = select_model_columns(df)
    logger.info("Feature count=%s", len(feature_cols))
    logger.info("Numeric=%s", numeric_features)
    logger.info("Categorical=%s", categorical_features)

    metrics_all = []
    preds_all = []

    # Logistic
    logit_metrics, logit_preds = fit_sklearn_model(
        model_name="logistic",
        estimator=LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs"),
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        feature_cols=feature_cols,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    metrics_all.append(logit_metrics)
    preds_all.append(logit_preds)

    # Random Forest
    rf_metrics, rf_preds = fit_sklearn_model(
        model_name="random_forest",
        estimator=RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        feature_cols=feature_cols,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    metrics_all.append(rf_metrics)
    preds_all.append(rf_preds)

    # XGBoost
    if HAS_XGB:
        xgb_metrics, xgb_preds = fit_sklearn_model(
            model_name="xgboost",
            estimator=XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="auc",
                random_state=42,
                n_jobs=-1,
            ),
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            feature_cols=feature_cols,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        metrics_all.append(xgb_metrics)
        preds_all.append(xgb_preds)
    else:
        logger.warning("xgboost is not installed. Skipping XGBoost.")

    # CatBoost
    if HAS_CATBOOST:
        cb_metrics, cb_preds = fit_catboost_model(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            feature_cols=feature_cols,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        metrics_all.append(cb_metrics)
        preds_all.append(cb_preds)
    else:
        logger.warning("catboost is not installed. Skipping CatBoost.")

    metrics_df = pd.concat(metrics_all, ignore_index=True)
    metrics_df.to_csv(OUTPUT_DIR / "stage2_model_metrics.csv", index=False, encoding="utf-8-sig")

    pred_base = preds_all[0].copy()
    for add_df in preds_all[1:]:
        merge_cols = [c for c in ["pid", "wave", "exit_label_t1"] if c in pred_base.columns and c in add_df.columns]
        score_cols = [c for c in add_df.columns if c.startswith("proba_")]
        pred_base = pred_base.merge(add_df[merge_cols + score_cols], on=merge_cols, how="left")

    pred_base.to_csv(OUTPUT_DIR / "stage2_test_predictions_all_models.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input_path": str(input_path),
        "train_shape": list(train_df.shape),
        "valid_shape": list(valid_df.shape),
        "test_shape": list(test_df.shape),
        "feature_count": len(feature_cols),
        "models_run": metrics_df["model"].drop_duplicates().tolist(),
    }
    with open(OUTPUT_DIR / "stage2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("==== Stage2 multi-model end ====")
    logger.info("\n%s", metrics_df)


if __name__ == "__main__":
    main()
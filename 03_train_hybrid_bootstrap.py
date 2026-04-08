from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
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
        logging.FileHandler(LOG_DIR / "klips_stage3_hybrid.log", encoding="utf-8"),
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


def fit_sklearn_pipeline(estimator, X_train, y_train, numeric_features, categorical_features):
    preprocessor = make_sklearn_preprocessor(numeric_features, categorical_features)
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
    clf.fit(X_train, y_train)
    return clf


def fit_catboost_pipeline(X_train, y_train, numeric_features, categorical_features):
    if not HAS_CATBOOST:
        raise RuntimeError("catboost not installed")

    X_train = X_train.copy().replace({pd.NA: np.nan})

    for col in numeric_features:
        if col in X_train.columns:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")

    for col in categorical_features:
        if col in X_train.columns:
            # CatBoost categorical은 문자열로 강제
            X_train[col] = X_train[col].astype(str)
            X_train.loc[pd.isna(X_train[col]), col] = "__MISSING__"

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
    return model


def predict_catboost(model, X, numeric_features, categorical_features):
    X = X.copy().replace({pd.NA: np.nan})

    for col in numeric_features:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].astype(str)
            X.loc[pd.isna(X[col]), col] = "__MISSING__"

    return model.predict_proba(X)[:, 1]


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_fn, n_boot: int = 300, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        p_b = y_prob[idx]

        if len(np.unique(y_b)) < 2:
            continue

        scores.append(metric_fn(y_b, p_b))

    if len(scores) == 0:
        return np.nan, np.nan, np.nan

    scores = np.array(scores)
    return float(np.mean(scores)), float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))


def save_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, out_dir: Path, n_bins: int = 10) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    cal_df = pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true})
    cal_df.to_csv(out_dir / f"calibration_{model_name}.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(prob_pred, prob_true, marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration curve - {model_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"calibration_{model_name}.png", dpi=200)
    plt.close()


def main() -> None:
    logger.info("==== Stage3 hybrid start ====")

    input_path = PROCESSED_DIR / "analysis_base_with_label.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path)
    logger.info("Loaded analysis base: %s", df.shape)

    train_df, valid_df, test_df = timewise_split(df, train_end=20, valid_end=23)
    logger.info("Train=%s, Valid=%s, Test=%s", train_df.shape, valid_df.shape, test_df.shape)

    feature_cols, numeric_features, categorical_features = select_model_columns(df)
    logger.info("Feature count=%s", len(feature_cols))

    X_train = sanitize_model_input(train_df[feature_cols], numeric_features, categorical_features)
    y_train = train_df["exit_label_t1"].astype(int).to_numpy()

    X_valid = sanitize_model_input(valid_df[feature_cols], numeric_features, categorical_features)
    y_valid = valid_df["exit_label_t1"].astype(int).to_numpy()

    X_test = sanitize_model_input(test_df[feature_cols], numeric_features, categorical_features)
    y_test = test_df["exit_label_t1"].astype(int).to_numpy()

    base_models = {
        "logistic": ("sklearn", LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs")),
        "random_forest": ("sklearn", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )),
    }

    if HAS_XGB:
        base_models["xgboost"] = ("sklearn", XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        ))

    if HAS_CATBOOST:
        base_models["catboost"] = ("catboost", None)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_train = pd.DataFrame(index=train_df.index)
    valid_meta = pd.DataFrame(index=valid_df.index)
    test_meta = pd.DataFrame(index=test_df.index)

    metrics_rows = []

    for model_name, (model_type, estimator) in base_models.items():
        logger.info("Training base model: %s", model_name)

        meta_col = f"meta_{model_name}"   # 핵심: train/valid/test 모두 같은 컬럼명 사용
        oof_pred = np.zeros(len(train_df))
        valid_fold_preds = []
        test_fold_preds = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
            X_tr = X_train.iloc[tr_idx].copy()
            y_tr = y_train[tr_idx]
            X_va = X_train.iloc[va_idx].copy()

            if model_type == "sklearn":
                model = fit_sklearn_pipeline(estimator, X_tr, y_tr, numeric_features, categorical_features)
                oof_pred[va_idx] = model.predict_proba(X_va)[:, 1]
                valid_fold_preds.append(model.predict_proba(X_valid)[:, 1])
                test_fold_preds.append(model.predict_proba(X_test)[:, 1])

            elif model_type == "catboost":
                model = fit_catboost_pipeline(X_tr, y_tr, numeric_features, categorical_features)
                oof_pred[va_idx] = predict_catboost(model, X_va, numeric_features, categorical_features)
                valid_fold_preds.append(predict_catboost(model, X_valid, numeric_features, categorical_features))
                test_fold_preds.append(predict_catboost(model, X_test, numeric_features, categorical_features))

        valid_pred = np.mean(np.column_stack(valid_fold_preds), axis=1)
        test_pred = np.mean(np.column_stack(test_fold_preds), axis=1)

        oof_train[meta_col] = oof_pred
        valid_meta[meta_col] = valid_pred
        test_meta[meta_col] = test_pred

        for split_name, y_true, y_prob in [
            ("valid", y_valid, valid_pred),
            ("test", y_test, test_pred),
        ]:
            row = {"model": model_name, "split": split_name}
            row.update(evaluate_binary_classifier(y_true, y_prob))
            metrics_rows.append(row)

    # Hybrid meta model
    meta_features_train = oof_train.copy()
    meta_features_valid = valid_meta.copy()
    meta_features_test = test_meta.copy()

    # 컬럼명/순서 강제 정렬
    meta_feature_order = meta_features_train.columns.tolist()
    meta_features_valid = meta_features_valid[meta_feature_order]
    meta_features_test = meta_features_test[meta_feature_order]

    meta_model = LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs")
    meta_model.fit(meta_features_train, y_train)

    hybrid_valid_prob = meta_model.predict_proba(meta_features_valid)[:, 1]
    hybrid_test_prob = meta_model.predict_proba(meta_features_test)[:, 1]

    for split_name, y_true, y_prob in [
        ("valid", y_valid, hybrid_valid_prob),
        ("test", y_test, hybrid_test_prob),
    ]:
        row = {"model": "hybrid_stack", "split": split_name}
        row.update(evaluate_binary_classifier(y_true, y_prob))
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUTPUT_DIR / "stage3_hybrid_metrics.csv", index=False, encoding="utf-8-sig")

    # Bootstrap CI on test
    ci_rows = []
    test_pred_map = {col.replace("meta_", ""): test_meta[col].to_numpy() for col in test_meta.columns}
    test_pred_map["hybrid_stack"] = hybrid_test_prob

    for model_name, prob in test_pred_map.items():
        for metric_name, metric_fn in [
            ("roc_auc", lambda y, p: roc_auc_score(y, p)),
            ("pr_auc", lambda y, p: average_precision_score(y, p)),
            ("brier", lambda y, p: brier_score_loss(y, p)),
        ]:
            mean_val, ci_low, ci_high = bootstrap_ci(y_test, prob, metric_fn, n_boot=300, seed=42)
            ci_rows.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    "bootstrap_mean": mean_val,
                    "ci_2.5": ci_low,
                    "ci_97.5": ci_high,
                }
            )

    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(OUTPUT_DIR / "stage3_bootstrap_ci.csv", index=False, encoding="utf-8-sig")

    # Calibration
    for model_name, prob in test_pred_map.items():
        save_calibration_curve(y_test, prob, model_name, OUTPUT_DIR, n_bins=10)

    # Prediction outputs
    pred_out = test_df[[c for c in ["pid", "wave", "exit_label_t1"] if c in test_df.columns]].copy()
    for model_name, prob in test_pred_map.items():
        pred_out[f"proba_{model_name}"] = prob
    pred_out.to_csv(OUTPUT_DIR / "stage3_test_predictions_with_hybrid.csv", index=False, encoding="utf-8-sig")

    # Meta coefficients
    coef_df = pd.DataFrame(
        {
            "meta_feature": meta_feature_order,
            "coefficient": meta_model.coef_[0],
        }
    )
    coef_df.to_csv(OUTPUT_DIR / "stage3_hybrid_meta_coefficients.csv", index=False, encoding="utf-8-sig")

    summary = {
        "train_shape": list(train_df.shape),
        "valid_shape": list(valid_df.shape),
        "test_shape": list(test_df.shape),
        "base_models": list(base_models.keys()),
        "hybrid_features": meta_feature_order,
    }
    with open(OUTPUT_DIR / "stage3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("==== Stage3 hybrid end ====")
    logger.info("\n%s", metrics_df)


if __name__ == "__main__":
    main()
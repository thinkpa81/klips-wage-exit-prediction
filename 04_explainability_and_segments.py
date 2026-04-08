from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


PROJECT_DIR = Path(r"G:\내 드라이브\project")
OUTPUT_DIR = PROJECT_DIR / "outputs_klips_sr"
PROCESSED_DIR = OUTPUT_DIR / "processed"
LOG_DIR = OUTPUT_DIR / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "klips_stage4_explainability_segment.log", encoding="utf-8"),
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


def prepare_catboost_input(X: pd.DataFrame, numeric_features: List[str], categorical_features: List[str]) -> pd.DataFrame:
    X = X.copy().replace({pd.NA: np.nan})

    for col in numeric_features:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].astype(object)
            X.loc[pd.isna(X[col]), col] = "__MISSING__"
            X[col] = X[col].astype(str)

    return X


def add_segment_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "age_final" in out.columns:
        out["age_group"] = pd.cut(
            out["age_final"],
            bins=[15, 24, 54, 64, np.inf],
            labels=["15-24", "25-54", "55-64", "65+"],
            right=True,
            include_lowest=True,
        )

    if "weekly_hours" in out.columns:
        out["weekly_hours_group"] = pd.cut(
            out["weekly_hours"],
            bins=[0, 19, 29, 34, 39, np.inf],
            labels=["1-19", "20-29", "30-34", "35-39", "40+"],
            right=True,
        )

    if "one_person_firm_flag" in out.columns:
        out["firm_size_group"] = np.where(out["one_person_firm_flag"] == 1, "1-person", "2+-person")

    return out


def evaluate_by_group(df: pd.DataFrame, prob_col: str, group_col: str, model_name: str) -> pd.DataFrame:
    rows = []
    if group_col not in df.columns:
        return pd.DataFrame()

    for group_value, sub in df.groupby(group_col, dropna=False):
        if len(sub) < 100:
            continue
        y_true = sub["exit_label_t1"].to_numpy()
        y_prob = sub[prob_col].to_numpy()

        if len(np.unique(y_true)) < 2:
            continue

        row = {
            "model": model_name,
            "group_col": group_col,
            "group_value": str(group_value),
            "n": len(sub),
            "event_rate": float(np.mean(y_true)),
        }
        row.update(evaluate_binary_classifier(y_true, y_prob))
        rows.append(row)

    return pd.DataFrame(rows)


def save_categorical_shap_details(
    X_shap: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: List[str],
    target_features: List[str],
    output_dir: Path,
) -> List[str]:
    """Save category-level mean SHAP details for selected categorical features.

    Important: these are mean SHAP contributions of the *feature* conditional on each
    observed category in the SHAP sample, not stand-alone causal effects of categories.
    """
    saved_files: List[str] = []
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    for feature in target_features:
        if feature not in X_shap.columns or feature not in shap_df.columns:
            continue

        detail_df = pd.DataFrame({
            "category": X_shap[feature].astype(str).values,
            "shap_value": pd.to_numeric(shap_df[feature], errors="coerce"),
        })
        detail_df = detail_df.dropna(subset=["shap_value"])
        if detail_df.empty:
            continue

        summary_df = (
            detail_df.groupby("category", dropna=False)
            .agg(
                n=("shap_value", "size"),
                mean_shap=("shap_value", "mean"),
                median_shap=("shap_value", "median"),
                mean_abs_shap=("shap_value", lambda s: np.abs(s).mean()),
            )
            .reset_index()
        )
        summary_df["direction"] = np.where(summary_df["mean_shap"] > 0, "higher_exit_risk", "lower_exit_risk")
        summary_df = summary_df.sort_values(["mean_shap", "mean_abs_shap", "n"], ascending=[False, False, False])

        csv_path = output_dir / f"stage4_{feature}_shap_details.csv"
        xlsx_path = output_dir / f"stage4_{feature}_shap_details.xlsx"
        summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, index=False, sheet_name="Sheet1")

        saved_files.extend([csv_path.name, xlsx_path.name])
        logger.info("Saved categorical SHAP detail file: %s", csv_path.name)

    return saved_files


def main() -> None:
    logger.info("==== Stage4 explainability + subgroup start ====")

    analysis_path = PROCESSED_DIR / "analysis_base_with_label.csv"
    pred_path = OUTPUT_DIR / "stage3_test_predictions_with_hybrid.csv"

    if not analysis_path.exists():
        raise FileNotFoundError(f"Missing file: {analysis_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing file: {pred_path}")

    df = pd.read_csv(analysis_path)
    pred_df = pd.read_csv(pred_path)

    logger.info("Analysis base shape=%s", df.shape)
    logger.info("Prediction file shape=%s", pred_df.shape)

    train_df, valid_df, test_df = timewise_split(df, train_end=20, valid_end=23)
    logger.info("Train=%s, Valid=%s, Test=%s", train_df.shape, valid_df.shape, test_df.shape)

    feature_cols, numeric_features, categorical_features = select_model_columns(df)
    shap_detail_files: List[str] = []

    if not HAS_CATBOOST:
        logger.warning("catboost not installed. Skip CatBoost SHAP/robustness.")
    else:
        X_train = prepare_catboost_input(train_df[feature_cols], numeric_features, categorical_features)
        y_train = train_df["exit_label_t1"].astype(int)

        X_test = prepare_catboost_input(test_df[feature_cols], numeric_features, categorical_features)
        y_test = test_df["exit_label_t1"].astype(int)

        cat_features_idx = [X_train.columns.get_loc(c) for c in categorical_features if c in X_train.columns]

        cb_model = CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            random_seed=42,
        )
        cb_model.fit(X_train, y_train, cat_features=cat_features_idx)

        cb_test_prob = cb_model.predict_proba(X_test)[:, 1]
        cb_metrics = evaluate_binary_classifier(y_test.to_numpy(), cb_test_prob)
        pd.DataFrame([{"model": "catboost_retrain_test", **cb_metrics}]).to_csv(
            OUTPUT_DIR / "stage4_catboost_retrain_test_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if HAS_SHAP:
            shap_sample_n = min(5000, len(X_test))
            X_shap = X_test.sample(n=shap_sample_n, random_state=42).copy()

            explainer = shap.TreeExplainer(cb_model)
            shap_values = explainer.shap_values(X_shap)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_imp = pd.DataFrame({
                "feature": X_shap.columns,
                "mean_abs_shap": mean_abs_shap,
            }).sort_values("mean_abs_shap", ascending=False)
            shap_imp.to_csv(
                OUTPUT_DIR / "stage4_catboost_shap_importance.csv",
                index=False,
                encoding="utf-8-sig",
            )

            plt.figure(figsize=(10, 7))
            shap.summary_plot(shap_values, X_shap, show=False)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "stage4_catboost_shap_summary.png", dpi=200, bbox_inches="tight")
            plt.close()

            shap_detail_files = save_categorical_shap_details(
                X_shap=X_shap,
                shap_values=shap_values,
                feature_names=X_shap.columns.tolist(),
                target_features=["occupation_major", "industry_major"],
                output_dir=OUTPUT_DIR,
            )
            logger.info("SHAP outputs saved.")
        else:
            logger.warning("shap package not installed. Skip SHAP.")

        if "one_person_firm_flag" in test_df.columns:
            test_2plus = test_df[test_df["one_person_firm_flag"] != 1].copy()
            if len(test_2plus) > 0:
                X_test_2plus = prepare_catboost_input(test_2plus[feature_cols], numeric_features, categorical_features)
                y_test_2plus = test_2plus["exit_label_t1"].astype(int).to_numpy()
                prob_2plus = cb_model.predict_proba(X_test_2plus)[:, 1]

                robust_metrics = evaluate_binary_classifier(y_test_2plus, prob_2plus)
                pd.DataFrame(
                    [{"model": "catboost_2plus_firm_test", "n": len(test_2plus), **robust_metrics}]
                ).to_csv(
                    OUTPUT_DIR / "stage4_robustness_2plus_firm.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

    merge_cols = [c for c in ["pid", "wave", "exit_label_t1"] if c in pred_df.columns and c in test_df.columns]
    test_eval = test_df.merge(pred_df, on=merge_cols, how="left")
    test_eval = add_segment_columns(test_eval)

    prob_cols = [c for c in test_eval.columns if c.startswith("proba_")]
    segment_results = []
    for prob_col in prob_cols:
        model_name = prob_col.replace("proba_", "")
        for group_col in ["age_group", "weekly_hours_group", "firm_size_group"]:
            seg_df = evaluate_by_group(test_eval, prob_col, group_col, model_name)
            if not seg_df.empty:
                segment_results.append(seg_df)

    if segment_results:
        segment_out = pd.concat(segment_results, ignore_index=True)
        segment_out.to_csv(
            OUTPUT_DIR / "stage4_segment_performance.csv",
            index=False,
            encoding="utf-8-sig",
        )
        logger.info("Segment performance saved.")

    summary = {
        "analysis_shape": list(df.shape),
        "test_shape": list(test_df.shape),
        "prediction_columns": prob_cols,
        "shap_enabled": HAS_SHAP,
        "catboost_enabled": HAS_CATBOOST,
        "categorical_shap_detail_files": shap_detail_files,
    }

    with open(OUTPUT_DIR / "stage4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("==== Stage4 explainability + subgroup end ====")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_DIR = Path(r"G:\내 드라이브\project")
OUTPUT_DIR = PROJECT_DIR / "outputs_klips_sr"
TABLE_DIR = OUTPUT_DIR / "paper_tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "logs" / "klips_stage5_paper_tables.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning("Missing file: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json_if_exists(path: Path) -> Dict:
    if not path.exists():
        logger.warning("Missing file: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_table(df: pd.DataFrame, base_name: str) -> None:
    if df.empty:
        logger.warning("Skip empty table: %s", base_name)
        return
    csv_path = TABLE_DIR / f"{base_name}.csv"
    xlsx_path = TABLE_DIR / f"{base_name}.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    logger.info("Saved table: %s", base_name)


def format_metric_cols(df: pd.DataFrame, metric_cols: List[str], ndigits: int = 4) -> pd.DataFrame:
    out = df.copy()
    for col in metric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(ndigits)
    return out


def build_table_1_dataset_summary() -> pd.DataFrame:
    stage1_summary = read_json_if_exists(OUTPUT_DIR / "run_summary.json")
    stage4_summary = read_json_if_exists(OUTPUT_DIR / "stage4_summary.json")
    rows = [
        {"구분": "분석 전체표본", "값": stage1_summary.get("analysis_base_shape", [None, None])[0], "비고": "exit_label_t1 생성 후 분석 베이스"},
        {"구분": "테스트 표본", "값": stage4_summary.get("test_shape", [None, None])[0], "비고": "시간순 분할 기준 test set"},
        {"구분": "이탈률", "값": stage1_summary.get("analysis_base_exit_rate"), "비고": "임금근로 상태 이탈 비율"},
    ]
    df = pd.DataFrame(rows)
    if "값" in df.columns:
        df["값"] = df["값"].apply(lambda x: round(x, 4) if isinstance(x, float) else x)
    return df


def build_table_2_main_model_performance() -> pd.DataFrame:
    df = read_csv_if_exists(OUTPUT_DIR / "stage3_hybrid_metrics.csv")
    if df.empty:
        return df
    metric_cols = ["roc_auc", "pr_auc", "f1", "brier", "recall_at_10", "lift_at_10", "recall_at_20", "lift_at_20"]
    df = format_metric_cols(df, metric_cols)
    order = {"logistic": 1, "random_forest": 2, "xgboost": 3, "catboost": 4, "hybrid_stack": 5}
    df["model_order"] = df["model"].map(order)
    df["split_order"] = df["split"].map({"valid": 1, "test": 2})
    df = df.sort_values(["split_order", "model_order"]).drop(columns=["model_order", "split_order"])
    return df.rename(columns={"model": "모형", "split": "데이터셋", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1", "brier": "Brier", "recall_at_10": "Recall@10", "lift_at_10": "Lift@10", "recall_at_20": "Recall@20", "lift_at_20": "Lift@20"})


def build_table_3_bootstrap_ci() -> pd.DataFrame:
    df = read_csv_if_exists(OUTPUT_DIR / "stage3_bootstrap_ci.csv")
    if df.empty:
        return df
    df = format_metric_cols(df, ["bootstrap_mean", "ci_2.5", "ci_97.5"])
    df = df[df["metric"].isin(["roc_auc", "pr_auc", "brier"])].copy()
    df["95% CI"] = df.apply(lambda x: f"[{x['ci_2.5']:.4f}, {x['ci_97.5']:.4f}]" if pd.notna(x["ci_2.5"]) and pd.notna(x["ci_97.5"]) else "", axis=1)
    return df[["model", "metric", "bootstrap_mean", "95% CI"]].rename(columns={"model": "모형", "metric": "지표", "bootstrap_mean": "Bootstrap 평균"})


def build_table_4_shap_topn(top_n: int = 15) -> pd.DataFrame:
    df = read_csv_if_exists(OUTPUT_DIR / "stage4_catboost_shap_importance.csv")
    if df.empty:
        return df
    df = df.sort_values("mean_abs_shap", ascending=False).head(top_n).copy()
    df["rank"] = range(1, len(df) + 1)
    df["mean_abs_shap"] = pd.to_numeric(df["mean_abs_shap"], errors="coerce").round(6)
    return df[["rank", "feature", "mean_abs_shap"]].rename(columns={"rank": "순위", "feature": "변수", "mean_abs_shap": "평균절대SHAP"})


def build_table_4b_category_shap(feature_name: str, top_n: int = 10) -> pd.DataFrame:
    path = OUTPUT_DIR / f"stage4_{feature_name}_shap_details.csv"
    df = read_csv_if_exists(path)
    if df.empty:
        return df
    df = df.copy()
    for col in ["n", "mean_shap", "median_shap", "mean_abs_shap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["mean_shap", "mean_abs_shap", "n"], ascending=[False, False, False]).head(top_n)
    feature_label = "직업 대분류" if feature_name == "occupation_major" else "산업 대분류"
    return df.rename(columns={"category": feature_label, "n": "표본수", "mean_shap": "평균SHAP", "median_shap": "중앙SHAP", "mean_abs_shap": "평균절대SHAP", "direction": "방향"})


def build_table_5_segment_performance() -> pd.DataFrame:
    df = read_csv_if_exists(OUTPUT_DIR / "stage4_segment_performance.csv")
    if df.empty:
        return df
    metric_cols = ["event_rate", "roc_auc", "pr_auc", "f1", "brier", "recall_at_10", "lift_at_10", "recall_at_20", "lift_at_20"]
    df = format_metric_cols(df, metric_cols)
    df = df[df["model"].isin(["catboost", "hybrid_stack"])].copy()
    return df.rename(columns={"model": "모형", "group_col": "세그먼트구분", "group_value": "세그먼트값", "n": "표본수", "event_rate": "이탈률", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1", "brier": "Brier", "recall_at_10": "Recall@10", "lift_at_10": "Lift@10", "recall_at_20": "Recall@20", "lift_at_20": "Lift@20"})


def build_table_6_robustness() -> pd.DataFrame:
    df = read_csv_if_exists(OUTPUT_DIR / "stage4_robustness_2plus_firm.csv")
    if df.empty:
        return df
    metric_cols = ["roc_auc", "pr_auc", "f1", "brier", "recall_at_10", "lift_at_10", "recall_at_20", "lift_at_20"]
    df = format_metric_cols(df, metric_cols)
    return df.rename(columns={"model": "모형", "n": "표본수", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1", "brier": "Brier", "recall_at_10": "Recall@10", "lift_at_10": "Lift@10", "recall_at_20": "Recall@20", "lift_at_20": "Lift@20"})


def build_table_7_meta_coefficients() -> pd.DataFrame:
    df = read_csv_if_exists(OUTPUT_DIR / "stage3_hybrid_meta_coefficients.csv")
    if df.empty:
        return df
    df["coefficient"] = pd.to_numeric(df["coefficient"], errors="coerce").round(6)
    df["abs_coef"] = df["coefficient"].abs()
    df = df.sort_values("abs_coef", ascending=False).drop(columns="abs_coef")
    return df.rename(columns={"meta_feature": "메타변수", "coefficient": "계수"})


def build_appendix_wide_performance() -> pd.DataFrame:
    df = read_csv_if_exists(OUTPUT_DIR / "stage3_hybrid_metrics.csv")
    if df.empty:
        return df
    metric_cols = ["roc_auc", "pr_auc", "f1", "brier", "recall_at_10", "lift_at_10", "recall_at_20", "lift_at_20"]
    df = format_metric_cols(df, metric_cols)
    wide = df.pivot(index="model", columns="split", values=metric_cols)
    wide.columns = [f"{metric}_{split}" for metric, split in wide.columns]
    return wide.reset_index().rename(columns={"model": "모형"})


def main() -> None:
    logger.info("==== Stage5 paper tables start ====")
    save_table(build_table_1_dataset_summary(), "table_1_dataset_summary")
    save_table(build_table_2_main_model_performance(), "table_2_main_model_performance")
    save_table(build_table_3_bootstrap_ci(), "table_3_bootstrap_ci")
    save_table(build_table_4_shap_topn(top_n=15), "table_4_shap_top15")
    save_table(build_table_4b_category_shap("occupation_major", top_n=10), "table_4b_occupation_shap_details")
    save_table(build_table_4b_category_shap("industry_major", top_n=10), "table_4c_industry_shap_details")
    save_table(build_table_5_segment_performance(), "table_5_segment_performance")
    save_table(build_table_6_robustness(), "table_6_robustness_2plus_firm")
    save_table(build_table_7_meta_coefficients(), "table_7_hybrid_meta_coefficients")
    save_table(build_appendix_wide_performance(), "appendix_wide_model_performance")
    summary = {"tables_created": [
        "table_1_dataset_summary", "table_2_main_model_performance", "table_3_bootstrap_ci",
        "table_4_shap_top15", "table_4b_occupation_shap_details", "table_4c_industry_shap_details",
        "table_5_segment_performance", "table_6_robustness_2plus_firm", "table_7_hybrid_meta_coefficients",
        "appendix_wide_model_performance",
    ]}
    with open(TABLE_DIR / "stage5_tables_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("==== Stage5 paper tables end ====")


if __name__ == "__main__":
    main()

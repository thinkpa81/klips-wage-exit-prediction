from __future__ import annotations

import json
import logging
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# KLIPS SR PIPELINE (Windows local execution version)
# =========================================================

PROJECT_DIR = Path(r"G:\내 드라이브\project")

RAW_DIR_CANDIDATES = [
    PROJECT_DIR / "raw",   # 권장 구조
    PROJECT_DIR,           # 루트에 파일이 섞여 있는 경우
]


def resolve_raw_dir(candidates: List[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


RAW_DIR = resolve_raw_dir(RAW_DIR_CANDIDATES)
OUTPUT_DIR = PROJECT_DIR / "outputs_klips_sr"
INTERIM_DIR = OUTPUT_DIR / "interim"
PROCESSED_DIR = OUTPUT_DIR / "processed"
LOG_DIR = OUTPUT_DIR / "logs"
TEMP_DIR = OUTPUT_DIR / "temp"

for d in [PROJECT_DIR, OUTPUT_DIR, INTERIM_DIR, PROCESSED_DIR, LOG_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "klips_sr_pipeline.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


SPECIAL_MISSING_VALUES = {
    -1, -2, -3, -4, -7, -8, -9,
    999, 9999, 99999, 999999, 9999999,
}


@dataclass
class FileMeta:
    path: Path
    wave: int
    source_type: str  # h, p, a, w
    filename: str


@dataclass
class ConceptRule:
    concept: str
    source_type: str
    pattern: str
    description: str = ""


def discover_klips_files(raw_dir: Path) -> List[FileMeta]:
    metas: List[FileMeta] = []
    regex = re.compile(r"klips(\d{2})([ahpw])(?:\d)?\.(xlsx|xls)$", re.IGNORECASE)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Configured RAW_DIR does not exist: {raw_dir}")

    all_files = [p for p in raw_dir.rglob("*") if p.is_file()]
    logger.info("Search root: %s", raw_dir)
    logger.info("Total files found under root: %s", len(all_files))

    for path in sorted(all_files):
        m = regex.search(path.name)
        if not m:
            continue
        wave = int(m.group(1))
        source_type = m.group(2).lower()
        metas.append(FileMeta(path=path, wave=wave, source_type=source_type, filename=path.name))

    logger.info("Discovered %s KLIPS files", len(metas))
    if metas:
        logger.info("Sample discovered files: %s", [m.filename for m in metas[:10]])
    else:
        logger.info("No KLIPS files matched regex pattern: %s", regex.pattern)

    return metas


def build_inventory(file_metas: List[FileMeta]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for meta in file_metas:
        try:
            df = read_excel_safely(meta.path, nrows=5)
            rows.append(
                {
                    "filename": meta.filename,
                    "full_path": str(meta.path),
                    "wave": meta.wave,
                    "source_type": meta.source_type,
                    "n_preview_rows": len(df),
                    "n_preview_cols": df.shape[1],
                    "columns_preview": ", ".join(map(str, df.columns[:10])),
                }
            )
        except Exception as e:
            rows.append(
                {
                    "filename": meta.filename,
                    "full_path": str(meta.path),
                    "wave": meta.wave,
                    "source_type": meta.source_type,
                    "n_preview_rows": np.nan,
                    "n_preview_cols": np.nan,
                    "columns_preview": f"READ_ERROR: {e}",
                }
            )

    inventory = pd.DataFrame(rows).sort_values(["wave", "source_type", "filename"])
    inventory.to_csv(OUTPUT_DIR / "inventory_preview.csv", index=False, encoding="utf-8-sig")
    return inventory


def sanitize_xlsx_synchvertical(src_path: Path) -> Path:
    temp_subdir = Path(tempfile.mkdtemp(prefix="klips_xlsx_fix_", dir=str(TEMP_DIR)))
    fixed_path = temp_subdir / src_path.name

    with ZipFile(src_path, "r") as zin, ZipFile(fixed_path, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)

            if item.filename.startswith("xl/worksheets/") and item.filename.endswith(".xml"):
                data = data.replace(b' synchVertical="1"', b" ")
                data = data.replace(b' synchVertical="0"', b" ")
                data = data.replace(b"synchVertical=", b"syncVertical=")

            zout.writestr(item, data)

    return fixed_path


def read_excel_safely(path: Path, nrows: int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        raise ValueError(f"Unsupported file extension: {path}")

    if suffix == ".xls":
        try:
            df = pd.read_excel(path, dtype=object, nrows=nrows)
        except Exception:
            df = pd.read_excel(path, nrows=nrows)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    try:
        df = pd.read_excel(path, dtype=object, nrows=nrows)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except TypeError as e:
        if "synchVertical" not in str(e):
            raise

        logger.warning("openpyxl synchVertical issue detected: %s", path)
        fixed_path = sanitize_xlsx_synchvertical(path)
        logger.warning("Retry with sanitized workbook: %s", fixed_path)

        try:
            df = pd.read_excel(fixed_path, dtype=object, nrows=nrows)
        except Exception:
            df = pd.read_excel(fixed_path, nrows=nrows)

        df.columns = [str(c).strip() for c in df.columns]
        return df


def add_wave_and_year(df: pd.DataFrame, wave: int) -> pd.DataFrame:
    out = df.copy()
    out["wave"] = wave
    out["survey_year"] = 1997 + wave
    return out


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace({pd.NA: np.nan})
    for col in out.columns:
        try:
            out[col] = out[col].replace(list(SPECIAL_MISSING_VALUES), np.nan)
        except Exception:
            pass
        if out[col].dtype == "object":
            out[col] = out[col].replace({str(v): np.nan for v in SPECIAL_MISSING_VALUES})
    return out


CONCEPT_RULES: List[ConceptRule] = [
    ConceptRule("pid", "p", r"^pid$", "개인 고유 ID"),
    ConceptRule("gender_raw", "p", r"p\d{2}0101$", "성별"),
    ConceptRule("relation_to_head_raw", "p", r"p\d{2}0102$", "가구주와의 관계"),
    ConceptRule("birth_year_raw", "p", r"p\d{2}0104$", "출생연도/생년 관련"),
    ConceptRule("region_raw", "p", r"p\d{2}5501$", "지역"),
    ConceptRule("education_level_raw", "p", r"p\d{2}0110$", "학력"),
    ConceptRule("major_field_raw", "p", r"p\d{2}0121$", "전공계열"),
    ConceptRule("marital_status_raw", "p", r"p\d{2}0781$", "혼인상태"),
    ConceptRule("health_status_raw", "p", r"p\d{2}6101$", "건강 자기평가"),

    ConceptRule("employment_status_raw", "p", r"p\d{2}0201$", "취업 여부/경제활동 상태"),
    ConceptRule("employment_type_raw", "p", r"p\d{2}0211$", "취업 형태"),
    ConceptRule("family_work_help_raw", "p", r"p\d{2}0212$", "가족일 무급 도움"),
    ConceptRule("employee_status_raw", "p", r"p\d{2}0314$", "종사상 지위"),
    ConceptRule("work_type_raw", "p", r"p\d{2}0315$", "근무형태"),
    ConceptRule("job_position_raw", "p", r"p\d{2}0316$", "직위"),
    ConceptRule("regular_worker_raw", "p", r"p\d{2}0317$", "정규직 여부"),

    ConceptRule("job_start_year_raw", "p", r"p\d{2}0301$", "현 일자리 시작연도"),
    ConceptRule("job_start_month_raw", "p", r"p\d{2}0302$", "현 일자리 시작월"),
    ConceptRule("job_start_day_raw", "p", r"p\d{2}0303$", "현 일자리 시작일"),
    ConceptRule("industry_raw", "p", r"p\d{2}0340$", "산업 대분류"),
    ConceptRule("occupation_raw", "p", r"p\d{2}0350$", "직종 대분류"),
    ConceptRule("firm_size_raw", "p", r"p\d{2}0402$", "사업장 규모"),
    ConceptRule("contract_type_raw", "p", r"p\d{2}0501$", "근로계약 유형"),
    ConceptRule("weekly_hours_raw", "p", r"p\d{2}1003$", "주당 근로시간"),
    ConceptRule("monthly_wage_raw", "p", r"p\d{2}1641$", "월평균 임금"),
    ConceptRule("annual_earned_income_raw", "p", r"p\d{2}1702$", "연간 근로소득"),
    ConceptRule("employment_insurance_raw", "p", r"p\d{2}2103$", "고용보험"),
    ConceptRule("union_member_raw", "p", r"p\d{2}2501$", "노조가입"),
    ConceptRule("shift_work_raw", "p", r"p\d{2}2601$", "교대제"),

    ConceptRule("household_size_raw", "h", r"h\d{2}0150$", "가구원수"),
    ConceptRule("housing_tenure_raw", "h", r"h\d{2}1406$", "입주형태"),
    ConceptRule("housing_type_raw", "h", r"h\d{2}1407$", "주택종류"),
    ConceptRule("housing_deposit_raw", "h", r"h\d{2}1413$", "임대보증금"),
    ConceptRule("housing_rent_raw", "h", r"h\d{2}1414$", "월세"),
    ConceptRule("household_labor_income_raw", "h", r"h\d{2}2101$", "가구근로소득"),
]


def find_columns_by_rule(columns: Iterable[str], rule: ConceptRule) -> List[str]:
    pat = re.compile(rule.pattern, re.IGNORECASE)
    return [c for c in columns if pat.search(str(c))]


def extract_concepts(df: pd.DataFrame, source_type: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    for rule in CONCEPT_RULES:
        if rule.source_type != source_type:
            continue
        matched = find_columns_by_rule(df.columns, rule)
        if not matched:
            continue
        if len(matched) > 1:
            logger.warning("Multiple columns matched for %s: %s", rule.concept, matched)
        out[rule.concept] = df[matched[0]]

    if "pid" not in out.columns:
        for cand in ["pid", "PID", "개인번호", "개인id", "개인ID"]:
            if cand in df.columns:
                out["pid"] = df[cand]
                break

    if "hhid" not in out.columns:
        for cand in ["hhid", "HHID", "가구번호", "가구id", "가구ID"]:
            if cand in df.columns:
                out["hhid"] = df[cand]
                break

    return out


def load_source_panels(file_metas: List[FileMeta]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    person_frames: List[pd.DataFrame] = []
    household_frames: List[pd.DataFrame] = []

    for meta in file_metas:
        if meta.source_type not in {"p", "h"}:
            continue

        logger.info("Reading %s", meta.path)
        raw = read_excel_safely(meta.path)
        raw = add_wave_and_year(raw, meta.wave)
        raw = normalize_missing_values(raw)
        extracted = extract_concepts(raw, meta.source_type)
        extracted["wave"] = meta.wave
        extracted["survey_year"] = 1997 + meta.wave
        extracted["source_file"] = meta.filename

        if meta.source_type == "p":
            person_frames.append(extracted)
        elif meta.source_type == "h":
            household_frames.append(extracted)

    person_df = pd.concat(person_frames, ignore_index=True, sort=False) if person_frames else pd.DataFrame()
    household_df = pd.concat(household_frames, ignore_index=True, sort=False) if household_frames else pd.DataFrame()

    return person_df, household_df


def build_panel_master(person_df: pd.DataFrame, household_df: pd.DataFrame) -> pd.DataFrame:
    if person_df.empty:
        raise ValueError("person_df is empty")

    panel = person_df.copy()

    if not household_df.empty and "hhid" in panel.columns and "hhid" in household_df.columns:
        hh_cols = [c for c in household_df.columns if c not in {"source_file"}]
        hh_cols = list(dict.fromkeys(hh_cols))
        hh_use = household_df[hh_cols].copy()

        merge_keys = [c for c in ["hhid", "wave", "survey_year"] if c in hh_use.columns and c in panel.columns]
        hh_non_keys = [c for c in hh_use.columns if c not in merge_keys]
        hh_non_keys = [c for c in hh_non_keys if c not in panel.columns]

        panel = panel.merge(hh_use[merge_keys + hh_non_keys], on=merge_keys, how="left")

    subset_keys = [c for c in ["pid", "wave"] if c in panel.columns]
    if subset_keys:
        panel = panel.drop_duplicates(subset=subset_keys)

    panel.to_csv(INTERIM_DIR / "panel_master_raw.csv", index=False, encoding="utf-8-sig")
    logger.info("Panel master shape: %s", panel.shape)
    return panel


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def infer_age_from_birth_year(birth_year: pd.Series, survey_year: pd.Series) -> pd.Series:
    birth_year_num = to_numeric(birth_year)
    survey_year_num = to_numeric(survey_year)
    return survey_year_num - birth_year_num + 1


def map_gender(series: pd.Series) -> pd.Series:
    s = to_numeric(series)
    mapping = {1: "male", 2: "female"}
    return s.map(mapping)


def derive_employment_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in [
        "employment_status_raw",
        "employment_type_raw",
        "family_work_help_raw",
        "employee_status_raw",
        "weekly_hours_raw",
        "monthly_wage_raw",
        "annual_earned_income_raw",
        "job_start_year_raw",
        "job_start_month_raw",
        "birth_year_raw",
        "firm_size_raw",
        "household_size_raw",
        "housing_deposit_raw",
        "housing_rent_raw",
        "household_labor_income_raw",
    ]:
        if col in out.columns:
            out[col] = to_numeric(out[col])

    if "gender_raw" in out.columns:
        out["gender"] = map_gender(out["gender_raw"])

    if "birth_year_raw" in out.columns:
        out["age_final"] = infer_age_from_birth_year(out["birth_year_raw"], out["survey_year"])

    if "employee_status_raw" in out.columns:
        out["is_wage_worker_t"] = out["employee_status_raw"].isin([1, 2, 3]).astype(float)
        out["is_non_wage_worker_t"] = out["employee_status_raw"].isin([4, 5, 6]).astype(float)
    else:
        out["is_wage_worker_t"] = np.nan
        out["is_non_wage_worker_t"] = np.nan

    if "employment_status_raw" in out.columns:
        out["is_employed_t"] = out["employment_status_raw"].notna().astype(float)
    else:
        out["is_employed_t"] = np.nan

    return out


def build_core_features(df: pd.DataFrame) -> pd.DataFrame:
    out = derive_employment_flags(df)

    if "age_final" in out.columns:
        out.loc[out["age_final"] < 15, "age_final"] = np.nan
        out["age_gt_90_flag"] = (out["age_final"] > 90).astype(float)

    if {"job_start_year_raw", "job_start_month_raw", "survey_year"}.issubset(out.columns):
        survey_month_assumed = 12
        tenure_months = (out["survey_year"] - out["job_start_year_raw"]) * 12 + (
            survey_month_assumed - out["job_start_month_raw"]
        )
        out["tenure_months"] = tenure_months.where(tenure_months >= 0, np.nan)
        out["tenure_years"] = out["tenure_months"] / 12.0

    if "weekly_hours_raw" in out.columns:
        out["weekly_hours"] = out["weekly_hours_raw"].copy()
        out.loc[out["weekly_hours"] <= 0, "weekly_hours"] = np.nan
        out["weekly_hours_missing"] = out["weekly_hours"].isna().astype(float)
        out["gt48"] = (out["weekly_hours"] > 48).astype(float)
        out["gt52"] = (out["weekly_hours"] > 52).astype(float)
        out["ge55"] = (out["weekly_hours"] >= 55).astype(float)
        out["weekly_hours_qc_gt112"] = (out["weekly_hours"] > 112).astype(float)
        out["weekly_hours_band"] = pd.cut(
            out["weekly_hours"],
            bins=[0, 19, 29, 34, 39, np.inf],
            labels=["1-19", "20-29", "30-34", "35-39", "40+"],
            right=True,
        )

    if "monthly_wage_raw" in out.columns:
        out["monthly_wage"] = out["monthly_wage_raw"].copy()
        out.loc[out["monthly_wage"] <= 0, "monthly_wage"] = np.nan
        out["monthly_wage_missing"] = out["monthly_wage"].isna().astype(float)
        out["log_monthly_wage"] = np.log1p(out["monthly_wage"])

    if "firm_size_raw" in out.columns:
        out["one_person_firm_flag"] = (out["firm_size_raw"] == 1).astype(float)

    if {"housing_rent_raw", "household_labor_income_raw"}.issubset(out.columns):
        income_monthly = out["household_labor_income_raw"] / 12.0
        out["housing_cost_burden"] = out["housing_rent_raw"] / income_monthly.replace(0, np.nan)

    # sklearn 충돌 방지: string dtype 대신 object
    if "education_level_raw" in out.columns:
        out["education_level"] = out["education_level_raw"].astype(object)
    if "marital_status_raw" in out.columns:
        out["marital_status"] = out["marital_status_raw"].astype(object)
    if "region_raw" in out.columns:
        out["region"] = out["region_raw"].astype(object)
    if "industry_raw" in out.columns:
        out["industry_major"] = out["industry_raw"].astype(object)
    if "occupation_raw" in out.columns:
        out["occupation_major"] = out["occupation_raw"].astype(object)
    if "housing_tenure_raw" in out.columns:
        out["housing_tenure_type"] = out["housing_tenure_raw"].astype(object)
    if "housing_type_raw" in out.columns:
        out["housing_type"] = out["housing_type_raw"].astype(object)

    out = out.replace({pd.NA: np.nan})
    return out


def make_exit_label(df: pd.DataFrame) -> pd.DataFrame:
    required = ["pid", "wave", "is_wage_worker_t"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for label generation: {missing}")

    out = df.sort_values(["pid", "wave"]).copy()
    out["wave_t1"] = out.groupby("pid")["wave"].shift(-1)
    out["is_wage_worker_t1"] = out.groupby("pid")["is_wage_worker_t"].shift(-1)
    out["has_next_wave"] = out["wave_t1"] == (out["wave"] + 1)

    analysis_base = out[(out["is_wage_worker_t"] == 1) & (out["has_next_wave"])].copy()
    analysis_base["exit_label_t1"] = (analysis_base["is_wage_worker_t1"] != 1).astype(int)

    analysis_base.to_csv(PROCESSED_DIR / "analysis_base_with_label.csv", index=False, encoding="utf-8-sig")
    logger.info("Analysis base shape: %s", analysis_base.shape)
    if not analysis_base.empty:
        logger.info("Exit rate: %.4f", analysis_base["exit_label_t1"].mean())
    return analysis_base


def timewise_split(df: pd.DataFrame, train_end: int, valid_end: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["wave"] <= train_end].copy()
    valid = df[(df["wave"] > train_end) & (df["wave"] <= valid_end)].copy()
    test = df[df["wave"] > valid_end].copy()

    logger.info("Train shape=%s, Valid shape=%s, Test shape=%s", train.shape, valid.shape, test.shape)
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
        c
        for c in candidate_features
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
    X = X.copy()
    X = X.replace({pd.NA: np.nan})

    for col in numeric_features:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].astype(object)
            X.loc[pd.isna(X[col]), col] = np.nan

    return X


def recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.1) -> float:
    n = len(y_true)
    top_n = max(1, int(math.ceil(n * k)))
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
    top_n = max(1, int(math.ceil(n * k)))
    idx = np.argsort(-y_prob)[:top_n]
    precision_top = y_true[idx].mean()
    return float(precision_top / base_rate)


def evaluate_binary_classifier(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
        "recall_at_10": recall_at_k(y_true, y_prob, 0.10),
        "lift_at_10": lift_at_k(y_true, y_prob, 0.10),
        "recall_at_20": recall_at_k(y_true, y_prob, 0.20),
        "lift_at_20": lift_at_k(y_true, y_prob, 0.20),
    }
    return metrics


def fit_baseline_logistic(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols, numeric_features, categorical_features = select_model_columns(train_df)
    target_col = "exit_label_t1"

    if not feature_cols:
        raise ValueError("No feature columns selected.")

    X_train = sanitize_model_input(train_df[feature_cols], numeric_features, categorical_features)
    y_train = train_df[target_col].astype(int)

    X_valid = sanitize_model_input(valid_df[feature_cols], numeric_features, categorical_features)
    y_valid = valid_df[target_col].astype(int)

    X_test = sanitize_model_input(test_df[feature_cols], numeric_features, categorical_features)
    y_test = test_df[target_col].astype(int)

    preprocessor = ColumnTransformer(
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

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs")),
        ]
    )

    clf.fit(X_train, y_train)

    valid_prob = clf.predict_proba(X_valid)[:, 1]
    test_prob = clf.predict_proba(X_test)[:, 1]

    valid_metrics = evaluate_binary_classifier(y_valid.to_numpy(), valid_prob)
    test_metrics = evaluate_binary_classifier(y_test.to_numpy(), test_prob)

    rows = []
    for split_name, metric_dict in [("valid", valid_metrics), ("test", test_metrics)]:
        row = {"model": "logistic_baseline", "split": split_name}
        row.update(metric_dict)
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_DIR / "baseline_logistic_metrics.csv", index=False, encoding="utf-8-sig")

    pred_cols = [c for c in ["pid", "wave", target_col] if c in test_df.columns]
    pred_out = test_df[pred_cols].copy() if pred_cols else pd.DataFrame(index=test_df.index)
    pred_out["proba_logistic"] = test_prob
    pred_out.to_csv(OUTPUT_DIR / "baseline_logistic_test_predictions.csv", index=False, encoding="utf-8-sig")

    logger.info("Baseline logistic results saved.")
    return result_df


def write_data_quality_report(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        pd.DataFrame([{"dataset": name, "note": "EMPTY_DATAFRAME"}]).to_csv(
            OUTPUT_DIR / f"dq_{name}.csv", index=False, encoding="utf-8-sig"
        )
        return

    rows = []
    for col in df.columns:
        rows.append(
            {
                "dataset": name,
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_rate": float(df[col].isna().mean()),
                "n_unique": int(df[col].nunique(dropna=True)),
            }
        )

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"dq_{name}.csv", index=False, encoding="utf-8-sig")


def save_path_diagnostics(raw_dir: Path) -> None:
    rows = [
        {"check": "PROJECT_DIR", "value": str(PROJECT_DIR), "exists": PROJECT_DIR.exists()},
        {"check": "RAW_DIR", "value": str(raw_dir), "exists": raw_dir.exists()},
    ]

    preview_files = []
    if raw_dir.exists():
        for i, p in enumerate(raw_dir.rglob("*")):
            if p.is_file():
                preview_files.append(str(p))
            if i >= 49:
                break

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "path_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"preview_file": preview_files}).to_csv(
        OUTPUT_DIR / "path_preview_files.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    logger.info("==== KLIPS SR pipeline start ====")
    logger.info("PROJECT_DIR=%s", PROJECT_DIR)
    logger.info("RAW_DIR=%s", RAW_DIR)

    save_path_diagnostics(RAW_DIR)

    file_metas = discover_klips_files(RAW_DIR)
    if not file_metas:
        raise FileNotFoundError(
            f"No KLIPS files discovered under RAW_DIR={RAW_DIR}. "
            "Check the path, filename pattern, extension (.xlsx/.xls), and whether files are inside subfolders."
        )

    inventory = build_inventory(file_metas)
    logger.info("Inventory preview saved: %s", OUTPUT_DIR / "inventory_preview.csv")
    logger.info("Inventory rows=%s", len(inventory))

    person_df, household_df = load_source_panels(file_metas)
    write_data_quality_report(person_df, "person_extracted")
    write_data_quality_report(household_df, "household_extracted")

    panel_master = build_panel_master(person_df, household_df)
    write_data_quality_report(panel_master, "panel_master")

    core_df = build_core_features(panel_master)
    core_df.to_csv(PROCESSED_DIR / "core_features.csv", index=False, encoding="utf-8-sig")
    write_data_quality_report(core_df, "core_features")

    analysis_base = make_exit_label(core_df)
    write_data_quality_report(analysis_base, "analysis_base")

    train_df, valid_df, test_df = timewise_split(analysis_base, train_end=20, valid_end=23)

    if min(len(train_df), len(valid_df), len(test_df)) == 0:
        logger.warning("One of the splits is empty. Skip model training.")
    else:
        metrics = fit_baseline_logistic(train_df, valid_df, test_df)
        logger.info("\n%s", metrics)

    summary = {
        "project_dir": str(PROJECT_DIR),
        "raw_dir": str(RAW_DIR),
        "n_discovered_files": len(file_metas),
        "person_shape": list(person_df.shape),
        "household_shape": list(household_df.shape),
        "panel_master_shape": list(panel_master.shape),
        "analysis_base_shape": list(analysis_base.shape),
        "analysis_base_exit_rate": float(analysis_base["exit_label_t1"].mean()) if not analysis_base.empty else None,
    }
    with open(OUTPUT_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("==== KLIPS SR pipeline end ====")


if __name__ == "__main__":
    main()
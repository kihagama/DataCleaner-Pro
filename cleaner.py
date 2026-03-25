"""
Data Cleaning Module
====================
Handles detection and automated cleaning of:
  - Missing values
  - Duplicate rows
  - Type mismatches
  - Outliers (IQR & z-score)
  - ML-based anomaly detection (Isolation Forest)
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Detection helpers
# ──────────────────────────────────────────────────────────────

def detect_missing(df: pd.DataFrame) -> pd.Series:
    """Return per-column null counts."""
    return df.isnull().sum()


def detect_duplicates(df: pd.DataFrame) -> int:
    """Return number of fully duplicate rows."""
    return int(df.duplicated().sum())


def detect_type_mismatches(df: pd.DataFrame) -> dict:
    """
    Identify columns whose inferred dtype differs from the stored dtype.
    Returns {col: suggested_type}.
    """
    suggestions = {}
    for col in df.columns:
        if df[col].dtype == object:
            # Try numeric
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() / max(len(df), 1) > 0.9:
                suggestions[col] = "numeric"
                continue
            # Try datetime
            try:
                pd.to_datetime(df[col], infer_datetime_format=True, errors="raise")
                suggestions[col] = "datetime"
            except Exception:
                pass
    return suggestions


def detect_outliers_iqr(df: pd.DataFrame) -> dict:
    """IQR-based outlier detection for numeric columns. Returns {col: count}."""
    result = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        result[col] = int(mask.sum())
    return result


def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> dict:
    """Z-score outlier detection. Returns {col: count}."""
    result = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        z = np.abs(stats.zscore(df[col].dropna()))
        result[col] = int((z > threshold).sum())
    return result


def detect_anomalies_isolation_forest(
    df: pd.DataFrame, contamination: float = 0.05
) -> pd.Series:
    """
    Isolation Forest anomaly detection on all numeric columns.
    Returns a boolean Series (True = anomaly).
    """
    numeric = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if numeric.empty or numeric.shape[1] == 0:
        logger.warning("No numeric columns available for anomaly detection.")
        return pd.Series(False, index=df.index)

    filled = numeric.fillna(numeric.median())
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(filled)          # -1 = anomaly, 1 = normal
    return pd.Series(preds == -1, index=df.index)


# ──────────────────────────────────────────────────────────────
# Cleaning helpers
# ──────────────────────────────────────────────────────────────

def impute_missing(
    df: pd.DataFrame, strategy: str = "auto"
) -> pd.DataFrame:
    """
    Impute missing values.
    strategy: 'mean' | 'median' | 'mode' | 'auto'
    'auto' uses median for numeric and mode for categorical.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype in [np.float64, np.int64, float, int] or pd.api.types.is_numeric_dtype(df[col]):
            if strategy in ("mean", "auto"):
                fill = df[col].mean() if strategy == "mean" else df[col].median()
            else:
                fill = df[col].mode().iloc[0] if not df[col].mode().empty else 0
            df[col] = df[col].fillna(fill)
            logger.info(f"Imputed numeric column '{col}' with {fill:.4f}")
        else:
            fill = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(fill)
            logger.info(f"Imputed categorical column '{col}' with '{fill}'")
    return df


def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to numeric or datetime where strongly suggested."""
    df = df.copy()
    suggestions = detect_type_mismatches(df)
    for col, suggested in suggestions.items():
        if suggested == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            logger.info(f"Converted '{col}' to numeric")
        elif suggested == "datetime":
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            logger.info(f"Converted '{col}' to datetime")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully duplicate rows."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Removed {before - len(df)} duplicate rows")
    return df


def handle_outliers(
    df: pd.DataFrame, method: str = "iqr", action: str = "cap"
) -> pd.DataFrame:
    """
    Handle outliers in numeric columns.
    method: 'iqr' | 'zscore'
    action: 'cap' (winsorize) | 'remove'
    """
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if method == "iqr":
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        else:
            mean, std = df[col].mean(), df[col].std()
            lo, hi = mean - 3 * std, mean + 3 * std

        if action == "cap":
            df[col] = df[col].clip(lower=lo, upper=hi)
        else:
            df = df[(df[col] >= lo) & (df[col] <= hi)]
        logger.info(f"Handled outliers in '{col}' via {method}/{action}")
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# Quality report
# ──────────────────────────────────────────────────────────────

def build_quality_report(df: pd.DataFrame) -> dict:
    """
    Generate a comprehensive data quality report dictionary.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_iqr = detect_outliers_iqr(df)
    outlier_z   = detect_outliers_zscore(df)
    missing     = detect_missing(df)
    dupes       = detect_duplicates(df)
    type_issues = detect_type_mismatches(df)

    col_reports = []
    for col in df.columns:
        col_reports.append({
            "column":        col,
            "dtype":         str(df[col].dtype),
            "null_count":    int(missing[col]),
            "null_pct":      round(missing[col] / max(len(df), 1) * 100, 2),
            "unique_values": int(df[col].nunique()),
            "outliers_iqr":  outlier_iqr.get(col, 0),
            "outliers_z":    outlier_z.get(col, 0),
            "type_issue":    type_issues.get(col, None),
        })

    return {
        "total_rows":     len(df),
        "total_cols":     len(df.columns),
        "total_missing":  int(missing.sum()),
        "duplicate_rows": dupes,
        "numeric_cols":   numeric_cols,
        "columns":        col_reports,
    }


# ──────────────────────────────────────────────────────────────
# Suggestions engine
# ──────────────────────────────────────────────────────────────

def suggest_cleaning_strategies(report: dict) -> list[str]:
    """Return plain-English cleaning suggestions based on the quality report."""
    tips = []
    for col in report["columns"]:
        name = col["column"]
        if col["null_pct"] > 50:
            tips.append(f"⚠️  '{name}' has {col['null_pct']}% missing — consider dropping the column.")
        elif col["null_pct"] > 0:
            tips.append(f"🔧 Impute missing values in '{name}' (null_pct={col['null_pct']}%).")
        if col["outliers_iqr"] > 0:
            tips.append(f"📊 '{name}' has {col['outliers_iqr']} IQR outliers — review or cap them.")
        if col["type_issue"]:
            tips.append(f"🔄 Convert '{name}' from object to {col['type_issue']}.")
    if report["duplicate_rows"] > 0:
        tips.append(f"🗑️  Remove {report['duplicate_rows']} duplicate row(s).")
    if not tips:
        tips.append("✅ Dataset looks clean — no major issues detected!")
    return tips

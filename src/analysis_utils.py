"""Exploratory data analysis and subgroup error audit for stroke classification."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def exploratory_report(df: pd.DataFrame) -> dict:
    """Describe problems without learning any preprocessing statistics for the model."""
    info = io.StringIO()
    df.info(buf=info)
    print("\n=== EDA: df.info() ===")
    print(info.getvalue())

    target_counts = df["stroke"].value_counts().sort_index()
    print("\n=== Class counts and proportions ===")
    for label, count in target_counts.items():
        print(f"stroke={label}: {count} / {len(df)} = {count / len(df):.2%}")

    missing = {c: int(n) for c, n in df.isna().sum().items() if n > 0}
    outliers = {}
    for col in ["age", "bmi", "avg_glucose_level"]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        q1, q3 = series.quantile([0.25, 0.75])
        lower, upper = float(q1 - 1.5 * (q3 - q1)), float(q3 + 1.5 * (q3 - q1))
        outliers[col] = {
            "iqr_lower": round(lower, 3),
            "iqr_upper": round(upper, 3),
            "count_outside_iqr": int(((series < lower) | (series > upper)).sum()),
            "minimum": float(series.min()),
            "maximum": float(series.max()),
        }

    report = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "target_proportions": {str(k): round(float(v / len(df)), 6) for k, v in target_counts.items()},
        "missing_values": missing,
        "smoking_status_unknown": int(df["smoking_status"].eq("Unknown").sum()),
        "smoking_status_unknown_note": "Literal 'Unknown' is retained as its own category; it is not pandas NaN.",
        "duplicate_full_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df["id"].duplicated().sum()),
        "age_zero_count": int(df["age"].eq(0).sum()),
        "bmi_gt_50_count": int(df["bmi"].gt(50).sum()),
        "gender_other_count": int(df["gender"].eq("Other").sum()),
        "outlier_screen_iqr": outliers,
        "data_cautions": [
            "IQR flags are candidates for examination, not necessarily invalid measurements.",
            "Age zero and unusual BMI values are investigated but not automatically removed.",
            "This retrospective table records stroke status, not the future probability of an incident.",
            "The test set must not be used to choose models, thresholds, or preprocessing statistics.",
        ],
    }
    print("\n=== Data quality ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def subgroup_error_table(X: pd.DataFrame, y_true, y_pred) -> pd.DataFrame:
    """Auditable confusion counts by group for one held-out test prediction."""
    frame = X.reset_index(drop=True).copy()
    frame["truth"] = np.asarray(y_true, dtype=int)
    frame["prediction"] = np.asarray(y_pred, dtype=int)
    frame["age_band"] = pd.cut(
        frame["age"],
        bins=[-np.inf, 18, 40, 60, 80, np.inf],
        labels=["under_18", "18_to_39", "40_to_59", "60_to_79", "80_plus"],
        right=False,
    )
    group_columns = ["age_band", "gender", "hypertension", "heart_disease",
                     "Residence_type", "smoking_status"]
    rows = []
    for column in group_columns:
        for value, group in frame.groupby(column, observed=True, dropna=False):
            tn, fp, fn, tp = confusion_matrix(
                group["truth"], group["prediction"], labels=[0, 1]
            ).ravel()
            positives = tp + fn
            flagged = tp + fp
            rows.append({
                "feature": column,
                "group": str(value),
                "n": int(len(group)),
                "actual_stroke_cases": int(positives),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                "recall": round(float(tp / positives), 5) if positives else None,
                "precision": round(float(tp / flagged), 5) if flagged else None,
                "small_positive_sample": bool(positives < 10),
            })
    result = pd.DataFrame(rows)
    # Each feature partition must independently cover every test patient.
    for feature in group_columns:
        part = result[result["feature"] == feature]
        assert int(part["n"].sum()) == len(frame), feature
    return result


def save_group_errors(table: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    print(f"\nSaved subgroup errors to {path}")
    print(table.to_string(index=False))

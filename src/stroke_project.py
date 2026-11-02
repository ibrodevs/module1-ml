"""Stroke risk project: basic, main, and advanced blocks.
Run from repository root: python src/stroke_project.py --variant all
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, fbeta_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "healthcare-dataset-stroke-data.csv"
OUT = ROOT / "results"
TARGET = "stroke"
DROP = ["id"]
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA)
    required = {"id", "gender", "age", "hypertension", "heart_disease", "ever_married",
                "work_type", "Residence_type", "avg_glucose_level", "bmi",
                "smoking_status", TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    return df


def preprocessors(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encode", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)])


def model_pipeline(X, estimator):
    return Pipeline([("preprocess", preprocessors(X)), ("model", estimator)])


def metric_dict(y, pred, scores):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, pred)),
            "precision_stroke": float(precision_score(y, pred, zero_division=0)),
            "recall_stroke": float(recall_score(y, pred, zero_division=0)),
            "f1_stroke": float(f1_score(y, pred, zero_division=0)),
            "f2_stroke": float(fbeta_score(y, pred, beta=2, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, scores)),
            "average_precision": float(average_precision_score(y, scores)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def choose_f2_threshold(y, scores):
    thresholds = np.unique(np.r_[np.linspace(0.01, 0.99, 99), scores])
    vals = [fbeta_score(y, scores >= t, beta=2, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(vals))])


def save(name, content):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")


def basic(df):
    print("\n=== Вариант A — базовый ===")
    print(f"Размер: {df.shape}; пропуски BMI: {df.bmi.isna().sum()}")
    print("Классы stroke:\n", df.stroke.value_counts().sort_index().to_string())
    X = df.drop(columns=[TARGET, *DROP]); y = df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=RANDOM_STATE)
    pipe = model_pipeline(Xtr, LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    pipe.fit(Xtr, ytr)
    prob = pipe.predict_proba(Xte)[:, 1]
    res = metric_dict(yte, prob >= .5, prob)
    save("basic_metrics.json", {"split": "stratified 80/20, random_state=42", "threshold": .5, **res})
    print(json.dumps(res, indent=2, ensure_ascii=False))


def main(df):
    print("\n=== Вариант B — основной ===")
    X = df.drop(columns=[TARGET, *DROP]); y = df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                           class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }
    results = {}
    for name, estimator in candidates.items():
        pipe = model_pipeline(Xtr, estimator)
        oof = cross_val_predict(pipe, Xtr, ytr, cv=cv, method="predict_proba", n_jobs=None)[:, 1]
        threshold = choose_f2_threshold(ytr.to_numpy(), oof)
        pipe.fit(Xtr, ytr)
        prob = pipe.predict_proba(Xte)[:, 1]
        results[name] = {"threshold_selected_on_training_oof": threshold,
                         **metric_dict(yte, prob >= threshold, prob)}
    save("main_metrics.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False))


def advanced(df):
    print("\n=== Вариант C — продвинутый ===")
    X = df.drop(columns=[TARGET, *DROP]); y = df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    strategies = {
        "unweighted_logistic_default_threshold": (LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), .5),
        "class_weight_balanced_threshold_tuned": (LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE), .5),
    }
    results = {}
    for name, (estimator, fixed_threshold) in strategies.items():
        pipe = model_pipeline(Xtr, estimator)
        oof = cross_val_predict(pipe, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        threshold = choose_f2_threshold(ytr.to_numpy(), oof) if "balanced" in name else fixed_threshold
        pipe.fit(Xtr, ytr)
        prob = pipe.predict_proba(Xte)[:, 1]
        results[name] = {"threshold": threshold, **metric_dict(yte, prob >= threshold, prob)}
    save("advanced_metrics.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False))


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["basic", "main", "advanced", "all"], default="all")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    df = load_data()
    for part in (["basic", "main", "advanced"] if args.variant == "all" else [args.variant]):
        {"basic": basic, "main": main, "advanced": advanced}[part](df)

if __name__ == "__main__":
    main_cli()

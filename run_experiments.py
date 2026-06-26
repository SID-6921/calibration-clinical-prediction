"""
Calibration is All You Need? A Multi-Dataset Investigation of When
Post-hoc Calibration Helps, Hurts, and Misleads in Clinical Risk Prediction.

Main experiment pipeline.
Author: Siddhardha Nanda, Columbia University BME
"""

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy import stats

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 10  # 10-fold for statistical power
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# DATASETS
# ============================================================

def load_pima():
    """Pima Indians Diabetes (n=768, binary)."""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ["pregnancies", "glucose", "bp", "skin", "insulin", "bmi", "dpf", "age", "outcome"]
    df = pd.read_csv(url, header=None, names=cols)
    for c in ["glucose", "bp", "skin", "insulin", "bmi"]:
        df[c] = df[c].replace(0, np.nan).fillna(df[c].median())
    X = df.drop("outcome", axis=1)
    y = df["outcome"]
    return X, y, "Pima Diabetes"


def load_heart_cleveland():
    """Cleveland Heart Disease (n=303, binary)."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    df = pd.read_csv(url, header=None, names=cols, na_values="?")
    df = df.dropna()
    X = df.drop("target", axis=1).astype(float)
    y = (df["target"].astype(int) > 0).astype(int)
    return X, y, "Cleveland Heart"


def load_breast_cancer():
    """Wisconsin Breast Cancer (n=569, binary)."""
    from sklearn.datasets import load_breast_cancer as lbc
    data = lbc()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(1 - data.target)  # malignant=1
    return X, y, "Breast Cancer"


def load_indian_liver():
    """Indian Liver Patient (n=583, binary)."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"
    cols = ["age", "gender", "tb", "db", "alkphos", "sgpt", "sgot", "tp", "alb", "ag_ratio", "target"]
    df = pd.read_csv(url, header=None, names=cols)
    df["gender"] = (df["gender"] == "Male").astype(int)
    df = df.dropna()
    X = df.drop("target", axis=1)
    y = (df["target"] == 1).astype(int)
    return X, y, "Indian Liver"


def load_early_diabetes():
    """Early Stage Diabetes Risk (n=520, binary)."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv"
    df = pd.read_csv(url)
    binary_cols = [c for c in df.columns if df[c].dtype == object and c != "class"]
    for c in binary_cols:
        df[c] = (df[c] == "Yes").astype(int) if "Yes" in df[c].values else pd.factorize(df[c])[0]
    X = df.drop("class", axis=1)
    y = (df["class"] == "Positive").astype(int)
    return X, y, "Early Diabetes"


def load_chronic_kidney():
    """Chronic Kidney Disease (n=400, binary)."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00336/Chronic_Kidney_Disease.zip"
    try:
        import zipfile, io, requests
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        with z.open([n for n in z.namelist() if n.endswith(".arff")][0]) as f:
            lines = f.read().decode("utf-8").split("\n")
        data_start = next(i for i, l in enumerate(lines) if l.strip().upper() == "@DATA") + 1
        data_lines = [l.strip() for l in lines[data_start:] if l.strip() and not l.startswith("%")]
        cols = ["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
                "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
                "htn", "dm", "cad", "appet", "pe", "ane", "class"]
        df = pd.DataFrame([l.split(",") for l in data_lines], columns=cols)
        df = df.replace("?", np.nan).replace("\t?", np.nan)
        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c])
            except (ValueError, TypeError):
                df[c] = pd.factorize(df[c])[0]
        df = df.dropna()
        X = df.drop("class", axis=1)
        y = (df["class"] == 0).astype(int)  # ckd=1
        return X, y, "Chronic Kidney"
    except Exception:
        return None, None, None


def get_all_datasets():
    loaders = [load_pima, load_heart_cleveland, load_breast_cancer,
               load_indian_liver, load_early_diabetes, load_chronic_kidney]
    datasets = []
    for loader in loaders:
        try:
            X, y, name = loader()
            if X is not None and len(X) > 50:
                datasets.append((X, y, name))
                print(f"  [OK] {name}: n={len(X)}, features={X.shape[1]}, "
                      f"prevalence={y.mean():.1%}")
        except Exception as e:
            print(f"  [SKIP] {loader.__name__}: {e}")
    return datasets


# ============================================================
# MODELS
# ============================================================

def get_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, random_state=RANDOM_STATE
        ),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=2000, learning_rate_init=0.001,
                random_state=RANDOM_STATE, early_stopping=True,
                validation_fraction=0.15, solver="adam",
            )),
        ]),
    }


# ============================================================
# CALIBRATION
# ============================================================

def platt_scale(y_train_prob, y_train_true, y_test_prob):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(y_train_prob.reshape(-1, 1), y_train_true)
    return lr.predict_proba(y_test_prob.reshape(-1, 1))[:, 1]


def isotonic_calibrate(y_train_prob, y_train_true, y_test_prob):
    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    ir.fit(y_train_prob, y_train_true)
    return ir.predict(y_test_prob)


def beta_calibrate(y_train_prob, y_train_true, y_test_prob):
    """Beta calibration: fits a logistic regression on log-odds features."""
    eps = 1e-8
    train_lo = np.log(np.clip(y_train_prob, eps, 1 - eps) /
                      (1 - np.clip(y_train_prob, eps, 1 - eps)))
    test_lo = np.log(np.clip(y_test_prob, eps, 1 - eps) /
                     (1 - np.clip(y_test_prob, eps, 1 - eps)))
    features_train = np.column_stack([train_lo, np.log(np.clip(y_train_prob, eps, 1))])
    features_test = np.column_stack([test_lo, np.log(np.clip(y_test_prob, eps, 1))])
    lr = LogisticRegression(max_iter=1000)
    lr.fit(features_train, y_train_true)
    return lr.predict_proba(features_test)[:, 1]


CALIBRATORS = {
    "Uncalibrated": None,
    "Platt": platt_scale,
    "Isotonic": isotonic_calibrate,
    "Beta": beta_calibrate,
}


# ============================================================
# METRICS
# ============================================================

def expected_calibration_error(y_true, y_prob, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(y_true) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece


def maximum_calibration_error(y_true, y_prob, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        mce = max(mce, abs(y_true[mask].mean() - y_prob[mask].mean()))
    return mce


def calibration_slope(y_true, y_prob):
    eps = 1e-8
    lo = np.clip(np.log(y_prob / (1 - y_prob + eps)), -10, 10).reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(lo, y_true)
    return float(lr.coef_[0][0])


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_single_experiment(X, y, dataset_name):
    """Run all models x calibrators on one dataset with k-fold CV."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for model_name, model_factory in get_models().items():
        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            model = copy.deepcopy(model_factory)
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            try:
                model.fit(X_tr, y_tr)
                y_prob_train = model.predict_proba(X_tr)[:, 1]
                y_prob_test = model.predict_proba(X_te)[:, 1]
            except Exception:
                continue

            for cal_name, cal_fn in CALIBRATORS.items():
                if cal_fn is None:
                    y_cal = y_prob_test
                else:
                    try:
                        y_cal = cal_fn(y_prob_train, y_tr.values, y_prob_test)
                    except Exception:
                        continue

                y_cal = np.clip(y_cal, 1e-8, 1 - 1e-8)

                results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "calibrator": cal_name,
                    "fold": fold_i,
                    "n_samples": len(X),
                    "prevalence": y.mean(),
                    "auc": roc_auc_score(y_te, y_cal),
                    "brier": brier_score_loss(y_te, y_cal),
                    "ece": expected_calibration_error(y_te.values, y_cal),
                    "mce": maximum_calibration_error(y_te.values, y_cal),
                    "log_loss": log_loss(y_te, y_cal),
                    "cal_slope": calibration_slope(y_te.values, y_cal),
                })

        print(f"    {model_name}: done")

    return results


def compute_degradation_analysis(df):
    """For each (dataset, model), compute whether calibration helped or hurt."""
    records = []
    for (ds, model), grp in df.groupby(["dataset", "model"]):
        uncal = grp[grp["calibrator"] == "Uncalibrated"]
        if len(uncal) == 0:
            continue
        ece_uncal = uncal["ece"].values

        for cal_name in ["Platt", "Isotonic", "Beta"]:
            cal = grp[grp["calibrator"] == cal_name]
            if len(cal) == 0:
                continue
            ece_cal = cal["ece"].values

            n = min(len(ece_uncal), len(ece_cal))
            if n < 3:
                continue

            diff = ece_cal[:n] - ece_uncal[:n]
            t_stat, p_val = stats.ttest_rel(ece_cal[:n], ece_uncal[:n])
            mean_diff = diff.mean()
            pct_change = (ece_cal[:n].mean() / ece_uncal[:n].mean() - 1) * 100

            if p_val < 0.05:
                if mean_diff < 0:
                    verdict = "HELPS"
                else:
                    verdict = "HURTS"
            else:
                verdict = "NO_EFFECT"

            records.append({
                "dataset": ds,
                "model": model,
                "calibrator": cal_name,
                "ece_uncal_mean": ece_uncal[:n].mean(),
                "ece_cal_mean": ece_cal[:n].mean(),
                "ece_pct_change": pct_change,
                "t_stat": t_stat,
                "p_value": p_val,
                "verdict": verdict,
                "n_folds": n,
            })

    return pd.DataFrame(records)


def run_all():
    print("=" * 65)
    print("Calibration in Clinical Risk Prediction -- Multi-Dataset Study")
    print("=" * 65)

    print("\n[1/4] Loading datasets...")
    datasets = get_all_datasets()

    print(f"\n[2/4] Running experiments ({len(datasets)} datasets x "
          f"{len(get_models())} models x {len(CALIBRATORS)} calibrators x "
          f"{N_SPLITS}-fold CV)...")
    all_results = []
    for X, y, name in datasets:
        print(f"\n  Dataset: {name}")
        results = run_single_experiment(X, y, name)
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_DIR / "all_results.csv", index=False)
    print(f"\n  Total experiment runs: {len(df)}")

    print("\n[3/4] Computing degradation analysis...")
    degradation = compute_degradation_analysis(df)
    degradation.to_csv(OUTPUT_DIR / "degradation_analysis.csv", index=False)

    # Summary
    print("\n" + "=" * 65)
    print("CALIBRATION VERDICT SUMMARY")
    print("=" * 65)
    if len(degradation) > 0:
        pivot = degradation.pivot_table(
            index=["dataset", "model"],
            columns="calibrator",
            values="verdict",
            aggfunc="first",
        )
        print(pivot.to_string())

        print("\n\nOverall verdict counts:")
        print(degradation["verdict"].value_counts().to_string())

        pct_hurts = (degradation["verdict"] == "HURTS").mean() * 100
        print(f"\nCalibration HURTS in {pct_hurts:.1f}% of (dataset, model, calibrator) combinations")

    print("\n[4/4] Computing summary statistics...")
    summary = df.groupby(["dataset", "model", "calibrator"]).agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ece_mean=("ece", "mean"), ece_std=("ece", "std"),
        brier_mean=("brier", "mean"), brier_std=("brier", "std"),
        mce_mean=("mce", "mean"),
        cal_slope_mean=("cal_slope", "mean"),
    ).round(4)
    summary.to_csv(OUTPUT_DIR / "summary_stats.csv")

    print(f"\n[OK] All results saved to {OUTPUT_DIR}/")
    return df, degradation


if __name__ == "__main__":
    run_all()

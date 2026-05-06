"""
Phase 4 - Cross-domain validation: Runner -> PMData
Train RF-Common on Runner dataset (6 shared features), evaluate via LOAO on PMData.
Target: meta AUC >= 0.55
"""

import logging
import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

WORKSPACE = os.path.dirname(os.path.abspath(__file__))

# 6 shared features: Runner name -> PMData name
FEATURE_MAP = {
    "acwr": "acwr",
    "session_load_proxy": "session_load",
    "mean_perceived_exertion": "fatigue",
    "mean_perceived_recovery": "readiness",
    "mean_perceived_success": "mood",
    "high_intensity_km_7d": "trimp_7d_sum",
}
RUNNER_COMMON = list(FEATURE_MAP.keys())
PMDATA_COMMON = list(FEATURE_MAP.values())

RUNNER_PROCESSED = os.path.join(WORKSPACE, "src", "outputs", "runner_dataset_processed.csv")
PMDATA_CSV = os.path.join(WORKSPACE, "notebooks", "outputs", "dataset_features_sin_normalizar.csv")
OUTPUT_CSV = os.path.join(WORKSPACE, "src", "outputs", "loao_crossdomain_pmdata.csv")

SEED = 42
TARGET_RATIO = 0.15

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
)


def load_runner_common():
    df = pd.read_csv(RUNNER_PROCESSED)
    # Keep only common runner features + target
    df = df[["participant_id"] + RUNNER_COMMON + ["injury"]].dropna()
    return df


def load_pmdata_common():
    df = pd.read_csv(PMDATA_CSV)
    # Rename PMData features to runner feature names for uniform handling
    rename = {v: k for k, v in FEATURE_MAP.items()}
    df = df[["participant_id", "is_injured"] + PMDATA_COMMON].copy()
    df = df.rename(columns=rename)
    df = df.rename(columns={"is_injured": "injury"})
    df["injury"] = df["injury"].astype(int)
    df = df.dropna(subset=RUNNER_COMMON)
    return df


def fit_normalizer_on_runner(X_runner):
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    pt.fit(X_runner)
    return pt


def apply_smote_safe(X, y, ratio=TARGET_RATIO, seed=SEED):
    pos = y.sum()
    neg = (y == 0).sum()
    if pos < 2 or neg < 2:
        return X, y
    desired_pos = int(neg * ratio)
    if desired_pos <= pos:
        return X, y
    k = min(5, int(pos) - 1)
    if k < 1:
        return X, y
    try:
        sm = SMOTE(sampling_strategy={1: desired_pos}, k_neighbors=k, random_state=seed)
        X_res, y_res = sm.fit_resample(X, y)
        return X_res, y_res
    except Exception:
        return X, y


def loao_pmdata(runner_df, pmdata_df):
    """Leave-one-athlete-out CV on PMData using model trained on Runner + (PMData - athlete).

    Strategy:
    - Train Runner-only RF-Common model (consistent features)
    - For each PMData athlete: evaluate directly (zero-shot cross-domain)
    """
    # 1. Fit normalizer on full Runner training data
    X_runner = runner_df[RUNNER_COMMON].values
    y_runner = runner_df["injury"].values

    log.info("Fitting normalizer on Runner dataset (%d rows)", len(runner_df))
    pt = fit_normalizer_on_runner(X_runner)
    X_runner_norm = pt.transform(X_runner)

    # 2. Augment runner training data
    X_runner_aug, y_runner_aug = apply_smote_safe(X_runner_norm, y_runner)
    log.info("Runner train after SMOTE: %d rows (%.1f%% positive)",
             len(y_runner_aug), 100 * y_runner_aug.mean())

    # 3. Train single RF on Runner data (all features = RUNNER_COMMON)
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_runner_aug, y_runner_aug)
    log.info("RF-Common trained on Runner dataset")

    # 4. LOAO over PMData athletes
    athletes = sorted(pmdata_df["participant_id"].unique())
    results = []

    for i, ath in enumerate(athletes):
        test_df = pmdata_df[pmdata_df["participant_id"] == ath]
        X_test = pt.transform(test_df[RUNNER_COMMON].values)
        y_test = test_df["injury"].values

        n_pos = y_test.sum()
        n_samples = len(y_test)

        if n_pos == 0:
            log.info("Fold %d/%d [%s]: SKIP (0 injuries, n=%d)", i + 1, len(athletes), ath, n_samples)
            results.append({
                "participant_id": ath,
                "n_samples": n_samples,
                "n_injuries": 0,
                "roc_auc": None,
                "skipped": True,
            })
            continue

        y_prob = rf.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = None

        log.info("Fold %d/%d [%s]: AUC=%.4f (n=%d, injuries=%d)",
                 i + 1, len(athletes), ath, auc if auc else 0, n_samples, int(n_pos))
        results.append({
            "participant_id": ath,
            "n_samples": n_samples,
            "n_injuries": int(n_pos),
            "roc_auc": round(auc, 4) if auc else None,
            "skipped": False,
        })

    return results


def main():
    log.info("=== PHASE 4: CROSS-DOMAIN VALIDATION (Runner -> PMData) ===")

    runner_df = load_runner_common()
    log.info("Runner dataset loaded: %d rows, features=%s", len(runner_df), RUNNER_COMMON)

    pmdata_df = load_pmdata_common()
    log.info("PMData loaded: %d rows, %d athletes", len(pmdata_df), pmdata_df["participant_id"].nunique())

    results = loao_pmdata(runner_df, pmdata_df)

    df_res = pd.DataFrame(results)
    valid = df_res[~df_res["skipped"]]
    n_valid = len(valid)

    if n_valid > 0:
        mean_auc = valid["roc_auc"].mean()
        std_auc = valid["roc_auc"].std()
    else:
        mean_auc = std_auc = 0.0

    # Append summary rows
    mean_row = pd.DataFrame([{"participant_id": "MEAN", "n_samples": None, "n_injuries": None,
                               "roc_auc": round(mean_auc, 4) if mean_auc else None, "skipped": False}])
    std_row = pd.DataFrame([{"participant_id": "STD", "n_samples": None, "n_injuries": None,
                              "roc_auc": round(std_auc, 4) if std_auc else None, "skipped": False}])
    df_res = pd.concat([df_res, mean_row, std_row], ignore_index=True)
    df_res.to_csv(OUTPUT_CSV, index=False)

    log.info("=== SUMMARY ===")
    log.info("Valid folds: %d / %d", n_valid, len(results))
    log.info("Cross-domain LOAO AUC = %.4f +/- %.4f", mean_auc, std_auc)
    log.info("Saved to %s", OUTPUT_CSV)

    if mean_auc >= 0.55:
        log.info("META >= 0.55: CUMPLIDA")
    else:
        log.info("META >= 0.55: NO cumplida (%.4f)", mean_auc)

    if mean_auc >= 0.65:
        log.info("META >= 0.65: CUMPLIDA (bonus)")


if __name__ == "__main__":
    main()

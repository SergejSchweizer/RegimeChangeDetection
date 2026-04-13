"""
Utilities for regime modeling with a Hidden Markov Model plus XGBoost.

The module supports a full workflow for time-ordered financial data:
1. split a feature frame into train/validation/test segments
2. fit a Gaussian HMM on a chosen subset of features
3. convert HMM state probabilities into additional predictive features
4. train a baseline XGBoost model and an HMM-augmented ensemble model
5. compare both models out of sample
6. search over multiple HMM/XGBoost feature-subset combinations

The design assumes a chronological dataset where rows are already sorted
in time order.
"""

import itertools
import warnings
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss
from xgboost import XGBClassifier
from hmmlearn.hmm import GaussianHMM
from tqdm.auto import tqdm


warnings.filterwarnings("ignore")


# =========================================================
# 1) SPLITTING
# =========================================================

def make_time_splits(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into chronological train, validation, and test sets.

    Parameters
    ----------
    df:
        Source dataframe ordered in time.
    train_frac:
        Fraction of rows assigned to the training segment.
    val_frac:
        Fraction of rows assigned to the validation segment. The remaining
        rows are assigned to the test segment.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(df_train, df_val, df_test)`` in chronological order.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    return df_train, df_val, df_test


# =========================================================
# 2) FEATURE SUBSETS
# =========================================================

def generate_feature_subsets(
    features: List[str],
    min_size: int,
    max_size: int
) -> List[List[str]]:
    """
    Generate all feature combinations between ``min_size`` and ``max_size``.

    Parameters
    ----------
    features:
        Candidate feature names.
    min_size:
        Smallest subset size to include.
    max_size:
        Largest subset size to include.

    Returns
    -------
    list[list[str]]
        Every combination of candidate features in the requested size range.
    """
    subsets: List[List[str]] = []
    for k in range(min_size, max_size + 1):
        subsets.extend([list(c) for c in itertools.combinations(features, k)])
    return subsets


# =========================================================
# 3) BASIC PREPROCESSING
# =========================================================

def clean_feature_frame(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Select a feature subset and normalize invalid numeric values.

    Parameters
    ----------
    df:
        Source dataframe.
    feature_cols:
        Columns to extract.

    Returns
    -------
    pd.DataFrame
        Copy of the selected columns with ``+/-inf`` replaced by ``NaN``.

    Raises
    ------
    ValueError
        If any requested feature column is missing.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df[feature_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def compute_entropy(prob_matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute row-wise entropy for a probability matrix.

    Parameters
    ----------
    prob_matrix:
        Array whose rows contain probability vectors.
    eps:
        Small constant added inside ``log`` for numerical stability.

    Returns
    -------
    np.ndarray
        Entropy value for each row.
    """
    return -(prob_matrix * np.log(prob_matrix + eps)).sum(axis=1)


# =========================================================
# 4) HMM
# =========================================================

def fit_hmm(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    n_states: int = 3,
    covariance_type: str = "full",
    n_iter: int = 200,
    random_state: int = 42
) -> Tuple[GaussianHMM, StandardScaler]:
    """
    Fit a Gaussian HMM on the training split.

    Parameters
    ----------
    df_train:
        Training dataframe.
    feature_cols:
        Columns used as HMM inputs.
    n_states:
        Number of latent HMM states.
    covariance_type:
        Covariance parameterization passed to ``GaussianHMM``.
    n_iter:
        Maximum number of EM iterations.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    tuple[GaussianHMM, StandardScaler]
        Fitted HMM plus the scaler used to standardize HMM inputs.

    Raises
    ------
    ValueError
        If no rows remain after dropping missing values.
    """
    x_df = clean_feature_frame(df_train, feature_cols).dropna()

    if x_df.empty:
        raise ValueError("No valid rows for HMM fitting after dropna().")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_df)

    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state
    )
    hmm.fit(x_scaled)

    return hmm, scaler


def add_hmm_features(
    df: pd.DataFrame,
    hmm: GaussianHMM,
    scaler: StandardScaler,
    feature_cols: List[str],
    prefix: str = "hmm"
) -> pd.DataFrame:
    """
    Generate HMM-derived features aligned to the input dataframe index.

    Parameters
    ----------
    df:
        Dataframe to transform.
    hmm:
        Fitted HMM model.
    scaler:
        Scaler fitted on the HMM training data.
    feature_cols:
        Raw feature columns expected by the HMM.
    prefix:
        Prefix used for generated output columns.

    Returns
    -------
    pd.DataFrame
        Dataframe containing:
        ``{prefix}_state``, ``{prefix}_prob_k``, ``{prefix}_max_prob``,
        and ``{prefix}_entropy``.
    """
    x_df = clean_feature_frame(df, feature_cols)
    valid_mask = x_df.notna().all(axis=1)

    out = pd.DataFrame(index=df.index)

    out[f"{prefix}_state"] = np.nan
    for k in range(hmm.n_components):
        out[f"{prefix}_prob_{k}"] = np.nan
    out[f"{prefix}_max_prob"] = np.nan
    out[f"{prefix}_entropy"] = np.nan

    if valid_mask.sum() == 0:
        return out

    x_valid = x_df.loc[valid_mask]
    x_scaled = scaler.transform(x_valid)

    states = hmm.predict(x_scaled)
    probs = hmm.predict_proba(x_scaled)

    out.loc[valid_mask, f"{prefix}_state"] = states
    for k in range(hmm.n_components):
        out.loc[valid_mask, f"{prefix}_prob_{k}"] = probs[:, k]

    out.loc[valid_mask, f"{prefix}_max_prob"] = probs.max(axis=1)
    out.loc[valid_mask, f"{prefix}_entropy"] = compute_entropy(probs)

    return out


# =========================================================
# 5) TARGET CREATION
# =========================================================

def create_target_from_hmm_state(
    df: pd.DataFrame,
    state_col: str = "hmm_state",
    mode: str = "next_state"
) -> pd.DataFrame:
    """
    Create a supervised target from inferred HMM states.

    Parameters
    ----------
    df:
        Input dataframe containing an HMM state column.
    state_col:
        Name of the state column to transform into a target.
    mode:
        ``"next_state"`` predicts the next state's label.
        ``"state_change"`` predicts whether the next row changes state.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with a new ``target`` column.
    """
    df = df.copy()

    if mode == "next_state":
        df["target"] = df[state_col].shift(-1)

    elif mode == "state_change":
        next_state = df[state_col].shift(-1)
        df["target"] = (df[state_col] != next_state).astype(float)
        df.loc[df[state_col].isna() | next_state.isna(), "target"] = np.nan

    else:
        raise ValueError("mode must be 'next_state' or 'state_change'")

    return df


# =========================================================
# 6) BUILD X / Y
# =========================================================

def build_xy(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Assemble model features and target, dropping incomplete rows.

    Parameters
    ----------
    df:
        Source dataframe.
    feature_cols:
        Feature columns for the design matrix.
    target_col:
        Name of the target column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(x, y)`` ready for model fitting or evaluation.
    """
    cols = feature_cols + [target_col]
    tmp = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    if tmp.empty:
        raise ValueError("No valid rows left after build_xy dropna().")

    x = tmp[feature_cols]
    y = tmp[target_col]

    return x, y


# =========================================================
# 7) LABEL ENCODING FOR XGBOOST
# =========================================================

def encode_labels_from_train(
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series
) -> Tuple[pd.Series, pd.Series, pd.Series, Dict]:
    """
    Encode labels into consecutive integers using training classes only.

    Parameters
    ----------
    y_train, y_val, y_test:
        Target series for the train, validation, and test splits.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series, dict]
        Encoded train/validation/test targets plus the class mapping.

    Raises
    ------
    ValueError
        If the training split contains fewer than two classes or if
        validation/test splits contain unseen labels.
    """
    train_classes = sorted(pd.Series(y_train).dropna().unique())
    if len(train_classes) < 2:
        raise ValueError("Training target has fewer than 2 classes.")

    class_mapping = {cls: i for i, cls in enumerate(train_classes)}

    unseen_val = set(pd.Series(y_val).dropna().unique()) - set(train_classes)
    unseen_test = set(pd.Series(y_test).dropna().unique()) - set(train_classes)

    if unseen_val:
        raise ValueError(f"Validation contains unseen classes: {sorted(unseen_val)}")
    if unseen_test:
        raise ValueError(f"Test contains unseen classes: {sorted(unseen_test)}")

    y_train_enc = pd.Series(y_train).map(class_mapping).astype(int)
    y_val_enc = pd.Series(y_val).map(class_mapping).astype(int)
    y_test_enc = pd.Series(y_test).map(class_mapping).astype(int)

    return y_train_enc, y_val_enc, y_test_enc, class_mapping


# =========================================================
# 8) XGBOOST
# =========================================================

def fit_xgb_classifier(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42
) -> XGBClassifier:
    """
    Fit an XGBoost classifier with automatic binary/multiclass handling.

    Parameters
    ----------
    x_train, y_train:
        Training features and encoded target labels.
    x_val, y_val:
        Validation features and labels used in the evaluation set.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    XGBClassifier
        Fitted classifier.
    """
    classes = np.sort(np.unique(y_train))
    n_classes = len(classes)

    if n_classes < 2:
        raise ValueError("Training target has fewer than 2 classes.")

    common_params = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
        tree_method="hist"
    )

    if n_classes == 2:
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **common_params
        )
    else:
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            **common_params
        )

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False
    )

    return model


def evaluate_classifier(
    model: XGBClassifier,
    x: pd.DataFrame,
    y: pd.Series
) -> Dict[str, float]:
    """
    Evaluate a fitted classifier on a labeled dataset.

    Parameters
    ----------
    model:
        Fitted XGBoost classifier.
    x:
        Feature matrix.
    y:
        True encoded labels.

    Returns
    -------
    dict[str, float]
        Dictionary containing accuracy, macro-F1, and log-loss.
    """
    y_pred = model.predict(x)
    y_prob = model.predict_proba(x)

    out = {
        "accuracy": accuracy_score(y, y_pred),
        "f1_macro": f1_score(y, y_pred, average="macro")
    }

    try:
        out["logloss"] = log_loss(y, y_prob)
    except Exception:
        out["logloss"] = np.nan

    return out


# =========================================================
# 9) ONE FULL PIPELINE EVALUATION
# =========================================================

def evaluate_pipeline(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    hmm_feature_cols: List[str],
    xgb_feature_cols: List[str],
    n_states: int = 3,
    target_mode: str = "next_state",
    hmm_prefix: str = "hmm",
    random_state: int = 42
) -> Dict:
    """
    Evaluate one HMM/XGBoost configuration end to end.

    The function fits an HMM on the training data, derives regime features,
    constructs targets, trains both a baseline and an HMM-augmented XGBoost
    model, and returns their validation/test metrics plus comparison deltas.

    Parameters
    ----------
    df_train, df_val, df_test:
        Chronological train/validation/test splits.
    hmm_feature_cols:
        Raw features used by the HMM.
    xgb_feature_cols:
        Raw features used by the XGBoost models.
    n_states:
        Number of latent HMM states.
    target_mode:
        Target construction mode passed to
        ``create_target_from_hmm_state``.
    hmm_prefix:
        Prefix used for generated HMM feature columns.
    random_state:
        Random seed.

    Returns
    -------
    dict
        Metrics, feature sets, label mappings, and transition matrix for
        this configuration.
    """
    # Fit HMM
    hmm, scaler = fit_hmm(
        df_train=df_train,
        feature_cols=hmm_feature_cols,
        n_states=n_states,
        random_state=random_state
    )

    # Add HMM features
    train_hmm = add_hmm_features(df_train, hmm, scaler, hmm_feature_cols, prefix=hmm_prefix)
    val_hmm = add_hmm_features(df_val, hmm, scaler, hmm_feature_cols, prefix=hmm_prefix)
    test_hmm = add_hmm_features(df_test, hmm, scaler, hmm_feature_cols, prefix=hmm_prefix)

    train_df = df_train.join(train_hmm)
    val_df = df_val.join(val_hmm)
    test_df = df_test.join(test_hmm)

    # Create target
    train_df = create_target_from_hmm_state(train_df, state_col=f"{hmm_prefix}_state", mode=target_mode)
    val_df = create_target_from_hmm_state(val_df, state_col=f"{hmm_prefix}_state", mode=target_mode)
    test_df = create_target_from_hmm_state(test_df, state_col=f"{hmm_prefix}_state", mode=target_mode)

    # The ensemble model gets the original XGB features plus soft regime
    # probabilities and uncertainty summaries from the HMM.
    hmm_output_cols = [f"{hmm_prefix}_prob_{k}" for k in range(n_states)] + [
        f"{hmm_prefix}_max_prob",
        f"{hmm_prefix}_entropy"
    ]

    baseline_cols = xgb_feature_cols.copy()
    ensemble_cols = xgb_feature_cols + hmm_output_cols

    # -------------------------
    # Baseline model
    # -------------------------
    x_train_base, y_train_base = build_xy(train_df, baseline_cols)
    x_val_base, y_val_base = build_xy(val_df, baseline_cols)
    x_test_base, y_test_base = build_xy(test_df, baseline_cols)

    y_train_base_enc, y_val_base_enc, y_test_base_enc, baseline_mapping = encode_labels_from_train(
        y_train_base, y_val_base, y_test_base
    )

    baseline_model = fit_xgb_classifier(
        x_train=x_train_base,
        y_train=y_train_base_enc,
        x_val=x_val_base,
        y_val=y_val_base_enc,
        random_state=random_state
    )

    baseline_val = evaluate_classifier(baseline_model, x_val_base, y_val_base_enc)
    baseline_test = evaluate_classifier(baseline_model, x_test_base, y_test_base_enc)

    # -------------------------
    # Ensemble model
    # -------------------------
    x_train_ens, y_train_ens = build_xy(train_df, ensemble_cols)
    x_val_ens, y_val_ens = build_xy(val_df, ensemble_cols)
    x_test_ens, y_test_ens = build_xy(test_df, ensemble_cols)

    y_train_ens_enc, y_val_ens_enc, y_test_ens_enc, ensemble_mapping = encode_labels_from_train(
        y_train_ens, y_val_ens, y_test_ens
    )

    ensemble_model = fit_xgb_classifier(
        x_train=x_train_ens,
        y_train=y_train_ens_enc,
        x_val=x_val_ens,
        y_val=y_val_ens_enc,
        random_state=random_state
    )

    ensemble_val = evaluate_classifier(ensemble_model, x_val_ens, y_val_ens_enc)
    ensemble_test = evaluate_classifier(ensemble_model, x_test_ens, y_test_ens_enc)

    # HMM train log-likelihood
    hmm_train_frame = clean_feature_frame(df_train, hmm_feature_cols).dropna()
    hmm_train_scaled = scaler.transform(hmm_train_frame)
    hmm_train_loglik = float(hmm.score(hmm_train_scaled))

    return {
        "hmm_feature_cols": hmm_feature_cols,
        "xgb_feature_cols": xgb_feature_cols,

        "baseline_val_accuracy": baseline_val["accuracy"],
        "baseline_val_f1": baseline_val["f1_macro"],
        "baseline_val_logloss": baseline_val["logloss"],

        "ensemble_val_accuracy": ensemble_val["accuracy"],
        "ensemble_val_f1": ensemble_val["f1_macro"],
        "ensemble_val_logloss": ensemble_val["logloss"],

        "baseline_test_accuracy": baseline_test["accuracy"],
        "baseline_test_f1": baseline_test["f1_macro"],
        "baseline_test_logloss": baseline_test["logloss"],

        "ensemble_test_accuracy": ensemble_test["accuracy"],
        "ensemble_test_f1": ensemble_test["f1_macro"],
        "ensemble_test_logloss": ensemble_test["logloss"],

        "delta_val_accuracy": ensemble_val["accuracy"] - baseline_val["accuracy"],
        "delta_val_f1": ensemble_val["f1_macro"] - baseline_val["f1_macro"],
        "delta_val_logloss": baseline_val["logloss"] - ensemble_val["logloss"],

        "delta_test_accuracy": ensemble_test["accuracy"] - baseline_test["accuracy"],
        "delta_test_f1": ensemble_test["f1_macro"] - baseline_test["f1_macro"],
        "delta_test_logloss": baseline_test["logloss"] - ensemble_test["logloss"],

        "hmm_train_loglik": hmm_train_loglik,
        "transition_matrix": hmm.transmat_.copy(),
        "baseline_label_mapping": baseline_mapping,
        "ensemble_label_mapping": ensemble_mapping
    }


# =========================================================
# 10) AUTOMATIC JOINT FEATURE SEARCH
# =========================================================

def automatic_hmm_xgb_feature_selection(
    df: pd.DataFrame,
    candidate_features: List[str],
    hmm_min_size: int = 3,
    hmm_max_size: int = 5,
    xgb_min_size: int = 3,
    xgb_max_size: int = 5,
    n_states: int = 3,
    target_mode: str = "next_state",
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    random_state: int = 42,
    sort_by: str = "delta_val_f1",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Search over HMM and XGBoost feature-subset combinations.

    Parameters
    ----------
    df:
        Chronologically ordered source dataframe.
    candidate_features:
        Pool of raw features from which HMM and XGBoost subsets are built.
    hmm_min_size, hmm_max_size:
        Minimum and maximum subset sizes for HMM features.
    xgb_min_size, xgb_max_size:
        Minimum and maximum subset sizes for XGBoost features.
    n_states:
        Number of HMM states.
    target_mode:
        Target construction mode.
    train_frac, val_frac:
        Split proportions used by ``make_time_splits``.
    random_state:
        Random seed.
    sort_by:
        Column used to sort the resulting dataframe.
    verbose:
        If ``True``, show a ``tqdm`` progress bar during the search.

    Returns
    -------
    pd.DataFrame
        One row per successful configuration, sorted by ``sort_by``.
    """
    df_train, df_val, df_test = make_time_splits(
        df=df,
        train_frac=train_frac,
        val_frac=val_frac
    )

    hmm_subsets = generate_feature_subsets(candidate_features, hmm_min_size, hmm_max_size)
    xgb_subsets = generate_feature_subsets(candidate_features, xgb_min_size, xgb_max_size)

    results = []
    total = len(hmm_subsets) * len(xgb_subsets)
    progress = tqdm(
        total=total,
        disable=not verbose,
        desc="Feature search",
        unit="combo"
    )

    try:
        for hmm_subset in hmm_subsets:
            for xgb_subset in xgb_subsets:
                delta_val_f1 = None
                status = "ok"
                try:
                    result = evaluate_pipeline(
                        df_train=df_train,
                        df_val=df_val,
                        df_test=df_test,
                        hmm_feature_cols=hmm_subset,
                        xgb_feature_cols=xgb_subset,
                        n_states=n_states,
                        target_mode=target_mode,
                        random_state=random_state
                    )
                    results.append(result)
                    delta_val_f1 = float(result["delta_val_f1"])

                except Exception:
                    status = "skipped"

                progress.update(1)
                postfix = {"status": status}
                if delta_val_f1 is not None:
                    postfix["delta_val_f1"] = f"{delta_val_f1:.4f}"
                progress.set_postfix(postfix)
    finally:
        progress.close()

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df = results_df.sort_values(by=sort_by, ascending=False).reset_index(drop=True)

    return results_df


# =========================================================
# 11) REFIT BEST CONFIGURATION ON TRAIN+VAL
# =========================================================

def refit_best_pipeline(
    df: pd.DataFrame,
    best_hmm_feature_cols: List[str],
    best_xgb_feature_cols: List[str],
    n_states: int = 3,
    target_mode: str = "next_state",
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    hmm_prefix: str = "hmm",
    random_state: int = 42
) -> Dict:
    """
    Refit the selected configuration on train+validation and test once.

    Parameters
    ----------
    df:
        Full source dataframe in chronological order.
    best_hmm_feature_cols:
        Selected HMM feature subset.
    best_xgb_feature_cols:
        Selected XGBoost feature subset.
    n_states:
        Number of HMM states.
    target_mode:
        Target construction mode.
    train_frac, val_frac:
        Split proportions used to reconstruct the original train/val/test
        boundaries.
    hmm_prefix:
        Prefix used for generated HMM feature columns.
    random_state:
        Random seed.

    Returns
    -------
    dict
        Test metrics plus fitted artifacts for the refit configuration.
    """
    df_train, df_val, df_test = make_time_splits(df, train_frac=train_frac, val_frac=val_frac)
    df_trainval = pd.concat([df_train, df_val], axis=0).copy()

    hmm, scaler = fit_hmm(
        df_train=df_trainval,
        feature_cols=best_hmm_feature_cols,
        n_states=n_states,
        random_state=random_state
    )

    trainval_hmm = add_hmm_features(df_trainval, hmm, scaler, best_hmm_feature_cols, prefix=hmm_prefix)
    test_hmm = add_hmm_features(df_test, hmm, scaler, best_hmm_feature_cols, prefix=hmm_prefix)

    trainval_df = df_trainval.join(trainval_hmm)
    test_df = df_test.join(test_hmm)

    trainval_df = create_target_from_hmm_state(trainval_df, state_col=f"{hmm_prefix}_state", mode=target_mode)
    test_df = create_target_from_hmm_state(test_df, state_col=f"{hmm_prefix}_state", mode=target_mode)

    # Rebuild the same ensemble feature set used during model selection.
    hmm_output_cols = [f"{hmm_prefix}_prob_{k}" for k in range(n_states)] + [
        f"{hmm_prefix}_max_prob",
        f"{hmm_prefix}_entropy"
    ]

    baseline_cols = best_xgb_feature_cols.copy()
    ensemble_cols = best_xgb_feature_cols + hmm_output_cols

    # Baseline
    x_train_base, y_train_base = build_xy(trainval_df, baseline_cols)
    x_test_base, y_test_base = build_xy(test_df, baseline_cols)

    y_train_base_enc, y_test_base_enc, _, baseline_mapping = encode_labels_from_train(
        y_train_base, y_test_base, y_test_base
    )

    baseline_model = fit_xgb_classifier(
        x_train=x_train_base,
        y_train=y_train_base_enc,
        x_val=x_test_base,
        y_val=y_test_base_enc,
        random_state=random_state
    )

    baseline_test = evaluate_classifier(baseline_model, x_test_base, y_test_base_enc)

    # Ensemble
    x_train_ens, y_train_ens = build_xy(trainval_df, ensemble_cols)
    x_test_ens, y_test_ens = build_xy(test_df, ensemble_cols)

    y_train_ens_enc, y_test_ens_enc, _, ensemble_mapping = encode_labels_from_train(
        y_train_ens, y_test_ens, y_test_ens
    )

    ensemble_model = fit_xgb_classifier(
        x_train=x_train_ens,
        y_train=y_train_ens_enc,
        x_val=x_test_ens,
        y_val=y_test_ens_enc,
        random_state=random_state
    )

    ensemble_test = evaluate_classifier(ensemble_model, x_test_ens, y_test_ens_enc)

    return {
        "best_hmm_feature_cols": best_hmm_feature_cols,
        "best_xgb_feature_cols": best_xgb_feature_cols,

        "baseline_test_accuracy": baseline_test["accuracy"],
        "baseline_test_f1": baseline_test["f1_macro"],
        "baseline_test_logloss": baseline_test["logloss"],

        "ensemble_test_accuracy": ensemble_test["accuracy"],
        "ensemble_test_f1": ensemble_test["f1_macro"],
        "ensemble_test_logloss": ensemble_test["logloss"],

        "delta_test_accuracy": ensemble_test["accuracy"] - baseline_test["accuracy"],
        "delta_test_f1": ensemble_test["f1_macro"] - baseline_test["f1_macro"],
        "delta_test_logloss": baseline_test["logloss"] - ensemble_test["logloss"],

        "transition_matrix": hmm.transmat_.copy(),
        "baseline_label_mapping": baseline_mapping,
        "ensemble_label_mapping": ensemble_mapping,
        "hmm_model": hmm,
        "hmm_scaler": scaler,
        "baseline_model": baseline_model,
        "ensemble_model": ensemble_model
    }


# =========================================================
# 12) OPTIONAL SMALL HELPER TO READ BEST ROW
# =========================================================

def extract_best_feature_sets(results_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Extract the best HMM and XGBoost feature subsets from search results.

    Parameters
    ----------
    results_df:
        Output of ``automatic_hmm_xgb_feature_selection`` sorted with the
        preferred configuration in the first row.

    Returns
    -------
    tuple[list[str], list[str]]
        Best HMM feature list and best XGBoost feature list.
    """
    if results_df.empty:
        raise ValueError("results_df is empty.")

    best_hmm_feature_cols = results_df.iloc[0]["hmm_feature_cols"]
    best_xgb_feature_cols = results_df.iloc[0]["xgb_feature_cols"]

    return best_hmm_feature_cols, best_xgb_feature_cols

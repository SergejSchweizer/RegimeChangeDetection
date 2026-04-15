"""
hmm_utils.py

Utilities for selecting input features for Hidden Markov Models (HMMs)
in market regime modeling.

Scope
-----
This module is intentionally limited to HMM-side feature selection only.

It supports:
1. chronological train / validation / test splitting
2. feature cleaning
3. Gaussian HMM fitting on candidate feature subsets
4. generation of HMM regime features
5. unsupervised scoring of HMM feature subsets using a regime-quality criterion
6. automatic search for strong HMM input subsets

Selection principle
-------------------
The best feature subset and number of hidden states are selected using an
unsupervised regime-quality criterion, subject to:

- convergence of the estimation algorithm
- a minimum occupancy constraint for each hidden state

Among admissible models, preference is given to HMMs that produce:
- persistent hidden states
- low posterior uncertainty
- economically interpretable regime durations
- balanced state usage

Important
---------
This file does NOT contain:
- XGBoost
- supervised market target creation
- downstream predictive evaluation

That comes later.

Typical use
-----------
Use this module to find HMM feature subsets that produce:
- statistically well-fitted latent-state models
- non-degenerate hidden states
- interpretable regime structure
"""

import itertools
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
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
        Time-ordered dataframe.
    train_frac:
        Fraction used for training.
    val_frac:
        Fraction used for validation. Remaining rows go to test.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (df_train, df_val, df_test)
    """
    n = len(df)

    if n == 0:
        raise ValueError("Input dataframe is empty.")
    if not (0 < train_frac < 1):
        raise ValueError("train_frac must lie in (0, 1).")
    if not (0 < val_frac < 1):
        raise ValueError("val_frac must lie in (0, 1).")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1.")

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    if df_train.empty or df_val.empty or df_test.empty:
        raise ValueError("One of the splits is empty. Adjust split fractions or provide more data.")

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
    Generate all feature combinations between min_size and max_size.

    Parameters
    ----------
    features:
        Candidate feature names.
    min_size:
        Minimum subset size.
    max_size:
        Maximum subset size.

    Returns
    -------
    list[list[str]]
        All combinations in the requested size range.
    """
    if min_size < 1:
        raise ValueError("min_size must be >= 1.")
    if max_size < min_size:
        raise ValueError("max_size must be >= min_size.")
    if max_size > len(features):
        raise ValueError("max_size cannot exceed number of features.")

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
    Select feature columns and replace +/-inf with NaN.

    Parameters
    ----------
    df:
        Source dataframe.
    feature_cols:
        Requested columns.

    Returns
    -------
    pd.DataFrame
        Cleaned feature frame.
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
        2D array of probabilities.
    eps:
        Small numerical stability constant.

    Returns
    -------
    np.ndarray
        Entropy per row.
    """
    prob_matrix = np.asarray(prob_matrix)
    if prob_matrix.ndim != 2:
        raise ValueError("prob_matrix must be 2-dimensional.")

    return -(prob_matrix * np.log(prob_matrix + eps)).sum(axis=1)


def filter_high_correlation_features(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.95
) -> List[str]:
    """
    Remove highly correlated features using train data only.

    The rule is deterministic: when two features exceed the threshold,
    the later one in the column order is dropped.

    Parameters
    ----------
    df_train:
        Training dataframe.
    feature_cols:
        Candidate features.
    threshold:
        Absolute correlation threshold.

    Returns
    -------
    list[str]
        Filtered feature list.
    """
    if not feature_cols:
        raise ValueError("feature_cols is empty.")
    if not (0 < threshold <= 1):
        raise ValueError("threshold must lie in (0, 1].")

    x_corr = clean_feature_frame(df_train, feature_cols).dropna()
    if x_corr.empty:
        return feature_cols.copy()

    corr = x_corr.corr().abs()
    to_drop = set()
    cols = list(corr.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr.iloc[i, j] >= threshold:
                to_drop.add(cols[j])

    return [c for c in feature_cols if c not in to_drop]


def _flatten_columns_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns into single strings.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    out = df.copy()
    out.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_")
        for col in out.columns
    ]
    return out


def _format_feature_list_for_tqdm(feature_cols: List[str], max_len: int = 60) -> str:
    """
    Format feature list into a compact string for tqdm postfix.
    """
    text = ",".join(feature_cols)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# =========================================================
# 4) HMM FITTING
# =========================================================

def fit_hmm(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    n_states: int = 3,
    covariance_type: str = "diag",
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
        HMM input columns.
    n_states:
        Number of hidden states.
    covariance_type:
        Covariance type for hmmlearn GaussianHMM.
    n_iter:
        Maximum EM iterations.
    random_state:
        Random seed.

    Returns
    -------
    tuple[GaussianHMM, StandardScaler]
        Fitted HMM and fitted scaler.
    """
    if n_states < 2:
        raise ValueError("n_states must be >= 2.")
    if not feature_cols:
        raise ValueError("feature_cols must not be empty.")

    x_df = clean_feature_frame(df_train, feature_cols).dropna()
    if x_df.empty:
        raise ValueError("No valid rows for HMM fitting after dropna().")
    if len(x_df) <= n_states:
        raise ValueError("Not enough valid rows relative to number of states.")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_df)

    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state
    )
    hmm.fit(x_scaled)

    if hasattr(hmm, "monitor_") and not hmm.monitor_.converged:
        warnings.warn("HMM did not converge.")

    return hmm, scaler


def add_hmm_features(
    df: pd.DataFrame,
    hmm: GaussianHMM,
    scaler: StandardScaler,
    feature_cols: List[str],
    prefix: str = "hmm"
) -> pd.DataFrame:
    """
    Generate HMM-derived regime features aligned to the dataframe index.

    Output columns
    --------------
    - {prefix}_state
    - {prefix}_prob_k
    - {prefix}_max_prob
    - {prefix}_entropy
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
# 5) HMM DIAGNOSTICS
# =========================================================

def _compute_state_sequence_stats(states: np.ndarray, n_states: int) -> Dict[str, float]:
    """
    Compute diagnostics for an inferred HMM state path.
    """
    states = np.asarray(states).astype(int)
    if len(states) == 0:
        raise ValueError("Empty state sequence.")

    counts = np.bincount(states, minlength=n_states)
    fractions = counts / counts.sum()

    run_lengths = []
    current_run = 1

    for i in range(1, len(states)):
        if states[i] == states[i - 1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)

    return {
        "min_state_fraction": float(fractions.min()),
        "max_state_fraction": float(fractions.max()),
        "median_run_length": float(np.median(run_lengths)),
        "mean_run_length": float(np.mean(run_lengths)),
        "n_runs": float(len(run_lengths)),
    }


def _count_hmm_parameters(n_states: int, n_features: int, covariance_type: str) -> int:
    """
    Rough parameter count for GaussianHMM, useful for normalized diagnostics.
    """
    if covariance_type not in {"diag", "full", "spherical", "tied"}:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    startprob_params = n_states - 1
    transmat_params = n_states * (n_states - 1)
    mean_params = n_states * n_features

    if covariance_type == "diag":
        cov_params = n_states * n_features
    elif covariance_type == "full":
        cov_params = n_states * (n_features * (n_features + 1) // 2)
    elif covariance_type == "spherical":
        cov_params = n_states
    else:  # tied
        cov_params = n_features * (n_features + 1) // 2

    return int(startprob_params + transmat_params + mean_params + cov_params)


def _safe_float(value: float, default: float = np.nan) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
        return default
    except Exception:
        return default


def evaluate_hmm_feature_subset(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    n_states: int = 3,
    covariance_type: str = "full",
    n_iter: int = 200,
    random_state: int = 42
) -> Dict:
    """
    Fit an HMM on one feature subset and compute regime diagnostics.

    Parameters
    ----------
    df_train:
        Training dataframe.
    feature_cols:
        Candidate HMM feature subset.
    n_states:
        Number of hidden states.
    covariance_type:
        Covariance type for GaussianHMM.
    n_iter:
        Maximum EM iterations.
    random_state:
        Random seed.

    Returns
    -------
    dict
        Model diagnostics for this subset.
    """
    hmm, scaler = fit_hmm(
        df_train=df_train,
        feature_cols=feature_cols,
        n_states=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state
    )

    x_df = clean_feature_frame(df_train, feature_cols).dropna()
    x_scaled = scaler.transform(x_df)

    states = hmm.predict(x_scaled)
    probs = hmm.predict_proba(x_scaled)

    seq_stats = _compute_state_sequence_stats(states, n_states=n_states)

    avg_self_transition = float(np.mean(np.diag(hmm.transmat_)))
    train_loglik = float(hmm.score(x_scaled))
    avg_entropy = float(compute_entropy(probs).mean())

    converged = True
    if hasattr(hmm, "monitor_") and hasattr(hmm.monitor_, "converged"):
        converged = bool(hmm.monitor_.converged)

    n_obs_used = int(len(x_df))
    n_features = int(len(feature_cols))
    n_params = _count_hmm_parameters(
        n_states=n_states,
        n_features=n_features,
        covariance_type=covariance_type
    )

    loglik_per_obs = train_loglik / n_obs_used
    loglik_per_obs_per_feature = train_loglik / (n_obs_used * n_features)
    aic = 2 * n_params - 2 * train_loglik
    bic = np.log(n_obs_used) * n_params - 2 * train_loglik

    return {
        "feature_cols": feature_cols,
        "n_states": n_states,
        "n_features": n_features,
        "n_obs_used": n_obs_used,
        "converged": converged,
        "train_loglik": train_loglik,
        "loglik_per_obs": float(loglik_per_obs),
        "loglik_per_obs_per_feature": float(loglik_per_obs_per_feature),
        "aic": float(aic),
        "bic": float(bic),
        "n_params": int(n_params),
        "avg_self_transition": avg_self_transition,
        "avg_entropy": avg_entropy,
        "min_state_fraction": seq_stats["min_state_fraction"],
        "max_state_fraction": seq_stats["max_state_fraction"],
        "median_run_length": seq_stats["median_run_length"],
        "mean_run_length": seq_stats["mean_run_length"],
        "n_runs": seq_stats["n_runs"],
        "transition_matrix": hmm.transmat_.copy()
    }


def simple_hmm_selection_score(
    row: pd.Series,
    min_state_fraction_threshold: float = 0.05
) -> float:
    """
    Regime-quality selection score for HMM feature subsets.

    Selection rule
    --------------
    Admissibility:
    - convergence
    - minimum state occupancy

    Ranking:
    - reward persistent regimes
    - reward balanced state usage
    - reward longer median run lengths
    - penalize posterior uncertainty
    - use normalized likelihood only as a weak tie-breaker

    Parameters
    ----------
    row:
        One diagnostics row.
    min_state_fraction_threshold:
        Minimum required share of observations in every hidden state.

    Returns
    -------
    float
        Selection score. Higher is better.
    """
    if not bool(row["converged"]):
        return -np.inf

    min_frac = _safe_float(row["min_state_fraction"], default=-np.inf)
    if not np.isfinite(min_frac) or min_frac < min_state_fraction_threshold:
        return -np.inf

    avg_self_transition = _safe_float(row["avg_self_transition"], default=-np.inf)
    avg_entropy = _safe_float(row["avg_entropy"], default=np.inf)
    median_run_length = _safe_float(row["median_run_length"], default=-np.inf)
    loglik_per_obs_per_feature = _safe_float(row.get("loglik_per_obs_per_feature", np.nan), default=0.0)

    if not np.isfinite(avg_self_transition) or not np.isfinite(avg_entropy) or not np.isfinite(median_run_length):
        return -np.inf

    score = (
        3.0 * avg_self_transition
        + 1.5 * min_frac
        - 0.25 * median_run_length
        - 2.5 * avg_entropy
        + 0.05 * loglik_per_obs_per_feature
    )
    return float(score)


# =========================================================
# 6) AUTOMATIC HMM FEATURE SEARCH
# =========================================================

def automatic_hmm_feature_selection(
    df: pd.DataFrame,
    candidate_features: List[str],
    subset_min_size: int = 1,
    subset_max_size: int = 3,
    n_states_list: List[int] = [2, 3],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    correlation_filter_threshold: Optional[float] = 0.95,
    min_state_fraction_threshold: float = 0.05,
    top_k: Optional[int] = 20,
    covariance_type: str = "full",
    n_iter: int = 200,
    random_state: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Automatically search for strong HMM feature subsets using a regime-quality
    selection rule.

    Procedure
    ---------
    1. keep only the train split
    2. optionally remove highly correlated candidates
    3. generate feature subsets
    4. fit HMM for each subset and state count
    5. rank by regime quality:
       - convergence
       - minimum state occupancy
       - persistence
       - posterior certainty
       - run-length structure

    Parameters
    ----------
    df:
        Full time-ordered dataframe.
    candidate_features:
        Candidate HMM input features.
    subset_min_size:
        Minimum subset size.
    subset_max_size:
        Maximum subset size.
    n_states_list:
        Hidden-state counts to evaluate.
    train_frac:
        Train fraction.
    val_frac:
        Validation fraction.
    correlation_filter_threshold:
        Optional correlation filter threshold on train data.
        Set to None to disable.
    min_state_fraction_threshold:
        Minimum required state occupancy for model admissibility.
    top_k:
        Keep only the top_k successful rows if not None.
    covariance_type:
        Covariance type for GaussianHMM.
    n_iter:
        Maximum EM iterations.
    random_state:
        Random seed.
    verbose:
        Whether to show a progress bar.

    Returns
    -------
    pd.DataFrame
        Ranked HMM subset diagnostics.
    """
    if not candidate_features:
        raise ValueError("candidate_features is empty.")
    if not n_states_list:
        raise ValueError("n_states_list is empty.")
    if subset_min_size < 1:
        raise ValueError("subset_min_size must be >= 1.")
    if subset_max_size < subset_min_size:
        raise ValueError("subset_max_size must be >= subset_min_size.")
    if not (0.0 <= min_state_fraction_threshold < 1.0):
        raise ValueError("min_state_fraction_threshold must lie in [0, 1).")

    df_train, _, _ = make_time_splits(df, train_frac=train_frac, val_frac=val_frac)

    filtered_features = candidate_features.copy()
    if correlation_filter_threshold is not None:
        filtered_features = filter_high_correlation_features(
            df_train=df_train,
            feature_cols=filtered_features,
            threshold=correlation_filter_threshold
        )

    if not filtered_features:
        raise ValueError("No features left after correlation filtering.")

    max_size_eff = min(subset_max_size, len(filtered_features))
    if subset_min_size > max_size_eff:
        raise ValueError("subset_min_size is larger than available filtered features.")

    subsets = generate_feature_subsets(
        filtered_features,
        subset_min_size,
        max_size_eff
    )

    total = len(subsets) * len(n_states_list)
    progress = tqdm(total=total, disable=not verbose, desc="HMM feature search", unit="fit")

    rows = []
    best_score = -np.inf
    best_desc = "None"

    try:
        for subset in subsets:
            for n_states in n_states_list:
                try:
                    diag = evaluate_hmm_feature_subset(
                        df_train=df_train,
                        feature_cols=subset,
                        n_states=n_states,
                        covariance_type=covariance_type,
                        n_iter=n_iter,
                        random_state=random_state
                    )

                    diag["selection_score"] = simple_hmm_selection_score(
                        pd.Series(diag),
                        min_state_fraction_threshold=min_state_fraction_threshold
                    )
                    diag["eligible"] = bool(np.isfinite(diag["selection_score"]))
                    diag["status"] = "ok"
                    diag["error"] = None
                    rows.append(diag)

                    score = diag["selection_score"]
                    if np.isfinite(score) and score > best_score:
                        best_score = score
                        best_desc = (
                            f"{_format_feature_list_for_tqdm(diag['feature_cols'])} | "
                            f"S={diag['avg_self_transition']:.2f} | "
                            f"R={diag['median_run_length']:.1f} | "
                            f"E={diag['avg_entropy']:.3f}"
                        )

                except Exception as e:
                    rows.append({
                        "feature_cols": subset,
                        "n_states": n_states,
                        "n_features": len(subset),
                        "n_obs_used": np.nan,
                        "converged": False,
                        "train_loglik": np.nan,
                        "loglik_per_obs": np.nan,
                        "loglik_per_obs_per_feature": np.nan,
                        "aic": np.nan,
                        "bic": np.nan,
                        "n_params": np.nan,
                        "avg_self_transition": np.nan,
                        "avg_entropy": np.nan,
                        "min_state_fraction": np.nan,
                        "max_state_fraction": np.nan,
                        "median_run_length": np.nan,
                        "mean_run_length": np.nan,
                        "n_runs": np.nan,
                        "transition_matrix": None,
                        "selection_score": -np.inf,
                        "eligible": False,
                        "status": "skipped",
                        "error": f"{type(e).__name__}: {e}"
                    })

                progress.update(1)
                if verbose:
                    if np.isfinite(best_score):
                        progress.set_postfix_str(f"best={best_score:.3f} | {best_desc}")
                    else:
                        progress.set_postfix_str("best=None")
    finally:
        progress.close()

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    ok_df = out[out["status"] == "ok"].copy()
    bad_df = out[out["status"] != "ok"].copy()

    if not ok_df.empty:
        ok_df = ok_df.sort_values(
            by=[
                "eligible",
                "selection_score",
                "avg_self_transition",
                "min_state_fraction",
                "median_run_length",
                "avg_entropy",
                "loglik_per_obs_per_feature",
            ],
            ascending=[False, False, False, False, False, True, False]
        ).reset_index(drop=True)

    if top_k is not None and top_k > 0:
        ok_df = ok_df.head(top_k)

    out = pd.concat([ok_df, bad_df], axis=0).reset_index(drop=True)
    return out


# =========================================================
# 7) HELPERS TO READ RESULTS
# =========================================================

def extract_best_hmm_feature_subset(
    results_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract the best HMM feature subset and corresponding number of states
    from automatic_hmm_feature_selection output.

    Parameters
    ----------
    results_df:
        Output dataframe from automatic_hmm_feature_selection.

    Returns
    -------
    pd.DataFrame
        One-row dataframe containing the best feature subset and best state count.
    """
    if results_df.empty:
        raise ValueError("results_df is empty.")

    ok_df = results_df[results_df["status"] == "ok"].copy()
    if ok_df.empty:
        raise ValueError("No successful rows in results_df.")

    eligible_df = ok_df[ok_df["eligible"] == True].copy()
    if eligible_df.empty:
        raise ValueError("No eligible rows found. All models failed the selection constraints.")

    best_row = eligible_df.iloc[[0]].copy()

    preferred_cols = [
        "feature_cols",
        "n_states",
        "n_features",
        "n_obs_used",
        "selection_score",
        "train_loglik",
        "loglik_per_obs",
        "loglik_per_obs_per_feature",
        "aic",
        "bic",
        "min_state_fraction",
        "avg_self_transition",
        "median_run_length",
        "avg_entropy",
        "eligible",
        "status"
    ]
    cols = [c for c in preferred_cols if c in best_row.columns]

    return best_row[cols].reset_index(drop=True)


def summarize_hmm_results(
    results_df: pd.DataFrame,
    top_n: int = 10,
    stringify_features: bool = False
) -> pd.DataFrame:
    """
    Return a compact summary view of the top HMM feature-selection results.

    Parameters
    ----------
    results_df:
        Output dataframe from automatic_hmm_feature_selection.
    top_n:
        Number of top successful rows to display.
    stringify_features:
        If True, convert feature_cols from list[str] to a printable string.

    Returns
    -------
    pd.DataFrame
        Compact summary table.
    """
    if results_df.empty:
        raise ValueError("results_df is empty.")

    ok_df = results_df[results_df["status"] == "ok"].copy()
    if ok_df.empty:
        raise ValueError("No successful rows in results_df.")

    cols = [
        "feature_cols",
        "n_states",
        "n_features",
        "n_obs_used",
        "eligible",
        "selection_score",
        "train_loglik",
        "loglik_per_obs",
        "loglik_per_obs_per_feature",
        "min_state_fraction",
        "avg_self_transition",
        "median_run_length",
        "avg_entropy",
        "aic",
        "bic",
    ]
    available_cols = [c for c in cols if c in ok_df.columns]

    summary_df = ok_df[available_cols].head(top_n).reset_index(drop=True)

    if stringify_features:
        summary_df["feature_cols"] = summary_df["feature_cols"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else str(x)
        )

    return summary_df


# =========================================================
# 8) FIT BEST MODEL CONVENIENCE FUNCTION
# =========================================================

def fit_best_hmm_from_results(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    covariance_type: str = "full",
    n_iter: int = 200,
    random_state: int = 42
) -> Tuple[GaussianHMM, StandardScaler, List[str], int]:
    """
    Fit the best HMM found by automatic_hmm_feature_selection on the train split.

    Parameters
    ----------
    df:
        Full time-ordered dataframe.
    results_df:
        Output of automatic_hmm_feature_selection.
    train_frac:
        Train fraction.
    val_frac:
        Validation fraction.
    covariance_type:
        Covariance type for GaussianHMM.
    n_iter:
        Maximum EM iterations.
    random_state:
        Random seed.

    Returns
    -------
    tuple
        (hmm, scaler, best_feature_cols, best_n_states)
    """
    best_df = extract_best_hmm_feature_subset(results_df)
    best_feature_cols = best_df.iloc[0]["feature_cols"]
    best_n_states = int(best_df.iloc[0]["n_states"])

    df_train, _, _ = make_time_splits(df, train_frac=train_frac, val_frac=val_frac)

    hmm, scaler = fit_hmm(
        df_train=df_train,
        feature_cols=best_feature_cols,
        n_states=best_n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state
    )

    return hmm, scaler, best_feature_cols, best_n_states


# =========================================================
# 9) REGIME INTERPRETATION
# =========================================================

def assign_regimes(
    df: pd.DataFrame,
    vol_col: str = "abs_return_close_perp_mean",
    activity_col: str = "volume_perp_mean"
) -> Tuple[Dict, Dict]:
    """
    Assign semantic labels to HMM states using volatility and activity.

    Parameters
    ----------
    df:
        DataFrame indexed by HMM state. May contain MultiIndex columns.
    vol_col:
        Column used as volatility proxy.
    activity_col:
        Column used as activity proxy.

    Returns
    -------
    tuple[dict, dict]
        (regime_labels, regime_colors)
    """
    df = _flatten_columns_if_needed(df).copy()

    for col in [vol_col, activity_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    if df[vol_col].isna().any() or df[activity_col].isna().any():
        raise ValueError("Regime interpretation columns contain NaN values.")

    df["regime_score"] = (
        df[vol_col].rank(method="first") +
        df[activity_col].rank(method="first")
    )

    ranked_states = df["regime_score"].sort_values().index.tolist()

    regime_labels: Dict = {}
    regime_colors: Dict = {}

    n = len(ranked_states)

    if n == 1:
        regime_labels[ranked_states[0]] = "Active"
        regime_colors[ranked_states[0]] = "lightgreen"
        return regime_labels, regime_colors

    if n == 2:
        regime_labels[ranked_states[0]] = "Low Activity"
        regime_colors[ranked_states[0]] = "lightgrey"
        regime_labels[ranked_states[1]] = "Stress"
        regime_colors[ranked_states[1]] = "lightcoral"
        return regime_labels, regime_colors

    if n == 3:
        regime_labels[ranked_states[0]] = "Low Activity"
        regime_colors[ranked_states[0]] = "lightgrey"

        regime_labels[ranked_states[1]] = "Active"
        regime_colors[ranked_states[1]] = "lightgreen"

        regime_labels[ranked_states[2]] = "Stress"
        regime_colors[ranked_states[2]] = "lightcoral"
        return regime_labels, regime_colors

    for i, state in enumerate(ranked_states):
        if i == 0:
            regime_labels[state] = "Low Activity"
            regime_colors[state] = "lightgrey"
        elif i == n - 1:
            regime_labels[state] = "Stress"
            regime_colors[state] = "lightcoral"
        else:
            regime_labels[state] = f"Active {i}"
            regime_colors[state] = "lightgreen"

    return regime_labels, regime_colors
# BTC Regime Change Analysis

This repository explores BTC market regime behavior using Deribit spot, perpetual, and funding-rate data. The codebase currently centers on two reusable utility modules:

- `deribit_utils.py` for data collection and dataset assembly
- `hmm_utils.py` for unsupervised Hidden Markov Model feature selection and regime diagnostics

The notebooks build on those utilities for exploratory analysis and future modeling work.

## Project Layout

- `deribit_utils.py`: fetches Deribit spot OHLCV, perpetual OHLCV, and funding-rate history, then merges them into a single time-indexed dataset
- `hmm_utils.py`: provides chronological splitting, feature cleaning, HMM fitting, HMM-derived feature generation, and automatic unsupervised HMM feature-subset search
- `deribit_data.csv`: base merged market dataset
- `deribit_enriched_data.csv`: engineered dataset used for regime analysis
- `00_plot_regime_change.ipynb`: exploratory notebook for feature engineering and volatility regime visualization
- `01_predict_regime_change.ipynb`: placeholder notebook for later predictive modeling work
- `requirements.txt`: Python dependencies for notebooks and utilities

## Requirements

Install dependencies with:

```powershell
pip install -r requirements.txt
```

## Utility Modules

### `deribit_utils.py`

This module handles data ingestion from the Deribit public API.

It includes helpers to:

- fetch spot OHLCV candles with `fetch_deribit_ohlcv(..., market_type="spot")`
- fetch perpetual OHLCV candles with `fetch_deribit_ohlcv(..., market_type="perpetual")`
- fetch perpetual funding-rate history with `fetch_deribit_funding_rates(...)`
- merge spot, perpetual, and funding datasets with `merge_deribit_dataframes(...)`
- generate and optionally save a combined dataset with `generate_merged_deribit_dataset(...)`

Notable behavior:

- large date ranges are requested in chunks to stay within Deribit candle limits
- timestamps are normalized to UTC and used as the dataframe index
- duplicate timestamps are removed after chunked downloads
- merged outputs use outer joins so downstream analysis can decide how to handle missing values
- callers can optionally drop rows with missing values in selected columns before saving

Typical example:

```python
from datetime import datetime, timezone
from deribit_utils import generate_merged_deribit_dataset

df = generate_merged_deribit_dataset(
    base_asset="BTC",
    start_dt=datetime(2024, 1, 1, tzinfo=timezone.utc),
    end_dt=datetime(2024, 3, 1, tzinfo=timezone.utc),
    ohlcv_resolution="60",
    funding_resolution="8h",
    csv_path="deribit_data.csv",
)
```

### `hmm_utils.py`

This module is intentionally focused on the HMM side of regime modeling. It does not build supervised targets or train downstream classifiers.

It supports:

- chronological train / validation / test splitting with `make_time_splits(...)`
- candidate feature-subset generation with `generate_feature_subsets(...)`
- feature cleaning and correlation filtering with `clean_feature_frame(...)` and `filter_high_correlation_features(...)`
- Gaussian HMM fitting with `fit_hmm(...)`
- creation of HMM-derived regime features with `add_hmm_features(...)`
- unsupervised subset diagnostics with `evaluate_hmm_feature_subset(...)`
- heuristic ranking of candidate subsets with `automatic_hmm_feature_selection(...)`
- extraction and summarization of the best search results with `extract_best_hmm_feature_subset(...)` and `summarize_hmm_results(...)`

The generated HMM feature frame includes:

- `{prefix}_state`
- `{prefix}_prob_0 ... {prefix}_prob_k`
- `{prefix}_max_prob`
- `{prefix}_entropy`

The HMM search is ranked using unsupervised regime-quality diagnostics rather than predictive accuracy. The model weighting inside `simple_hmm_selection_score(...)` favors:

- persistent regimes via higher self-transition probabilities
- balanced state usage
- shorter, more responsive median regime run lengths
- lower state uncertainty
- higher normalized likelihood only as a weak tie-breaker

More specifically, `automatic_hmm_feature_selection(...)` optimizes over:

- all feature subsets between `subset_min_size` and `subset_max_size`
- each hidden-state count listed in `n_states_list`
- only the chronological training split from `make_time_splits(...)`

Before the search starts, the candidate list can be reduced with `filter_high_correlation_features(...)`, which removes highly correlated columns using train data only. When two features exceed the threshold, the later feature in the provided column order is dropped.

For every surviving subset and state count, `evaluate_hmm_feature_subset(...)` fits a Gaussian HMM on the training split and computes:

- `avg_self_transition`
- `min_state_fraction`
- `median_run_length`
- `avg_entropy`
- `train_loglik`

The optimization criterion used for ranking is the heuristic `selection_score`, defined as:

```python
3.0 * avg_self_transition
+ 1.5 * min_state_fraction
- 0.25 * median_run_length
- 2.5 * avg_entropy
+ 0.05 * loglik_per_obs_per_feature
```

Only converged fits with every state's occupancy above `min_state_fraction_threshold` receive a finite score. In practice, this means the search strongly prefers sticky, reasonably balanced, low-entropy regimes, penalizes overly long median runs, and uses normalized log-likelihood only as a light secondary term rather than the primary ranking objective. `train_loglik` is still reported for inspection, but it is not the score that drives model selection.

Successful candidates are sorted in descending order by:

1. `selection_score`
2. `avg_self_transition`
3. `min_state_fraction`

Then `extract_best_hmm_feature_subset(...)` returns the feature subset and `n_states` from the first successful row in that ranked table.

Typical example:

```python
from hmm_utils import automatic_hmm_feature_selection, extract_best_hmm_feature_subset

candidate_features = [
    "ret_spot",
    "ret_perp",
    "rolling_vol_24h",
    "rolling_vol_72h",
]

results = automatic_hmm_feature_selection(
    df=df,
    candidate_features=candidate_features,
    subset_min_size=1,
    subset_max_size=3,
    n_states_list=[2, 3],
)

best_features, best_n_states = extract_best_hmm_feature_subset(results)
```

## Notebook Guide

### `00_plot_regime_change.ipynb`

This notebook is the exploratory entry point for the project. It focuses on market feature engineering and visual inspection of volatility regimes.

It is used to:

- load existing Deribit CSV data or regenerate it from the API
- compute return and volatility-oriented features
- visualize rolling volatility and rule-based high-volatility periods
- inspect when BTC appears calm versus stressed

The regime labels in this notebook are rule-based volatility labels, not latent HMM states.

### `01_predict_regime_change.ipynb`

This notebook currently exists as a scaffold for a later predictive stage. At the moment, the reusable modeling utilities in the repository are HMM-focused rather than full end-to-end supervised prediction utilities.

## Typical Workflow

1. Use `deribit_utils.py` to fetch and merge BTC spot, perpetual, and funding-rate history.
2. Run `00_plot_regime_change.ipynb` to engineer features and inspect volatility behavior.
3. Use `hmm_utils.py` to search for HMM feature subsets that produce stable, interpretable latent regimes.
4. Add HMM-derived regime features back to the dataset for downstream analysis or future predictive modeling.

## Notes

- The repository assumes time-ordered market data.
- `hmm_utils.py` is unsupervised and focused on regime extraction quality, not classifier performance.
- `xgboost` is present in `requirements.txt`, but no XGBoost utility module is currently included in the repository.

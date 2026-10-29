"""Simple, shared model steps for the pricing analysis."""

# Import the packages used for data preparation and ordinary least squares.
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm


# Define the outcome and the five predictors in the original notebook.
PRICE = "blended_cost_usd_per_1m"
BASE_FEATURES = [
    "aa_intelligence_index",
    "output_tokens_per_second",
    "time_to_first_token_s",
    "time_to_first_answer_s",
    "parameter_count",
]


def clean_paid(raw):
    """Keep positive, finite prices and return a complete row-count ledger."""
    # Keep the original row positions even when the input index is different.
    data = raw.copy()
    data["row_id"] = np.arange(len(data), dtype=int)
    price = pd.to_numeric(data[PRICE], errors="coerce")

    # Give each source row one mutually exclusive cleaning status.
    missing = price.isna()
    nonfinite = price.notna() & ~np.isfinite(price)
    zero = np.isfinite(price) & price.eq(0)
    negative = np.isfinite(price) & price.lt(0)
    paid = np.isfinite(price) & price.gt(0)
    ledger = pd.DataFrame(
        {
            "status": ["missing", "zero", "negative", "nonfinite", "paid"],
            "count": [
                int(missing.sum()),
                int(zero.sum()),
                int(negative.sum()),
                int(nonfinite.sum()),
                int(paid.sum()),
            ],
        }
    )

    # Use natural log prices only after removing nonpositive prices.
    data[PRICE] = price
    paid_data = data.loc[paid].copy()
    paid_data["log_price"] = np.log(paid_data[PRICE])
    return paid_data, ledger


def numeric_values(data, feature):
    """Coerce one predictor and treat infinite entries as missing."""
    # Return missing values when an entire source column is absent.
    if feature not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype=float)

    # Keep only numeric, finite predictor values.
    return pd.to_numeric(data[feature], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def open_source_values(data):
    """Normalize common status values, with missing or invalid values set to zero."""
    # Follow the original notebook's missing-to-zero encoding.
    if "is_open_source" not in data.columns:
        return pd.Series(0.0, index=data.index)
    values = data["is_open_source"].astype("string").str.strip().str.lower()
    mapping = {
        "true": 1.0,
        "false": 0.0,
        "yes": 1.0,
        "no": 0.0,
        "1": 1.0,
        "0": 0.0,
        "1.0": 1.0,
        "0.0": 0.0,
    }
    return values.map(mapping).fillna(0.0).astype(float)


def provider_values(data):
    """Use Unknown for missing provider names."""
    # Preserve provider labels while making the alphabetical ordering explicit.
    if "provider" not in data.columns:
        return pd.Series("Unknown", index=data.index, dtype="string")
    return data["provider"].fillna("Unknown").astype(str)


def design_rank(design):
    """Check rank after scaling, without changing the fitted predictor units."""
    # Center and scale nonconstant columns for the numerical rank check only.
    scaled = design.to_numpy(dtype=float).copy()
    for column_number, column in enumerate(design.columns):
        if column != "const":
            values = scaled[:, column_number]
            scaled[:, column_number] = (values - values.mean()) / values.std()

    # Stop before inference if the predictors contain an exact linear dependency.
    rank = int(np.linalg.matrix_rank(scaled))
    if rank < design.shape[1]:
        raise ValueError(
            f"Training design is rank deficient: rank {rank} for "
            f"{design.shape[1]} columns. Check redundant predictors and provider "
            f"indicators. Columns: {design.columns.tolist()}"
        )
    return rank


def prepare_design(
    train, test=None, include_year=True, exclude_features=None, rare_provider_min=0
):
    """Learn preprocessing on training rows, then apply it to optional test rows."""
    # Start with an intercept and the requested continuous predictors.
    if len(train) == 0:
        raise ValueError("Training data must contain at least one row.")
    if not isinstance(rare_provider_min, (int, np.integer)) or rare_provider_min < 0:
        raise ValueError("rare_provider_min must be a nonnegative integer.")
    excluded = set(exclude_features or [])
    features = BASE_FEATURES + (["release_year"] if include_year else [])
    features = [feature for feature in features if feature not in excluded]
    x_train = pd.DataFrame({"const": 1.0}, index=train.index)
    x_test = None if test is None else pd.DataFrame({"const": 1.0}, index=test.index)
    medians = {}
    dropped = []
    drop_reasons = {}

    # Learn each median from training rows only and omit all-missing columns.
    for feature in features:
        train_values = numeric_values(train, feature)
        median = train_values.median()
        if pd.isna(median):
            dropped.append(feature)
            drop_reasons[feature] = "all_missing_in_training"
            continue
        medians[feature] = float(median)
        x_train[feature] = train_values.fillna(median).astype(float)
        if test is not None:
            x_test[feature] = numeric_values(test, feature).fillna(median).astype(float)

    # Apply the original missing-to-zero open-source rule in both partitions.
    if "is_open_source" not in excluded:
        x_train["is_open_source"] = open_source_values(train)
        if test is not None:
            x_test["is_open_source"] = open_source_values(test)

    # Learn alphabetical provider categories and omit the first category.
    categories = []
    original_categories = []
    grouping = {}
    group_counts = {}
    unseen_providers = []
    unseen_mask = [] if test is None else [False] * len(test)
    pooled_mask = [] if test is None else [False] * len(test)
    if "provider" not in excluded:
        train_providers = provider_values(train)
        counts = train_providers.value_counts()
        original_categories = sorted(counts.index.tolist())
        grouping = {
            provider: "Other creators" if counts[provider] < rare_provider_min else provider
            for provider in original_categories
        }
        train_providers = train_providers.map(grouping)
        categories = sorted(train_providers.unique().tolist())
        group_counts = {
            provider: int(count) for provider, count in train_providers.value_counts().items()
        }
        test_providers = None if test is None else provider_values(test)

        # Pool rare creators using training counts, and keep unseen labels visible.
        if test is not None:
            unseen = ~test_providers.isin(original_categories)
            unseen_providers = sorted(test_providers.loc[unseen].unique().tolist())
            unseen_mask = unseen.tolist()
            mapped_test = test_providers.map(grouping)
            if "Other creators" in categories:
                mapped_test = mapped_test.fillna("Other creators")
                pooled_mask = mapped_test.eq("Other creators").tolist()
            test_providers = mapped_test

        # Omit one alphabetical category and align every test dummy to training.
        for provider in categories[1:]:
            column = "provider_" + provider
            if column not in excluded:
                x_train[column] = train_providers.eq(provider).astype(float)
                if test is not None:
                    x_test[column] = test_providers.eq(provider).astype(float)

    # Drop training constants and keep the same columns for test predictions.
    for column in list(x_train.columns):
        if column != "const" and x_train[column].nunique() <= 1:
            x_train = x_train.drop(columns=column)
            dropped.append(column)
            drop_reasons[column] = "constant_in_training"
    if x_test is not None:
        x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)

    # Record preprocessing choices and verify that inference has a full-rank design.
    rank = design_rank(x_train)
    info = {
        "feature_medians": medians,
        "provider_reference": categories[0] if categories else None,
        "provider_categories": categories,
        "original_provider_categories": original_categories,
        "provider_grouping": grouping,
        "group_counts": group_counts,
        "rare_provider_min": int(rare_provider_min),
        "unseen_test_providers": unseen_providers,
        "unseen_test_mask": unseen_mask,
        "pooled_test_mask": pooled_mask,
        "dropped_columns": dropped,
        "dropped_column_reasons": drop_reasons,
        "rank": rank,
        "columns": x_train.columns.tolist(),
    }
    return x_train, x_test, info


def fit_pair(data, include_year=True, exclude_features=None, rare_provider_min=0):
    """Fit the full OLS model, select once at p < .05, and refit the reduced model."""
    # Prepare the design without altering its original measurement units.
    design, _, info = prepare_design(
        data,
        include_year=include_year,
        exclude_features=exclude_features,
        rare_provider_min=rare_provider_min,
    )
    outcome = pd.to_numeric(data["log_price"], errors="coerce").astype(float)
    if not np.isfinite(outcome).all():
        raise ValueError("log_price must be finite for every model row.")
    if len(data) <= design.shape[1]:
        raise ValueError("The full model needs at least one residual degree of freedom.")

    # Select from the full model's conventional p values exactly once.
    full = sm.OLS(outcome, design).fit()
    p_values = full.pvalues.drop("const")
    selected = p_values[p_values < 0.05].index.tolist()
    reduced = sm.OLS(outcome, design[["const"] + selected]).fit()
    return {
        "full": full,
        "reduced": reduced,
        "selected": selected,
        "X": design,
        "info": info,
    }


def coefficient_table(result):
    """Return conventional and HC3 inference without changing OLS coefficients."""
    # Save the conventional coefficient estimates and confidence intervals.
    terms = result.model.exog_names
    confidence = np.asarray(result.conf_int())
    table = pd.DataFrame(
        {
            "term": terms,
            "coef": np.asarray(result.params),
            "std_error": np.asarray(result.bse),
            "p_value": np.asarray(result.pvalues),
            "ci_low": confidence[:, 0],
            "ci_high": confidence[:, 1],
        }
    )

    # Leave HC3 inference unavailable when a row has leverage equal to one.
    hc3_columns = ["hc3_std_error", "hc3_p_value", "hc3_ci_low", "hc3_ci_high"]
    table[hc3_columns] = np.nan
    table["hc3_valid"] = False
    table["hc3_reason"] = ""
    leverage = np.asarray(result.get_influence().hat_matrix_diag)
    leverage_one = np.flatnonzero(leverage >= 1.0 - 1e-10)
    table.attrs["leverage_one_positions"] = leverage_one.tolist()
    table.attrs["maximum_leverage"] = float(leverage.max())
    table.attrs["hc3_available"] = False
    if len(leverage_one):
        issue = f"HC3 unavailable: leverage equals one at row positions {leverage_one.tolist()}."
        table.attrs["hc3_issue"] = issue
        table["hc3_reason"] = issue
        warnings.warn(issue, RuntimeWarning, stacklevel=2)
        return table

    # Compute HC3 with the same t reference used for conventional OLS inference.
    try:
        robust = result.get_robustcov_results(cov_type="HC3", use_t=True)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError, ZeroDivisionError) as error:
        issue = f"HC3 unavailable: {error}"
        table.attrs["hc3_issue"] = issue
        table["hc3_reason"] = issue
        warnings.warn(issue, RuntimeWarning, stacklevel=2)
        return table

    # Verify that robust inference leaves the fitted OLS estimates unchanged.
    if not np.allclose(result.params, robust.params, rtol=1e-12, atol=1e-12):
        raise RuntimeError("HC3 unexpectedly changed the OLS coefficient estimates.")
    robust_confidence = np.asarray(robust.conf_int())
    robust_values = np.column_stack(
        [robust.bse, robust.pvalues, robust_confidence[:, 0], robust_confidence[:, 1]]
    )

    # Report any numerical HC3 failure instead of exporting invalid inference.
    if not np.isfinite(robust_values).all():
        issue = "HC3 unavailable: its standard errors, p values, or intervals are nonfinite."
        table.attrs["hc3_issue"] = issue
        table["hc3_reason"] = issue
        warnings.warn(issue, RuntimeWarning, stacklevel=2)
        return table
    table[hc3_columns] = robust_values
    table["hc3_valid"] = True
    table.attrs["hc3_available"] = True
    table.attrs["hc3_issue"] = None
    return table

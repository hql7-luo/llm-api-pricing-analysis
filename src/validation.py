"""Compare full and selected models using held-out model families."""

# Import simple splitting, error metrics, and the shared model preparation.
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from models import PRICE, BASE_FEATURES, fit_pair, prepare_design


# Keep recognizable reasoning and effort variants of one base model together.
def model_groups(data):
    groups = []
    for name, creator in zip(data["model_name"], data["provider"]):
        base = re.sub(r"\([^)]*\)", "", str(name).lower())
        base = re.sub(r"claude\s+(\d+(?:\.\d+)?)\s+(opus|sonnet|haiku)", r"claude \2 \1", base)
        base = re.sub(r"\b(non[- ]reasoning|reasoning|thinking|preview|instruct)\b", "", base)
        base = base.replace("+", " plus ")
        base = re.sub(r"[^a-z0-9]+", " ", base).strip()
        groups.append(str(creator) + "::" + base)
    return pd.Series(groups, index=data.index, name="model_group")


# Report errors on both log and original dollar scales.
def error_metrics(actual_log, predicted_log, actual_price, predicted_price):
    return {
        "n": len(actual_log),
        "log_mae": mean_absolute_error(actual_log, predicted_log),
        "log_rmse": np.sqrt(mean_squared_error(actual_log, predicted_log)),
        "price_mae": mean_absolute_error(actual_price, predicted_price),
        "price_rmse": np.sqrt(mean_squared_error(actual_price, predicted_price)),
        "oof_log_r2": r2_score(actual_log, predicted_log) if len(actual_log) > 1 else np.nan,
    }


# Fit all preprocessing and one-pass selection inside each training fold.
def cross_validate(data, n_splits=5, seed=451):
    groups = model_groups(data)
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    predictions, fold_results, fold_audit = [], [], []
    for fold, (train_positions, test_positions) in enumerate(splitter.split(data, groups=groups), 1):
        train = data.iloc[train_positions].copy()
        test = data.iloc[test_positions].copy()
        assert set(groups.iloc[train_positions]).isdisjoint(groups.iloc[test_positions])

        # Keep the full model as the predeclared ranking model, not the CV winner.
        for scenario in ["full", "reduced", "without_year_parameters"]:
            include_year = scenario != "without_year_parameters"
            excluded = ["parameter_count"] if not include_year else None
            pair = fit_pair(train, include_year=include_year, exclude_features=excluded, rare_provider_min=5)
            x_train, x_test, info = prepare_design(
                train, test, include_year=include_year,
                exclude_features=excluded, rare_provider_min=5,
            )
            fitted = pair["reduced"] if scenario == "reduced" else pair["full"]
            x_test = x_test[fitted.params.index]
            predicted_log = np.asarray(fitted.predict(x_test))

            # Estimate the log-to-price correction from training residuals only.
            smearing = float(np.exp(fitted.resid).mean())
            predicted_price = np.exp(predicted_log) * smearing
            if not np.isfinite(predicted_price).all():
                raise ValueError("Nonfinite held-out price predictions; inspect the model.")
            fold_results.append({"fold": fold, "model": scenario, **error_metrics(
                test["log_price"], predicted_log, test[PRICE], predicted_price,
            )})

            # Preserve row-level provenance for the held-out predictions.
            record = test[["row_id", "model_name", "provider", PRICE, "log_price"]].copy()
            record["model_group"] = groups.iloc[test_positions].to_numpy()
            record["fold"] = fold
            record["model"] = scenario
            record["predicted_log_price"] = predicted_log
            record["predicted_price"] = predicted_price
            record["smearing_factor"] = smearing
            record["unseen_creator"] = info["unseen_test_mask"]
            record["pooled_creator"] = info["pooled_test_mask"]
            record["log_residual"] = record["log_price"] - predicted_log
            record["price_ratio"] = record[PRICE] / predicted_price
            predictions.append(record)
            fold_audit.append({
                "fold": fold, "model": scenario,
                "train_rows": train["row_id"].tolist(), "test_rows": test["row_id"].tolist(),
                "train_groups": sorted(set(groups.iloc[train_positions])),
                "test_groups": sorted(set(groups.iloc[test_positions])),
                "feature_medians": info["feature_medians"],
                "provider_reference": info["provider_reference"],
                "provider_grouping": info["provider_grouping"],
                "creator_group_counts": info["group_counts"],
                "dropped_columns": info["dropped_columns"],
                "dropped_column_reasons": info["dropped_column_reasons"],
                "selected_terms": fitted.params.index.tolist(),
                "unseen_creators": info["unseen_test_providers"],
                "smearing_factor": smearing,
            })

        # Use simple training means as a common baseline in every fold.
        baseline_log = np.repeat(train["log_price"].mean(), len(test))
        baseline_price = np.repeat(train[PRICE].mean(), len(test))
        fold_results.append({"fold": fold, "model": "training_mean", **error_metrics(
            test["log_price"], baseline_log, test[PRICE], baseline_price,
        )})
        record = test[["row_id", "model_name", "provider", PRICE, "log_price"]].copy()
        record["model_group"] = groups.iloc[test_positions].to_numpy()
        record["fold"] = fold
        record["model"] = "training_mean"
        record["predicted_log_price"] = baseline_log
        record["predicted_price"] = baseline_price
        record["unseen_creator"] = ~test["provider"].isin(train["provider"])
        record["pooled_creator"] = False
        predictions.append(record)

    # Pool held-out rows once, rather than averaging unequal fold metrics.
    predictions = pd.concat(predictions, ignore_index=True).sort_values(["model", "row_id"])
    scores = []
    for model, rows in predictions.groupby("model"):
        assert len(rows) == len(data) and rows["row_id"].is_unique
        for scope, subset in [("all", rows), ("seen_creators", rows.loc[~rows.unseen_creator]), ("unseen_creators", rows.loc[rows.unseen_creator])]:
            if len(subset):
                scores.append({"model": model, "scope": scope, **error_metrics(
                    subset["log_price"], subset["predicted_log_price"],
                    subset[PRICE], subset["predicted_price"],
                )})
    return predictions, pd.DataFrame(fold_results), pd.DataFrame(scores), fold_audit


# Identify candidates using price deviation, with missing-data flags visible.
def price_deviation_table(data, predictions):
    full = predictions.loc[predictions["model"].eq("full")].copy()
    features = BASE_FEATURES + ["release_year"]
    details = data[["row_id"] + features].copy()
    details["missing_predictor_count"] = details[features].isna().sum(axis=1)
    ranked = full.merge(details, on="row_id", validate="one_to_one")
    ranked["deviation_percent"] = 100 * (ranked["price_ratio"] - 1)
    ranked["known_release_year_issue"] = ranked["model_name"].str.contains(
        r"Claude (?:Opus|Sonnet) 4\.6", case=False, regex=True,
    ) & ranked["release_year"].ne(2026)
    ranked["eligible_for_main_ranking"] = (
        ~ranked["unseen_creator"] & ranked["missing_predictor_count"].eq(0)
        & ~ranked["known_release_year_issue"]
    )

    # Make sensitivity to model specification visible beside every candidate.
    for alternative in ["reduced", "without_year_parameters"]:
        comparison = predictions.loc[predictions["model"].eq(alternative), ["row_id", "price_ratio"]]
        comparison = comparison.rename(columns={"price_ratio": alternative + "_price_ratio"})
        ranked = ranked.merge(comparison, on="row_id", validate="one_to_one")
    ranked["same_direction_all_specs"] = (
        ranked["price_ratio"].lt(1).eq(ranked["reduced_price_ratio"].lt(1))
        & ranked["price_ratio"].lt(1).eq(ranked["without_year_parameters_price_ratio"].lt(1))
    )
    ranked["snapshot_date"] = "2026-03-31"
    ranked["price_deviation_rank"] = np.nan
    eligible = ranked["eligible_for_main_ranking"]
    ranked.loc[eligible, "price_deviation_rank"] = ranked.loc[eligible, "price_ratio"].rank(method="min")
    return ranked.sort_values("price_ratio").reset_index(drop=True)

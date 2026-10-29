"""Independently check the saved analysis artifacts against their source rows."""

# Import only file readers and arithmetic, without using the analysis functions.
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Pin the downloaded snapshot and spell out the predictors independently.
EXPECTED_SHA256 = "1c83338dc40f826ce19f3c6d17a3e03970979587fb860dccc79846f691e07177"
PRICE = "blended_cost_usd_per_1m"
FEATURES = [
    "aa_intelligence_index", "output_tokens_per_second", "time_to_first_token_s",
    "time_to_first_answer_s", "parameter_count", "release_year",
]
CHECKS = []


# Fail with a readable message and count each successful artifact check.
def require(condition, message):
    if not condition:
        raise AssertionError(message)
    CHECKS.append(message)


# Compare saved numerical values with independently calculated values.
def close(actual, expected, message):
    require(np.allclose(actual, expected, rtol=1e-9, atol=1e-11), message)


# Verify the source fingerprint and independently rebuild the price ledger.
def check_source(root):
    manifest = json.loads((root / "data/manifest.json").read_text())
    source = root / "data/raw" / manifest["filename"]
    source_bytes = source.read_bytes()
    fingerprint = hashlib.sha256(source_bytes).hexdigest()
    require(fingerprint == manifest["csv_sha256"], "Manifest CSV hash matches source")
    require(fingerprint == EXPECTED_SHA256, "Source matches the pinned regression snapshot")
    require(len(source_bytes) == manifest["csv_bytes"], "Manifest CSV size matches source")
    raw = pd.read_csv(source)
    require(raw.shape == (453, 34), "Raw source has 453 rows and 34 columns")

    # Classify every raw price before checking the exported cleaning ledger.
    prices = pd.to_numeric(raw[PRICE], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(prices)
    counts = {
        "missing": int(np.isnan(prices).sum()),
        "zero": int((finite & (prices == 0)).sum()),
        "negative": int((finite & (prices < 0)).sum()),
        "nonfinite": int((~finite & ~np.isnan(prices)).sum()),
        "paid": int((finite & (prices > 0)).sum()),
    }
    ledger = pd.read_csv(root / "outputs/tables/cleaning_ledger.csv")
    require(ledger["status"].is_unique, "Cleaning ledger statuses are unique")
    require(dict(zip(ledger["status"], ledger["count"])) == counts, "Ledger matches source prices")
    require(counts == {"missing": 138, "zero": 16, "negative": 0, "nonfinite": 0, "paid": 299},
            "Pinned cleaning counts are unchanged")
    require(ledger["count"].sum() == 453, "Cleaning ledger covers every source row")
    return raw, set(np.flatnonzero(finite & (prices > 0)).tolist())


# Guard the observed report and original-notebook regression results.
def check_reproduction(root):
    comparison = pd.read_csv(root / "outputs/tables/model_comparison.csv")
    require(not comparison.duplicated(["specification", "model"]).any(),
            "Model-comparison rows are unique")
    values = comparison.set_index(["specification", "model"])
    expected = [
        ("report_with_year", "full", 0.6318923011303204, 0.5797084510989865, 37),
        ("report_with_year", "reduced", 0.5745438108695393, 0.5551370373302551, 13),
        ("notebook_no_year", "full", 0.6232296120158871, 0.5714596350409709, 36),
        ("notebook_no_year", "reduced", 0.5665712417357569, 0.5403495730863188, 17),
    ]

    # Use numeric tolerances only for floating-point output rounding.
    for specification, model, r_squared, adjusted, predictors in expected:
        row = values.loc[(specification, model)]
        close(row["r_squared"], r_squared, specification + " " + model + " R squared")
        close(row["adjusted_r_squared"], adjusted, specification + " " + model + " adjusted R squared")
        require(row["predictors"] == predictors, specification + " " + model + " predictor count")
        require(row["n"] == 299, specification + " " + model + " sample size")


# Check saved folds, disjoint groups, training medians, and recognizable families.
def check_folds(root, raw, paid_ids):
    predictions = pd.read_csv(root / "outputs/private/oof_predictions.csv")
    groups = pd.read_csv(root / "outputs/tables/validation_groups.csv")
    audits = json.loads((root / "outputs/tables/cv_audit.json").read_text())
    scenarios = {"full", "reduced", "without_year_parameters", "training_mean"}
    require(set(predictions["model"]) == scenarios, "All four prediction scenarios are present")
    require(not predictions.duplicated(["model", "row_id"]).any(), "Each scenario predicts each row once")
    require(groups["row_id"].is_unique and set(groups["row_id"]) == paid_ids,
            "Validation groups cover the paid rows once")
    group_map = groups.set_index("row_id")["model_group"]

    # Confirm complete held-out coverage and identical fold membership across scenarios.
    for scenario, rows in predictions.groupby("model"):
        require(set(rows["row_id"]) == paid_ids, scenario + " covers every paid row")
        require(set(rows["fold"]) == set(range(1, 6)), scenario + " uses five folds")
        close(rows[PRICE], raw.loc[rows["row_id"], PRICE], scenario + " observed prices match source")
        close(rows["log_price"], np.log(rows[PRICE]), scenario + " observed log prices match source")
        require(rows["model_group"].tolist() == group_map.loc[rows["row_id"]].tolist(),
                scenario + " group labels match saved groups")
    fold_matrix = predictions.pivot(index="row_id", columns="model", values="fold")
    require(fold_matrix.nunique(axis=1).eq(1).all(), "Every scenario uses the same folds")
    audit_keys = {(audit["model"], audit["fold"]) for audit in audits}
    expected_keys = {(model, fold) for model in scenarios - {"training_mean"} for fold in range(1, 6)}
    require(len(audits) == 15 and audit_keys == expected_keys, "All 15 model-fold audits are present")

    # Rebuild every training median directly from the raw training-row positions.
    for audit in audits:
        label = audit["model"] + " fold " + str(audit["fold"])
        train_ids, test_ids = audit["train_rows"], audit["test_rows"]
        require(len(set(train_ids)) == len(train_ids) and len(set(test_ids)) == len(test_ids),
                label + " has no duplicated split rows")
        require(set(train_ids).isdisjoint(test_ids), label + " train and test rows are disjoint")
        require(set(train_ids) | set(test_ids) == paid_ids, label + " partitions the paid sample")
        train_groups = set(group_map.loc[train_ids])
        test_groups = set(group_map.loc[test_ids])
        require(train_groups.isdisjoint(test_groups), label + " model families are disjoint")
        require(set(audit["train_groups"]) == train_groups and set(audit["test_groups"]) == test_groups,
                label + " group audit matches row membership")
        matching = predictions.loc[predictions["model"].eq(audit["model"])
                                   & predictions["fold"].eq(audit["fold"])]
        require(set(matching["row_id"]) == set(test_ids), label + " predictions match test rows")
        features = FEATURES if audit["model"] != "without_year_parameters" else FEATURES[:-2]
        medians = {}
        for feature in features:
            values = pd.to_numeric(raw.loc[train_ids, feature], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values):
                medians[feature] = float(np.median(values))
        require(set(medians) == set(audit["feature_medians"]), label + " median feature set matches training")
        for feature, median in medians.items():
            close(audit["feature_medians"][feature], median, label + " training median " + feature)

        # Verify that creator pooling uses training creator counts only.
        train_creators = raw.loc[train_ids, "provider"].fillna("Unknown")
        counts = train_creators.value_counts()
        mapping = {creator: "Other creators" if count < 5 else creator for creator, count in counts.items()}
        require(audit["provider_grouping"] == mapping, label + " creator pooling uses training counts")
        pooled_counts = train_creators.map(mapping).value_counts().to_dict()
        require(audit["creator_group_counts"] == pooled_counts, label + " pooled creator counts match")

    # Use actual source names to guard known reasoning and instruction variants.
    name_groups = groups.set_index("model_name")["model_group"]
    pairs = [
        ("Qwen3 235B A22B (Reasoning)", "Qwen3 235B A22B (Non-reasoning)"),
        ("Qwen3 235B A22B 2507 (Reasoning)", "Qwen3 235B A22B 2507 Instruct"),
        ("Qwen3 Next 80B A3B (Reasoning)", "Qwen3 Next 80B A3B Instruct"),
        ("Qwen3 VL 32B (Reasoning)", "Qwen3 VL 32B Instruct"),
        ("Qwen3 Omni 30B A3B (Reasoning)", "Qwen3 Omni 30B A3B Instruct"),
    ]
    for first, second in pairs:
        require(name_groups.loc[first] == name_groups.loc[second], first + " shares its base-model group")
    require(name_groups.loc["Command-R (Mar '24)"] != name_groups.loc["Command-R+ (Apr '24)"],
            "Command-R and Command-R+ remain separate model families")
    return predictions


# Recalculate every published error metric using only NumPy and saved predictions.
def check_metrics(root, predictions):
    metrics = pd.read_csv(root / "outputs/tables/cv_metrics.csv")
    require(not metrics.duplicated(["model", "scope"]).any(), "CV metric rows are unique")
    require(len(metrics) == 12, "All twelve model and scope metric rows are present")
    for _, saved in metrics.iterrows():
        rows = predictions.loc[predictions["model"].eq(saved["model"])]
        if saved["scope"] == "seen_creators":
            rows = rows.loc[~rows["unseen_creator"]]
        elif saved["scope"] == "unseen_creators":
            rows = rows.loc[rows["unseen_creator"]]
        else:
            require(saved["scope"] == "all", "CV scope is recognized")

        # Calculate absolute errors, squared errors, and R squared independently.
        log_error = rows["log_price"].to_numpy() - rows["predicted_log_price"].to_numpy()
        price_error = rows[PRICE].to_numpy() - rows["predicted_price"].to_numpy()
        log_values = rows["log_price"].to_numpy()
        calculated = {
            "log_mae": np.mean(np.abs(log_error)),
            "log_rmse": np.sqrt(np.mean(log_error ** 2)),
            "price_mae": np.mean(np.abs(price_error)),
            "price_rmse": np.sqrt(np.mean(price_error ** 2)),
            "oof_log_r2": 1 - np.sum(log_error ** 2) / np.sum((log_values - log_values.mean()) ** 2),
        }
        label = saved["model"] + " " + saved["scope"]
        require(saved["n"] == len(rows), label + " metric sample size matches")
        for metric, value in calculated.items():
            close(saved[metric], value, label + " " + metric)


# Compare the public ranking with the full model's held-out private predictions.
def check_ranking(root, raw, paid_ids, predictions):
    ranking = pd.read_csv(root / "outputs/tables/price_deviation_ranking.csv")
    require(ranking["row_id"].is_unique and set(ranking["row_id"]) == paid_ids,
            "Public ranking covers paid rows once")
    full = predictions.loc[predictions["model"].eq("full")].set_index("row_id")
    ordered = ranking.set_index("row_id").loc[sorted(paid_ids)]
    full = full.loc[ordered.index]
    ratios = raw.loc[ordered.index, PRICE].to_numpy() / full["predicted_price"].to_numpy()
    close(ordered["price_ratio"], ratios, "Public full-model ratios match private predictions")
    close(ordered["deviation_percent"], 100 * (ratios - 1), "Deviation percentages match price ratios")

    # Rebuild known year issues and the predeclared eligibility conditions.
    source = raw.loc[ordered.index]
    known_issues = source["model_name"].str.contains(r"Claude (?:Opus|Sonnet) 4\.6", regex=True, case=False)
    known_issues = known_issues & source["release_year"].ne(2026)
    require(known_issues.sum() == 5, "Five known release-year errors are identified")
    require(np.array_equal(ordered["known_release_year_issue"], known_issues),
            "Public known-year flags match source rows")
    missing_counts = source[FEATURES].isna().sum(axis=1)
    expected_eligible = ~full["unseen_creator"] & missing_counts.eq(0) & ~known_issues
    require(np.array_equal(ordered["eligible_for_main_ranking"], expected_eligible),
            "Ranking eligibility follows the declared source-data rules")
    eligible = ordered.loc[ordered["eligible_for_main_ranking"]]
    excluded = ordered.loc[~ordered["eligible_for_main_ranking"]]
    close(np.sort(eligible["price_deviation_rank"]), np.arange(1, len(eligible) + 1),
          "Eligible ranking positions are consecutive")
    require(excluded["price_deviation_rank"].isna().all(), "Excluded rows have no rank")
    require(not ordered.loc[known_issues, "eligible_for_main_ranking"].any(),
            "Known release-year errors never enter the eligible ranking")


# Validate robust inference using the saved coefficient and influence artifacts.
def check_hc3(root):
    table = pd.read_csv(root / "outputs/tables/coefficients_extension_pooled_full.csv")
    require(table["term"].is_unique and len(table) == 23, "Pooled full coefficients have 23 unique terms")
    require(table["hc3_valid"].all(), "Pooled full HC3 inference is valid for every term")
    require(table["hc3_reason"].astype("string").fillna("").eq("").all(), "Valid HC3 rows have no failure reason")
    numeric = ["coef", "std_error", "p_value", "ci_low", "ci_high",
               "hc3_std_error", "hc3_p_value", "hc3_ci_low", "hc3_ci_high"]
    require(np.isfinite(table[numeric].to_numpy()).all(), "Pooled coefficient inference is finite")
    close((table["ci_low"] + table["ci_high"]) / 2, table["coef"], "Conventional intervals center on OLS coefficients")
    close((table["hc3_ci_low"] + table["hc3_ci_high"]) / 2, table["coef"], "HC3 intervals preserve OLS coefficients")
    influence = pd.read_csv(root / "outputs/tables/influence_extension_pooled_full.csv")
    require(influence["leverage"].max() < 1 - 1e-10, "Pooled full model has no leverage-one rows")


# Scan public prose and source files without examining private output contents.
def check_public_text(root):
    paths = [path for path in root.rglob("*") if path.suffix.lower() in {".txt", ".md", ".py"}
             and not {".git", ".venv", "node_modules", "private"}.intersection(path.relative_to(root).parts)]
    require(len(paths) > 0, "Public text files are available for style checks")
    for path in paths:
        require("\u2014" not in path.read_text(encoding="utf-8"), str(path.relative_to(root)) + " contains no em dash")


# Run the checks on the existing artifacts and print compact section totals.
def main():
    root = Path(__file__).resolve().parents[1]
    before = len(CHECKS)
    raw, paid_ids = check_source(root)
    print(f"PASS source and ledger: {len(CHECKS) - before} checks")
    for label, action in [
        ("reproduction", lambda: check_reproduction(root)),
        ("folds and preprocessing", lambda: check_folds(root, raw, paid_ids)),
    ]:
        before = len(CHECKS)
        result = action()
        print(f"PASS {label}: {len(CHECKS) - before} checks")
        if label == "folds and preprocessing":
            predictions = result

    # Reuse the saved predictions only after their provenance checks have passed.
    for label, action in [
        ("CV metrics", lambda: check_metrics(root, predictions)),
        ("ranking", lambda: check_ranking(root, raw, paid_ids, predictions)),
        ("HC3", lambda: check_hc3(root)),
        ("public text", lambda: check_public_text(root)),
    ]:
        before = len(CHECKS)
        action()
        print(f"PASS {label}: {len(CHECKS) - before} checks")
    print(f"PASS total: {len(CHECKS)} checks; {len(raw)} source rows; {len(paid_ids)} paid rows")


# Keep importing this file separate from running its artifact checks.
if __name__ == "__main__":
    main()

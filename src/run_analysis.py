"""Reproduce the course results and run the independent extension."""

# Import the small analysis modules and statistical diagnostics.
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from download_data import download
from models import PRICE, BASE_FEATURES, clean_paid, fit_pair, coefficient_table
from validation import cross_validate, model_groups, price_deviation_table
from plots import reproduction_plot, diagnostic_plot, coefficient_plot, validation_plot, ranking_plot, selection_plot

# Resolve paths from this file, so running from the repository root is reliable.
ROOT = Path(__file__).resolve().parents[1]


# Keep JSON portable by replacing unavailable numeric values with null.
def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


# Calculate diagnostics without removing observations to improve fit.
def diagnose(result):
    design = pd.DataFrame(result.model.exog, columns=result.model.exog_names)
    vif = pd.DataFrame([
        {"term": name, "vif": variance_inflation_factor(design.to_numpy(), i)}
        for i, name in enumerate(design.columns) if name != "const"
    ])
    lm, lm_p, f_value, f_p = het_breuschpagan(result.resid, result.model.exog)
    influence = result.get_influence()
    leverage = influence.hat_matrix_diag
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cooks = influence.cooks_distance[0]
    diagnostics = {
        "breusch_pagan_lm": lm, "breusch_pagan_p": lm_p,
        "breusch_pagan_f": f_value, "breusch_pagan_f_p": f_p,
        "jarque_bera_p": stats.jarque_bera(result.resid).pvalue,
        "maximum_leverage": np.max(leverage),
        "leverage_one_rows": int(np.sum(leverage >= 1 - 1e-10)),
        "cook_screen_count": int(np.sum(cooks > 4 / result.nobs)),
        "cook_nonfinite_count": int(np.sum(~np.isfinite(cooks))),
        "largest_vif": vif["vif"].max(),
    }
    influence_table = pd.DataFrame({"sample_position": np.arange(int(result.nobs)), "leverage": leverage, "cooks_distance": cooks})
    return vif, diagnostics, influence_table


# Run each analysis once and save numbers before writing any conclusions.
def run():
    tables = ROOT / "outputs" / "tables"
    figures = ROOT / "outputs" / "figures"
    private = ROOT / "outputs" / "private"
    for folder in [tables, figures, private]:
        folder.mkdir(parents=True, exist_ok=True)
    # Remove a superseded local artifact so source prices cannot enter public tables.
    stale_predictions = tables / "oof_predictions.csv"
    if stale_predictions.exists():
        stale_predictions.unlink()
    raw = pd.read_csv(download())
    data, ledger = clean_paid(raw)
    ledger.to_csv(tables / "cleaning_ledger.csv", index=False)
    missing = data[BASE_FEATURES + ["release_year", "is_open_source"]].isna().sum()
    pd.DataFrame({"feature": missing.index, "missing_rows": missing.values, "paid_rows": len(data)}).to_csv(tables / "missing_predictors.csv", index=False)

    # Reproduce both historical versions using their original creator indicators.
    pairs = {
        "notebook_no_year": fit_pair(data, include_year=False),
        "report_with_year": fit_pair(data, include_year=True),
        "extension_pooled": fit_pair(data, include_year=True, rare_provider_min=5),
        "sensitivity_no_year_parameters": fit_pair(data, include_year=False, exclude_features=["parameter_count"], rare_provider_min=5),
    }
    summaries, diagnostics = [], {}
    for specification, pair in pairs.items():
        for model in ["full", "reduced"]:
            result = pair[model]
            coefficients = coefficient_table(result)
            coefficients.to_csv(tables / f"coefficients_{specification}_{model}.csv", index=False)
            summaries.append({
                "specification": specification, "model": model,
                "n": int(result.nobs), "predictors": int(result.df_model),
                "r_squared": result.rsquared, "adjusted_r_squared": result.rsquared_adj,
                "f_p_value": result.f_pvalue, "hc3_available": bool(coefficients.hc3_valid.all()),
            })
            if specification in ["report_with_year", "extension_pooled"]:
                vif, stats_result, influence = diagnose(result)
                vif.to_csv(tables / f"vif_{specification}_{model}.csv", index=False)
                influence.to_csv(tables / f"influence_{specification}_{model}.csv", index=False)
                diagnostics[f"{specification}_{model}"] = stats_result
    model_summary = pd.DataFrame(summaries)
    model_summary.to_csv(tables / "model_comparison.csv", index=False)

    # Check the report's mixed scale: continuous predictors per SD, binary 0 to 1.
    report_pair = pairs["report_with_year"]
    reported = {"aa_intelligence_index": .28, "parameter_count": .26, "release_year": -.33, "output_tokens_per_second": -.74, "is_open_source": -1.36}
    comparison = []
    for term, value in reported.items():
        scale = 1 if term == "is_open_source" else report_pair["X"][term].std(ddof=0)
        comparison.append({"term": term, "report_effect": value, "reproduced_effect": report_pair["reduced"].params[term] * scale, "scale": "0 to 1" if term == "is_open_source" else "1 SD of predictor, ddof=0"})
    pd.DataFrame(comparison).to_csv(tables / "report_coefficient_check.csv", index=False)

    # Evaluate fixed full and training-selected models on held-out base-model groups.
    predictions, fold_scores, cv_scores, audit = cross_validate(data)
    predictions.to_csv(private / "oof_predictions.csv", index=False)
    fold_scores.to_csv(tables / "cv_by_fold.csv", index=False)
    cv_scores.to_csv(tables / "cv_metrics.csv", index=False)
    (tables / "cv_audit.json").write_text(json.dumps(json_ready(audit), indent=2) + "\n")
    groups = data[["row_id", "model_name", "model_slug", "provider"]].copy()
    groups["model_group"] = model_groups(data)
    groups.to_csv(tables / "validation_groups.csv", index=False)

    # Keep complete source-level comparisons local, and publish derived deviations.
    ranking = price_deviation_table(data, predictions)
    ranking.to_csv(private / "price_deviation_details.csv", index=False)
    public_columns = ["row_id", "model_name", "provider", "model_group", "fold", "snapshot_date", "price_ratio", "deviation_percent", "price_deviation_rank", "missing_predictor_count", "unseen_creator", "pooled_creator", "known_release_year_issue", "eligible_for_main_ranking", "same_direction_all_specs", "reduced_price_ratio", "without_year_parameters_price_ratio"]
    ranking[public_columns].to_csv(tables / "price_deviation_ranking.csv", index=False)

    # Put the original dollar-scale OLS comparison in a small appendix only.
    report = pairs["report_with_year"]
    terms = report["reduced"].params.index
    level = sm.OLS(data[PRICE], report["X"][terms]).fit()
    appendix = pd.DataFrame([
        {"outcome": "log_price", "r_squared": report["reduced"].rsquared, "adjusted_r_squared": report["reduced"].rsquared_adj},
        {"outcome": "price", "r_squared": level.rsquared, "adjusted_r_squared": level.rsquared_adj},
    ])
    appendix.to_csv(tables / "appendix_price_scale.csv", index=False)
    coefficient_table(level).to_csv(tables / "appendix_price_coefficients.csv", index=False)

    # Export figures from the computed models and held-out predictions.
    reproduction_plot(model_summary.loc[model_summary.specification.isin(["notebook_no_year", "report_with_year"])], figures)
    diagnostic_plot(pairs["extension_pooled"]["full"], figures)
    coefficient_plot(coefficient_table(pairs["extension_pooled"]["full"]), pairs["extension_pooled"]["X"], figures)
    validation_plot(cv_scores, figures)
    ranking_plot(ranking, figures)
    selection_plot(ranking, figures)

    # Save concise machine-readable evidence for the README and one-page summaries.
    evidence = {
        "snapshot_date": "2026-03-31", "raw_rows": len(raw), "paid_rows": len(data),
        "normalized_model_groups": int(model_groups(data).nunique()),
        "cleaning": ledger.to_dict("records"), "models": summaries,
        "diagnostics": diagnostics, "cv_metrics": cv_scores.to_dict("records"),
        "ranking_eligible_rows": int(ranking.eligible_for_main_ranking.sum()),
        "known_release_year_error_rows": int(ranking.known_release_year_issue.sum()),
        "intelligence_log_price_correlation_after_imputation": data["aa_intelligence_index"].fillna(data["aa_intelligence_index"].median()).corr(data["log_price"]),
        "newer_kaggle_snapshot_found": False,
    }
    (tables / "results.json").write_text(json.dumps(json_ready(evidence), indent=2, allow_nan=False) + "\n")
    print(model_summary.to_string(index=False))
    print("\n5-fold held-out errors:\n", cv_scores.loc[cv_scores.scope.eq("all")].to_string(index=False))
    print(f"\nMain ranking: {evidence['ranking_eligible_rows']} eligible rows; {evidence['normalized_model_groups']} validation groups.")
    return evidence


# Run the full pipeline with one reproducible command.
if __name__ == "__main__":
    run()

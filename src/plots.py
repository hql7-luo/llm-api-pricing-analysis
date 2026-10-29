"""Create compact, labeled figures from the actual saved analysis."""

# Use a file-based plotting backend so the same commands work in CI.
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# Keep a restrained style and readable exported labels.
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})
BLUE, TEAL, ORANGE = "#315c88", "#188477", "#c06a3a"


# Compare historical versions on their common log-price outcome.
def reproduction_plot(table, folder):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = table["specification"] + " / " + table["model"]
    bars = ax.barh(labels, table["r_squared"], color=[BLUE, TEAL, BLUE, TEAL])
    ax.invert_yaxis()
    ax.bar_label(bars, fmt="%.4f", padding=4)
    ax.set(xlim=(0, 0.75), xlabel="In-sample R-squared on log price", title="Adding release year reproduces the report specification")
    fig.tight_layout()
    fig.savefig(folder / "reproduction_comparison.png", bbox_inches="tight")
    plt.close(fig)


# Show residual shape and influence for the pooled full extension model.
def diagnostic_plot(result, folder):
    influence = result.get_influence()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].scatter(result.fittedvalues, result.resid, s=15, color=BLUE, alpha=.65)
    axes[0, 0].axhline(0, color="black", lw=1)
    axes[0, 0].set(xlabel="Fitted log price", ylabel="Residual", title="Residuals versus fitted")
    sm.qqplot(result.resid, line="s", ax=axes[0, 1], markerfacecolor=BLUE, markeredgecolor=BLUE, markersize=3)
    axes[0, 1].set_title("Q-Q plot")
    axes[1, 0].scatter(result.fittedvalues, np.sqrt(np.abs(influence.resid_studentized_internal)), s=15, color=TEAL, alpha=.65)
    axes[1, 0].set(xlabel="Fitted log price", ylabel="sqrt(abs(studentized residual))", title="Scale-location")
    cooks = influence.cooks_distance[0]
    axes[1, 1].stem(np.arange(len(cooks)), cooks, markerfmt=" ", basefmt=" ", linefmt=ORANGE)
    axes[1, 1].axhline(4 / len(cooks), color="black", lw=1, linestyle=":", label="4/n screening line")
    axes[1, 1].set(xlabel="Paid-sample row position", ylabel="Cook's distance", title="Influential observations")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Extension diagnostics: creators with fewer than 5 rows pooled", fontsize=12)
    fig.tight_layout()
    fig.savefig(folder / "residual_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


# Compare conventional and HC3 intervals on a common effect scale.
def coefficient_plot(table, design, folder):
    labels = {"aa_intelligence_index": "Intelligence (+1 SD)", "output_tokens_per_second": "Output speed (+1 SD)", "time_to_first_token_s": "First-token latency (+1 SD)", "time_to_first_answer_s": "First-answer latency (+1 SD)", "parameter_count": "Parameters (+1 SD)", "release_year": "Release year (+1 SD)", "is_open_source": "Open-source indicator (0 to 1)"}
    rows = table.loc[table.term.isin(labels)].copy().set_index("term")
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (term, row) in enumerate(rows.iterrows()):
        scale = 1 if term == "is_open_source" else design[term].std(ddof=0)
        for offset, prefix, color in [(-.1, "", BLUE), (.1, "hc3_", ORANGE)]:
            low, high = row[prefix + "ci_low"] * scale, row[prefix + "ci_high"] * scale
            point = row["coef"] * scale
            ax.errorbar(point, i + offset, xerr=[[point - low], [high - point]], fmt="o", ms=4, color=color, capsize=2)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(len(rows)), [labels[t] for t in rows.index])
    ax.invert_yaxis()
    ax.plot([], [], "o", color=BLUE, label="Conventional 95% CI")
    ax.plot([], [], "o", color=ORANGE, label="HC3 95% CI")
    ax.legend(loc="upper left", fontsize=8)
    ax.set(xlabel="Association with log price, other included variables held fixed", title="Robust intervals can change the strength of the evidence")
    fig.tight_layout()
    fig.savefig(folder / "robust_coefficients.png", bbox_inches="tight")
    plt.close(fig)


# Keep all error metrics on the scale named in each panel.
def validation_plot(metrics, folder):
    rows = metrics.loc[metrics.scope.eq("all")].set_index("model").loc[["training_mean", "full", "reduced", "without_year_parameters"]]
    names = ["Training mean", "Full", "Selected", "No year / parameters"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric, label in zip(axes.flat, ["log_mae", "log_rmse", "price_mae", "price_rmse"], ["MAE in log price", "RMSE in log price", "MAE, USD per 1M tokens", "RMSE, USD per 1M tokens"]):
        bars = ax.barh(names, rows[metric], color=["#a6adb7", BLUE, TEAL, ORANGE])
        ax.invert_yaxis()
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set(xlabel=label, xlim=(0, rows[metric].max() * 1.25))
    fig.suptitle("5-fold grouped validation: each row predicted while held out", fontsize=12)
    fig.tight_layout()
    fig.savefig(folder / "cross_validation_errors.png", bbox_inches="tight")
    plt.close(fig)


# Show model-based price deviation without claiming a quality ranking.
def ranking_plot(ranking, folder):
    candidates = ranking.loc[ranking.eligible_for_main_ranking & ranking.same_direction_all_specs].copy()
    cheaper = candidates.loc[candidates.price_ratio.lt(1)].drop_duplicates("model_group").head(8)
    expensive = candidates.loc[candidates.price_ratio.gt(1)].sort_values("price_ratio", ascending=False).drop_duplicates("model_group").head(8)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, rows, title, color in [(axes[0], cheaper, "Lower than predicted", TEAL), (axes[1], expensive, "Higher than predicted", ORANGE)]:
        positions = np.arange(len(rows))
        ax.hlines(positions, 1, rows.price_ratio, color=color, lw=2)
        ax.scatter(rows.price_ratio, positions, color=color, s=35)
        ax.set_yticks(positions, rows.model_name, fontsize=9)
        ax.invert_yaxis()
        ax.set_xscale("log")
        if title == "Lower than predicted":
            ax.set_xticks([.03, .1, .3, 1], ["0.03x", "0.1x", "0.3x", "1x"])
        else:
            ax.set_xticks([1, 2, 4, 6], ["1x", "2x", "4x", "6x"])
        ax.minorticks_off()
        ax.axvline(1, color="black", lw=1)
        ax.set(xlabel="Actual / held-out predicted price (log axis)", title=title)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Price-deviation candidates, March 31, 2026", fontsize=14)
    fig.text(.5, .01, "Full model with training-only smearing correction. Complete inputs, seen creators, no known year error; direction agrees across specifications.\nOne example per normalized base model. These are exploratory price signals, not quality or procurement recommendations.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .95))
    fig.savefig(folder / "price_deviation_candidates.png", bbox_inches="tight")
    plt.close(fig)


# Put the price deviation signal next to capability and output speed.
def selection_plot(ranking, folder):
    rows = ranking.loc[ranking.eligible_for_main_ranking].copy()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    points = ax.scatter(rows.aa_intelligence_index, rows.blended_cost_usd_per_1m,
                        c=np.log2(rows.price_ratio), cmap="BrBG_r", vmin=-3, vmax=3,
                        s=20 + np.minimum(rows.output_tokens_per_second, 200) / 2, alpha=.75)
    ax.set_yscale("log")
    ax.set(xlabel="Intelligence Index supplied in the snapshot", ylabel="Actual USD per 1M tokens (log axis)", title="Use price deviation to shortlist models for task-specific tests")
    fig.colorbar(points, ax=ax, label="log2(actual / predicted price), clipped at +/-3")
    fig.text(.12, .01, "Larger points indicate faster output (size capped at 200 tokens/s).\nSource: Kaggle v1 compilation. This chart does not measure quality for your task.", fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(folder / "business_selection_map.png", bbox_inches="tight")
    plt.close(fig)

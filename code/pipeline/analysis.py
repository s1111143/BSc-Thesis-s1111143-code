"""Build result tables and figures and run the statistical tests."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, shapiro, ttest_rel, wilcoxon
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests

from common.config import (
    ANALYSIS_DIR,
    CHUNKING_STRATEGIES,
    GENERATION_MODELS,
    GENERATION_DIR,
    RETRIEVAL_DIR,
    SEED,
    ensure_directories,
)
from common.io_utils import write_json


def partial_eta_squared(f_value: float, df_effect: float, df_error: float) -> float:
    """Compute partial eta squared from F and degrees of freedom."""
    numerator = f_value * df_effect
    denominator = numerator + df_error
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def paired_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Effect size for paired t-tests (mean diff / std diff)."""
    diff = x - y
    std = diff.std(ddof=1)
    if std == 0:
        return 0.0
    return float(diff.mean() / std)


def rank_biserial_from_pairs(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial correlation from paired differences."""
    diff = x - y
    diff = diff[diff != 0]
    if diff.size == 0:
        return 0.0
    ranks = rankdata(np.abs(diff))
    pos = ranks[diff > 0].sum()
    neg = ranks[diff < 0].sum()
    return float((pos - neg) / (pos + neg))


def is_binary_metric(values: np.ndarray) -> bool:
    """True if the metric only takes values 0 or 1, like per-query Recall@5."""
    return set(np.unique(values).tolist()).issubset({0.0, 1.0})


def paired_bootstrap_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = 10000, ci: float = 95.0
) -> tuple[float, float, float]:
    """Bootstrap CI for the mean paired difference (x - y)."""
    diff = x - y
    rng = np.random.default_rng(SEED)
    n = diff.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(diff.mean()), float(lo), float(hi)


def mcnemar_pairwise(pivot_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """McNemar post-hoc for paired binary outcomes, Holm-corrected."""
    comparisons: list[dict] = []
    for left, right in combinations(list(pivot_df.columns), 2):
        x = pivot_df[left].to_numpy()
        y = pivot_df[right].to_numpy()
        both = int(np.sum((x == 1) & (y == 1)))
        left_only = int(np.sum((x == 1) & (y == 0)))
        right_only = int(np.sum((x == 0) & (y == 1)))
        neither = int(np.sum((x == 0) & (y == 0)))

        result = mcnemar([[both, left_only], [right_only, neither]], exact=True)
        prop_diff, ci_low, ci_high = paired_bootstrap_ci(x, y)
        comparisons.append(
            {
                "metric": metric_name,
                "left": str(left),
                "right": str(right),
                "test": "mcnemar_exact",
                "statistic": float(result.statistic),
                "p_raw": float(result.pvalue),
                "discordant_left_only": left_only,
                "discordant_right_only": right_only,
                "prop_diff": prop_diff,
                "prop_diff_ci_low": ci_low,
                "prop_diff_ci_high": ci_high,
            }
        )
    if comparisons:
        p_values = [row["p_raw"] for row in comparisons]
        _, adjusted, _, _ = multipletests(p_values, method="holm")
        for row, p_adj in zip(comparisons, adjusted):
            row["p_holm"] = float(p_adj)
    return pd.DataFrame(comparisons)


def wilcoxon_pairwise_with_ci(pivot_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Wilcoxon signed-rank (Pratt) post-hoc, Holm-corrected, with bootstrap CIs."""
    comparisons: list[dict] = []
    for left, right in combinations(list(pivot_df.columns), 2):
        x = pivot_df[left].to_numpy()
        y = pivot_df[right].to_numpy()
        n_tied = int(np.sum(x == y))
        stat, p_raw = wilcoxon(x, y, zero_method="pratt")
        mean_diff, ci_low, ci_high = paired_bootstrap_ci(x, y)
        comparisons.append(
            {
                "metric": metric_name,
                "left": str(left),
                "right": str(right),
                "test": "wilcoxon_signed_rank_pratt",
                "statistic": float(stat),
                "p_raw": float(p_raw),
                "rank_biserial": rank_biserial_from_pairs(x, y),
                "n_tied_pairs": n_tied,
                "mean_diff": mean_diff,
                "mean_diff_ci_low": ci_low,
                "mean_diff_ci_high": ci_high,
            }
        )
    if comparisons:
        p_values = [row["p_raw"] for row in comparisons]
        _, adjusted, _, _ = multipletests(p_values, method="holm")
        for row, p_adj in zip(comparisons, adjusted):
            row["p_holm"] = float(p_adj)
    return pd.DataFrame(comparisons)


def shapiro_normality_check(df: pd.DataFrame, metric: str, subject: str, within: list[str]) -> dict:
    """Run Shapiro-Wilk on subject-centered condition scores."""
    pivot = df.pivot_table(index=subject, columns=within, values=metric)
    centered = pivot.sub(pivot.mean(axis=1), axis=0).values.flatten()
    centered = centered[~np.isnan(centered)]
    if centered.size < 4:
        return {"status": "insufficient_data", "p_value": None}
    stat, p_value = shapiro(centered)
    return {"status": "ok", "statistic": float(stat), "p_value": float(p_value)}


def two_way_rm_anova(df: pd.DataFrame, metric: str) -> dict:
    """Run two-way repeated-measures ANOVA on raw metric values."""
    fit = AnovaRM(df, depvar=metric, subject="query_id", within=["strategy", "model"]).fit()
    table = fit.anova_table.reset_index()

    effects: dict[str, dict] = {}
    for effect_name in ["strategy", "model", "strategy:model"]:
        row = table.loc[table["index"] == effect_name].iloc[0]
        f_value = float(row["F Value"])
        df_num = float(row["Num DF"])
        df_den = float(row["Den DF"])
        effects[effect_name] = {
            "f_value": f_value,
            "num_df": df_num,
            "den_df": df_den,
            "p_value": float(row["Pr > F"]),
            "partial_eta_squared": partial_eta_squared(f_value, df_num, df_den),
        }
    return {"method": "two_way_rm_anova", "effects": effects}


def generation_simple_effects_posthoc(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Strategy-pair comparisons within each model, Holm-corrected."""
    comparisons: list[dict] = []
    model_order = [model for model in GENERATION_MODELS if model in set(df["model"].unique())]
    strategy_order = [s for s in CHUNKING_STRATEGIES if s in set(df["strategy"].unique())]

    for model_name in model_order:
        model_df = df.loc[df["model"] == model_name, ["query_id", "strategy", metric_name]]
        pivot = model_df.pivot(index="query_id", columns="strategy", values=metric_name)
        for left, right in combinations(strategy_order, 2):
            if left not in pivot.columns or right not in pivot.columns:
                continue
            paired = pivot[[left, right]].dropna()
            if paired.empty:
                continue
            x = paired[left].to_numpy()
            y = paired[right].to_numpy()

            if np.allclose(x, y):
                t_stat = 0.0
                t_p = 1.0
                w_stat = 0.0
                w_p = 1.0
            else:
                t_stat, t_p = ttest_rel(x, y, nan_policy="omit")
                w_stat, w_p = wilcoxon(x, y, zero_method="pratt")

            mean_diff, ci_low, ci_high = paired_bootstrap_ci(x, y)
            comparisons.append(
                {
                    "metric": metric_name,
                    "model": model_name,
                    "left_strategy": left,
                    "right_strategy": right,
                    "n_pairs": int(len(paired)),
                    "mean_diff": mean_diff,
                    "mean_diff_ci_low": ci_low,
                    "mean_diff_ci_high": ci_high,
                    "ttest_statistic": float(t_stat),
                    "ttest_p_raw": float(t_p),
                    "cohens_d": paired_cohens_d(x, y),
                    "wilcoxon_statistic": float(w_stat),
                    "wilcoxon_p_raw": float(w_p),
                    "rank_biserial": rank_biserial_from_pairs(x, y),
                }
            )

    if comparisons:
        t_p_values = [row["ttest_p_raw"] for row in comparisons]
        _, t_adjusted, _, _ = multipletests(t_p_values, method="holm")
        w_p_values = [row["wilcoxon_p_raw"] for row in comparisons]
        _, w_adjusted, _, _ = multipletests(w_p_values, method="holm")
        for row, t_p_holm, w_p_holm in zip(comparisons, t_adjusted, w_adjusted):
            row["ttest_p_holm"] = float(t_p_holm)
            row["wilcoxon_p_holm"] = float(w_p_holm)
    return pd.DataFrame(comparisons)


def retrieval_stats(df: pd.DataFrame, metric: str) -> tuple[dict, pd.DataFrame]:
    """Omnibus and pairwise tests for one retrieval metric."""
    pivot = df.pivot(index="query_id", columns="strategy", values=metric).dropna()
    if pivot.empty:
        return {"metric": metric, "status": "no_data"}, pd.DataFrame()

    if is_binary_metric(pivot.to_numpy()):
        q = cochrans_q(pivot.to_numpy().astype(int), return_object=True)
        n_queries = int(pivot.shape[0])
        n_conditions = int(pivot.shape[1])

        kendalls_w = float(q.statistic / (n_queries * (n_conditions - 1)))
        stats_summary = {
            "metric": metric,
            "method": "cochran_q",
            "statistic": float(q.statistic),
            "df": float(q.df),
            "p_value": float(q.pvalue),
            "kendalls_w": kendalls_w,
            "condition_means": {str(col): float(pivot[col].mean()) for col in pivot.columns},
        }
        return stats_summary, mcnemar_pairwise(pivot, metric)

    normality = shapiro_normality_check(df, metric, "query_id", ["strategy"])
    columns = list(pivot.columns)
    stat, p_value = friedmanchisquare(*[pivot[col].to_numpy() for col in columns])
    kendalls_w = float(stat / (pivot.shape[0] * (len(columns) - 1)))
    stats_summary = {
        "metric": metric,
        "method": "friedman",
        "normality": normality,
        "chi2": float(stat),
        "p_value": float(p_value),
        "kendalls_w": kendalls_w,
    }
    return stats_summary, wilcoxon_pairwise_with_ci(pivot, metric)


def art_two_way_rm(df: pd.DataFrame, metric: str) -> dict:
    """Approximate ART-style two-way repeated-measures test for non-normal metrics."""
    grand_mean = df[metric].mean()
    mean_a = df.groupby("strategy")[metric].mean().to_dict()
    mean_b = df.groupby("model")[metric].mean().to_dict()
    mean_ab = df.groupby(["strategy", "model"])[metric].mean().to_dict()

    def aligned(effect: str, row: pd.Series) -> float:
        y = row[metric]
        a = row["strategy"]
        b = row["model"]
        if effect == "strategy":
            return float(y - mean_ab[(a, b)] + mean_a[a])
        if effect == "model":
            return float(y - mean_ab[(a, b)] + mean_b[b])
        if effect == "interaction":
            return float(y - mean_a[a] - mean_b[b] + grand_mean)
        raise ValueError(effect)

    results: dict[str, dict] = {}
    for effect in ["strategy", "model", "interaction"]:
        aligned_values = df.apply(lambda row: aligned(effect, row), axis=1)
        ranked = rankdata(aligned_values)
        tmp = df.copy()
        tmp["ranked"] = ranked
        fit = AnovaRM(tmp, depvar="ranked", subject="query_id", within=["strategy", "model"]).fit()
        table = fit.anova_table.reset_index()
        key = "strategy:model" if effect == "interaction" else effect
        row = table.loc[table["index"] == key].iloc[0]
        f_value = float(row["F Value"])
        num_df = float(row["Num DF"])
        den_df = float(row["Den DF"])
        results[key] = {
            "f_value": f_value,
            "num_df": num_df,
            "den_df": den_df,
            "p_value": float(row["Pr > F"]),
            "partial_eta_squared": partial_eta_squared(f_value, num_df, den_df),
        }
    return results


def generation_stats(df: pd.DataFrame, metric: str) -> tuple[dict, pd.DataFrame]:
    """Primary two-way RM ANOVA + ART robustness check for one metric."""
    normality = shapiro_normality_check(df, metric, "query_id", ["strategy", "model"])
    summary = {
        "metric": metric,
        "normality": normality,
        "primary": two_way_rm_anova(df, metric),
        "robustness": {"method": "art_approximation", "effects": art_two_way_rm(df, metric)},
    }
    posthoc = generation_simple_effects_posthoc(df, metric)
    return summary, posthoc


def load_retrieval_per_query() -> pd.DataFrame:
    """Load per-query retrieval metrics for each chunking strategy."""
    rows: list[pd.DataFrame] = []
    for strategy in CHUNKING_STRATEGIES:
        path = RETRIEVAL_DIR / strategy / "per_query_metrics.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["strategy"] = strategy
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_ragas_rows() -> pd.DataFrame:
    """Load per-query RAGAS rows for all strategy/model combinations."""
    rows: list[pd.DataFrame] = []
    for strategy in CHUNKING_STRATEGIES:
        for model in GENERATION_MODELS:
            path = GENERATION_DIR / strategy / f"{model}_ragas.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df["strategy"] = strategy
            df["model"] = model

            rename_map = {}
            if "answer_relevance" in df.columns:
                rename_map["answer_relevance"] = "answer_relevancy"

            if "context_relevancy" in df.columns:
                rename_map["context_relevancy"] = "context_utilization"
            if "context_relevance" in df.columns:
                rename_map["context_relevance"] = "context_utilization"
            if rename_map:
                df = df.rename(columns=rename_map)
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_generation_usage_rows() -> pd.DataFrame:
    """Load latency/token logs from generated answers."""
    rows: list[dict] = []
    for strategy in CHUNKING_STRATEGIES:
        for model in GENERATION_MODELS:
            path = GENERATION_DIR / strategy / f"{model}_answers.jsonl"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    rows.append(
                        {
                            "query_id": row["query_id"],
                            "strategy": strategy,
                            "model": model,
                            "latency_s": row.get("latency_s", 0.0),
                            "prompt_tokens": row.get("prompt_tokens", 0),
                            "completion_tokens": row.get("completion_tokens", 0),
                            "total_tokens": row.get("total_tokens", 0),
                        }
                    )
    return pd.DataFrame(rows)


def plot_retrieval_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Plot mean Recall@5 and MRR@5 per strategy."""
    means = df.groupby("strategy")[["recall_at_5", "mrr_at_5"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(means["strategy"], means["recall_at_5"], color="#4e79a7")
    axes[0].set_title("Recall@5")
    axes[0].set_ylim(0, 1)
    axes[1].bar(means["strategy"], means["mrr_at_5"], color="#f28e2b")
    axes[1].set_title("MRR@5")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_latency_tokens(df: pd.DataFrame, output_path: Path) -> None:
    """Save mean generation latency and total tokens per strategy/model condition."""
    grouped = df.groupby(["strategy", "model"])[["latency_s", "total_tokens"]].mean().reset_index()
    strategies = list(grouped["strategy"].unique())
    models = list(grouped["model"].unique())
    x = np.arange(len(strategies))
    width = 0.8 / max(len(models), 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, model in enumerate(models):
        sub = grouped[grouped["model"] == model].set_index("strategy").reindex(strategies)
        axes[0].bar(x + i * width, sub["latency_s"], width, label=model)
        axes[1].bar(x + i * width, sub["total_tokens"], width, label=model)
    for ax, title in zip(axes, ["Mean latency (s)", "Mean total tokens"]):
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(strategies)
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_generation_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Plot each RAGAS metric's mean by strategy, one line per model."""
    metrics = ["faithfulness", "answer_relevancy", "context_utilization"]
    available = [metric for metric in metrics if metric in df.columns]
    if not available:
        return

    grouped = df.groupby(["strategy", "model"])[available].mean().reset_index()
    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        for model in grouped["model"].unique():
            sub = grouped[grouped["model"] == model]
            ax.plot(sub["strategy"], sub[metric], marker="o", label=model)
        ax.set_ylim(0, 1)
        ax.set_title(metric)
        ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def run_analysis() -> dict:
    """Run all analysis outputs that have required upstream files available."""
    ensure_directories()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {"retrieval": {}, "generation": {}}

    retrieval_df = load_retrieval_per_query()
    if not retrieval_df.empty:
        retrieval_means = (
            retrieval_df.groupby("strategy")[["recall_at_5", "mrr_at_5"]]
            .agg(["mean", "std", "median"])
            .reset_index()
        )
        retrieval_means.columns = [
            "_".join(col).strip("_") if isinstance(col, tuple) else col
            for col in retrieval_means.columns
        ]
        retrieval_means.to_csv(ANALYSIS_DIR / "retrieval_results_table.csv", index=False)
        plot_retrieval_summary(retrieval_df, ANALYSIS_DIR / "retrieval_metrics_plot.pdf")

        retrieval_stats_summary = {}
        for metric in ["recall_at_5", "mrr_at_5"]:
            metric_summary, posthoc = retrieval_stats(retrieval_df, metric)
            retrieval_stats_summary[metric] = metric_summary
            posthoc.to_csv(ANALYSIS_DIR / f"retrieval_posthoc_{metric}.csv", index=False)
        summary["retrieval"] = retrieval_stats_summary

    ragas_df = load_ragas_rows()
    usage_df = load_generation_usage_rows()
    if not ragas_df.empty:
        ragas_metrics = [
            metric
            for metric in ["faithfulness", "answer_relevancy", "context_utilization"]
            if metric in ragas_df.columns
        ]
        agg = ragas_df.groupby(["strategy", "model"])[ragas_metrics].agg(["mean", "std", "median"])
        agg = agg.reset_index()
        agg.columns = [
            "_".join(col).strip("_") if isinstance(col, tuple) else col for col in agg.columns
        ]
        agg.to_csv(ANALYSIS_DIR / "generation_ragas_results_table.csv", index=False)
        plot_generation_summary(ragas_df, ANALYSIS_DIR / "generation_ragas_plot.pdf")

        generation_stats_summary = {}
        for metric in ragas_metrics:
            metric_summary, posthoc = generation_stats(ragas_df[["query_id", "strategy", "model", metric]], metric)
            generation_stats_summary[metric] = metric_summary
            posthoc.to_csv(ANALYSIS_DIR / f"generation_posthoc_{metric}.csv", index=False)
        summary["generation"]["ragas"] = generation_stats_summary

    if not usage_df.empty:
        usage_table = usage_df.groupby(["strategy", "model"])[
            ["latency_s", "prompt_tokens", "completion_tokens", "total_tokens"]
        ].mean()
        usage_table = usage_table.reset_index()
        usage_table.to_csv(ANALYSIS_DIR / "generation_latency_tokens_table.csv", index=False)
        plot_latency_tokens(usage_df, ANALYSIS_DIR / "generation_latency_tokens_plot.pdf")
        summary["generation"]["usage_rows"] = int(len(usage_df))

    write_json(ANALYSIS_DIR / "analysis_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run statistical analysis and produce tables/figures.")
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    summary = run_analysis()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

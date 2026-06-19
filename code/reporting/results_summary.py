"""Consolidate retrieval and answer-quality results into one JSON."""

import json

import pandas as pd

from common.config import (
    ANALYSIS_DIR,
    CHUNKING_STRATEGIES,
    EFFICIENCY_DIR,
    GENERATION_DIR,
    GENERATION_MODELS,
    RETRIEVAL_DIR,
)

METRICS = ["faithfulness", "answer_relevancy", "context_utilization"]


def eta_magnitude(eta: float) -> str:
    """Cohen-style label for partial eta squared, so a small effect reads as small."""
    if eta < 0.01:
        return "negligible"
    if eta < 0.06:
        return "small"
    if eta < 0.14:
        return "medium"
    return "large"


def load_ragas_csv(strategy: str, model: str) -> pd.DataFrame:
    """Load one RAGAS CSV and normalize metric names used in outputs."""
    df = pd.read_csv(GENERATION_DIR / strategy / f"{model}_ragas.csv")
    rename_map = {}
    if "answer_relevance" in df.columns:
        rename_map["answer_relevance"] = "answer_relevancy"

    if "context_relevance" in df.columns:
        rename_map["context_relevance"] = "context_utilization"
    if "context_relevancy" in df.columns:
        rename_map["context_relevancy"] = "context_utilization"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def answer_quality_means() -> dict:
    """Per-condition RAGAS triad mean/median/sd over the scored (non-NaN) rows."""
    out = {}
    for strategy in CHUNKING_STRATEGIES:
        for model in GENERATION_MODELS:
            df = load_ragas_csv(strategy, model).dropna(subset=METRICS)
            out[f"{strategy}|{model}"] = {
                "n": int(len(df)),
                **{
                    m: {
                        "mean": float(df[m].mean()),
                        "median": float(df[m].median()),
                        "sd": float(df[m].std()),
                    }
                    for m in METRICS
                },
            }
    return out


def retrieval_means() -> dict:
    """Recall@5 / MRR@5 per strategy, read from the retrieval metric files."""
    out = {}
    for strategy in CHUNKING_STRATEGIES:
        metrics = json.loads((RETRIEVAL_DIR / strategy / "metrics.json").read_text())
        out[strategy] = metrics["retrieval_metrics"]
    return out


def efficiency_summary() -> dict:
    """Chunking/index/search latency per strategy, if measured."""
    path = EFFICIENCY_DIR / "efficiency_summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    summary = json.loads((ANALYSIS_DIR / "analysis_summary.json").read_text())

    two_way = summary.get("generation", {}).get("ragas", {})
    for metric_stats in two_way.values():
        for route_name in ["primary", "robustness"]:
            route = metric_stats.get(route_name, {})
            for effect in route.get("effects", {}).values():
                effect["magnitude"] = eta_magnitude(effect["partial_eta_squared"])

    results = {
        "retrieval": {
            "means": retrieval_means(),
            "tests": summary.get("retrieval", {}),
        },
        "answer_quality": {
            "condition_means": answer_quality_means(),
            "two_way_effects": two_way,
        },
        "efficiency": efficiency_summary(),
    }

    out_path = ANALYSIS_DIR / "results_summary.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

"""Generate answers from top-5 contexts and score them with RAGAS."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from common.config import (
    CACHE_DIR,
    CHUNKS_DIR,
    EMBEDDING_MODEL,
    GENERATION_DIR,
    MAX_GENERATION_CALLS,
    MAX_RAGAS_EVAL_ROWS,
    MISTRAL_GENERATION_MODEL,
    OPENAI_GENERATION_MODEL,
    PREPARED_DIR,
    RAGAS_BATCH_ROWS,
    RAGAS_JUDGE_MODEL,
    RAGAS_MAX_RETRIES,
    RAGAS_MAX_WAIT,
    RAGAS_MAX_WORKERS,
    RETRIEVAL_DIR,
    SEED,
    TEMPERATURE,
    ensure_directories,
    load_environment,
)
from common.io_utils import append_jsonl, read_json, stable_hash, write_json

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from mistralai import Mistral
except Exception:
    Mistral = None

try:
    from langchain_anthropic import ChatAnthropic
except Exception:
    ChatAnthropic = None

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    OpenAIEmbeddings = None


try:
    import lzma as _lzma
except Exception:
    try:
        import backports.lzma as _backports_lzma

        sys.modules["lzma"] = _backports_lzma
    except Exception:
        pass

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, faithfulness
    from ragas.run_config import RunConfig

    try:
        from ragas.metrics import context_relevancy as ragas_context_metric

        RAGAS_CONTEXT_METRIC_NAME = "context_relevancy"
    except Exception:
        try:
            from ragas.metrics import ContextUtilization

            ragas_context_metric = ContextUtilization()
            RAGAS_CONTEXT_METRIC_NAME = "context_utilization"
        except Exception:
            from ragas.metrics import context_precision as ragas_context_metric

            RAGAS_CONTEXT_METRIC_NAME = "context_precision"
except Exception:
    Dataset = None
    evaluate = None
    LangchainEmbeddingsWrapper = None
    LangchainLLMWrapper = None
    RunConfig = None
    answer_relevancy = None
    ragas_context_metric = None
    RAGAS_CONTEXT_METRIC_NAME = None
    faithfulness = None


MODEL_CONFIGS = {
    OPENAI_GENERATION_MODEL: {"provider": "openai"},
    MISTRAL_GENERATION_MODEL: {"provider": "mistral"},
}


def openai_available() -> bool:
    """Return True when OpenAI calls can run safely."""
    return bool(os.getenv("OPENAI_API_KEY")) and OpenAI is not None


def mistral_available() -> bool:
    """Return True when Mistral calls can run safely."""
    return bool(os.getenv("MISTRAL_API_KEY")) and Mistral is not None


def anthropic_available() -> bool:
    """Return True when Anthropic judge calls can run safely."""
    return bool(os.getenv("ANTHROPIC_API_KEY")) and ChatAnthropic is not None


def build_prompt(question: str, contexts: list[str]) -> str:
    """Build the shared QA prompt for both generator models."""
    numbered_contexts = "\n\n".join(
        [f"[Context {idx + 1}]\n{text}" for idx, text in enumerate(contexts)]
    )
    return (
        "Answer the question using only the provided context. "
        "If context is insufficient, say that clearly.\n\n"
        f"Question: {question}\n\n"
        f"{numbered_contexts}\n\n"
        "Answer:"
    )


def generation_cache_file(model_name: str, key: str) -> Path:
    """Return cache path for one generation request key."""
    return CACHE_DIR / "generation" / model_name / f"{key}.json"


def call_openai(model_name: str, prompt: str) -> dict:
    """Generate one answer with OpenAI chat completions."""
    if not openai_available():
        raise RuntimeError("OpenAI generation requested but OPENAI_API_KEY is unavailable.")
    client = OpenAI()
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        temperature=TEMPERATURE,
        seed=SEED,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_s = time.perf_counter() - started
    content = response.choices[0].message.content or ""
    usage = response.usage
    return {
        "answer": content.strip(),
        "latency_s": latency_s,
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def call_mistral(model_name: str, prompt: str) -> dict:
    """Generate one answer with Mistral chat completions."""
    if not mistral_available():
        raise RuntimeError("Mistral generation requested but MISTRAL_API_KEY is unavailable.")
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    started = time.perf_counter()
    response = client.chat.complete(
        model=model_name,
        temperature=TEMPERATURE,
        random_seed=SEED,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_s = time.perf_counter() - started
    content = response.choices[0].message.content
    usage = getattr(response, "usage", None)
    return {
        "answer": (content or "").strip(),
        "latency_s": latency_s,
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0) if usage else 0,
    }


def cached_generate(model_name: str, question: str, contexts: list[str]) -> dict:
    """Generate one answer with disk cache and retry."""
    payload = {
        "model": model_name,
        "temperature": TEMPERATURE,
        "question": question,
        "contexts": contexts,
    }
    key = stable_hash(payload)
    cache_path = generation_cache_file(model_name, key)
    if cache_path.exists():
        return read_json(cache_path)

    attempts = 0
    while True:
        try:
            if MODEL_CONFIGS[model_name]["provider"] == "openai":
                result = call_openai(model_name, build_prompt(question, contexts))
            else:
                result = call_mistral(model_name, build_prompt(question, contexts))
            result["cache_key"] = key
            write_json(cache_path, result)
            return result
        except Exception as exc:
            attempts += 1
            if attempts >= 3:
                raise RuntimeError(f"Generation failed for model {model_name}: {exc}") from exc
            time.sleep(2**attempts)


def load_chunk_map(strategy: str) -> dict[str, dict]:
    """Map chunk_id to chunk row for context reconstruction."""
    path = CHUNKS_DIR / f"{strategy}_chunks.jsonl"
    mapping: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            mapping[row["chunk_id"]] = row
    return mapping


def load_rankings(strategy: str) -> dict[str, list[dict]]:
    """Load retrieval rankings grouped by query."""
    path = RETRIEVAL_DIR / strategy / "retrieval_rankings.csv"
    grouped: dict[str, list[dict]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped.setdefault(row["query_id"], []).append(row)
    for query_id in grouped:
        grouped[query_id] = sorted(grouped[query_id], key=lambda x: int(x["rank"]))
    return grouped


def load_queries() -> dict[str, str]:
    """Load sampled query id -> text mapping."""
    path = PREPARED_DIR / "sampled_queries.jsonl"
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            mapping[row["_id"]] = row["text"]
    return mapping


def run_generation_for_strategy_and_model(strategy: str, model_name: str) -> dict:
    """Generate answers for one strategy/model pair with checkpoint resume."""
    model_dir = GENERATION_DIR / strategy
    model_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_dir / f"{model_name}_answers.jsonl"
    summary_path = model_dir / f"{model_name}_summary.json"

    if MODEL_CONFIGS[model_name]["provider"] == "openai" and not openai_available():
        summary = {
            "strategy": strategy,
            "model": model_name,
            "status": "skipped_no_openai_key",
            "output_path": str(output_path),
        }
        write_json(summary_path, summary)
        return summary

    if MODEL_CONFIGS[model_name]["provider"] == "mistral" and not mistral_available():
        summary = {
            "strategy": strategy,
            "model": model_name,
            "status": "skipped_no_mistral_key",
            "output_path": str(output_path),
        }
        write_json(summary_path, summary)
        return summary

    query_map = load_queries()
    ranking_map = load_rankings(strategy)
    chunk_map = load_chunk_map(strategy)

    completed_ids: set[str] = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                completed_ids.add(json.loads(line)["query_id"])

    pending_query_ids = [query_id for query_id in ranking_map if query_id not in completed_ids]
    if len(pending_query_ids) > MAX_GENERATION_CALLS:
        raise ValueError(
            f"Pending calls ({len(pending_query_ids)}) exceed MAX_GENERATION_CALLS={MAX_GENERATION_CALLS}."
        )

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency = 0.0
    generated_now = 0

    for query_id in pending_query_ids:
        question = query_map[query_id]
        ranked = ranking_map[query_id][:5]
        contexts = [chunk_map[row["chunk_id"]]["text"] for row in ranked]
        result = cached_generate(model_name, question, contexts)
        generated_now += 1
        total_prompt_tokens += int(result.get("prompt_tokens", 0))
        total_completion_tokens += int(result.get("completion_tokens", 0))
        total_latency += float(result.get("latency_s", 0.0))
        append_jsonl(
            output_path,
            {
                "query_id": query_id,
                "strategy": strategy,
                "model": model_name,
                "question": question,
                "contexts": contexts,
                "answer": result["answer"],
                "latency_s": result.get("latency_s", 0.0),
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "total_tokens": result.get("total_tokens", 0),
                "cache_key": result.get("cache_key"),
            },
        )

    total_rows = 0
    with output_path.open("r", encoding="utf-8") as handle:
        for _ in handle:
            total_rows += 1

    summary = {
        "strategy": strategy,
        "model": model_name,
        "status": "completed",
        "generated_now": generated_now,
        "total_rows": total_rows,
        "prompt_tokens_generated_now": total_prompt_tokens,
        "completion_tokens_generated_now": total_completion_tokens,
        "avg_latency_s_generated_now": (total_latency / generated_now) if generated_now else 0.0,
        "output_path": str(output_path),
    }
    write_json(summary_path, summary)
    return summary


def chunked(items: list, size: int):
    """Yield successive size-length slices, used as RAGAS checkpoint batches."""
    for start in range(0, len(items), size):
        yield items[start:start + size]


def compute_ragas_for_strategy_model(
    strategy: str, model_name: str, max_rows: int | None = None
) -> dict:
    """Score a strategy/model's answers with the RAGAS triad, resumable in batches."""
    output_path = GENERATION_DIR / strategy / f"{model_name}_answers.jsonl"
    ragas_path = GENERATION_DIR / strategy / f"{model_name}_ragas.csv"
    summary_path = GENERATION_DIR / strategy / f"{model_name}_ragas_summary.json"

    if not output_path.exists():
        summary = {
            "strategy": strategy,
            "model": model_name,
            "status": "skipped_no_answers",
            "context_metric_used": RAGAS_CONTEXT_METRIC_NAME,
        }
        write_json(summary_path, summary)
        return summary

    if (
        not anthropic_available()
        or not openai_available()
        or Dataset is None
        or evaluate is None
        or ragas_context_metric is None
        or LangchainEmbeddingsWrapper is None
        or LangchainLLMWrapper is None
        or RunConfig is None
        or OpenAIEmbeddings is None
    ):
        summary = {
            "strategy": strategy,
            "model": model_name,
            "status": "skipped_ragas_dependencies_or_key_missing",
            "context_metric_used": RAGAS_CONTEXT_METRIC_NAME,
        }
        write_json(summary_path, summary)
        return summary

    rows: list[dict] = []
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    if max_rows is not None:
        rows = rows[:max_rows]
    if len(rows) > MAX_RAGAS_EVAL_ROWS:
        raise ValueError(
            f"RAGAS rows ({len(rows)}) exceed MAX_RAGAS_EVAL_ROWS={MAX_RAGAS_EVAL_ROWS}."
        )

    metric_cols = ["faithfulness", "answer_relevancy", "context_relevancy"]
    scored_ids: set[str] = set()
    if ragas_path.exists():
        good = pd.read_csv(ragas_path).dropna(subset=metric_cols)
        good.to_csv(ragas_path, index=False)
        scored_ids = set(good["query_id"].astype(str))
    pending = [row for row in rows if str(row["query_id"]) not in scored_ids]

    judge_model = os.getenv("RAGAS_JUDGE_MODEL", RAGAS_JUDGE_MODEL)
    ragas_run_config = RunConfig(
        timeout=180,
        max_retries=RAGAS_MAX_RETRIES,
        max_wait=RAGAS_MAX_WAIT,
        max_workers=RAGAS_MAX_WORKERS,
    )
    ragas_llm = LangchainLLMWrapper(
        ChatAnthropic(
            model_name=judge_model,
            temperature=TEMPERATURE,
            timeout=120,
            max_retries=8,
            max_tokens_to_sample=8192,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        ),
        run_config=ragas_run_config,
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=60,
            max_retries=6,
        ),
        run_config=ragas_run_config,
    )

    columns = ["query_id", "strategy", "model", "faithfulness", "answer_relevancy", "context_relevancy"]
    for batch in chunked(pending, RAGAS_BATCH_ROWS):
        dataset = Dataset.from_dict(
            {
                "question": [row["question"] for row in batch],
                "answer": [row["answer"] for row in batch],
                "contexts": [row["contexts"] for row in batch],
            }
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, ragas_context_metric],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=ragas_run_config,
            batch_size=len(batch),
            raise_exceptions=False,
        )
        df = result.to_pandas()
        if (
            RAGAS_CONTEXT_METRIC_NAME
            and RAGAS_CONTEXT_METRIC_NAME != "context_relevancy"
            and RAGAS_CONTEXT_METRIC_NAME in df.columns
        ):
            df = df.rename(columns={RAGAS_CONTEXT_METRIC_NAME: "context_relevancy"})
        df["query_id"] = [row["query_id"] for row in batch]
        df["strategy"] = strategy
        df["model"] = model_name

        if df[metric_cols].isna().all(axis=None):
            raise RuntimeError(
                f"RAGAS judge returned only NaN for a full batch ({strategy}/{model_name}). "
                "The judge API is likely failing (e.g. Anthropic credit too low or bad key). "
                "Fix the cause and re-run; scored rows so far are kept and resumed."
            )
        df[columns].to_csv(ragas_path, mode="a", header=not ragas_path.exists(), index=False)

    full = pd.read_csv(ragas_path)
    summary = {
        "strategy": strategy,
        "model": model_name,
        "status": "completed",
        "rows": int(len(full)),
        "faithfulness_mean": float(full["faithfulness"].mean()),
        "answer_relevancy_mean": float(full["answer_relevancy"].mean()),
        "context_relevancy_mean": float(full["context_relevancy"].mean()),
        "context_metric_used": RAGAS_CONTEXT_METRIC_NAME,
        "output_path": str(ragas_path),
        "judge_model": judge_model,
    }
    write_json(summary_path, summary)
    return summary


def run_generation(strategies: list[str], models: list[str], run_ragas: bool = True) -> dict:
    """Run generation for selected strategy/model combinations."""
    ensure_directories()
    load_environment()

    summary: dict[str, dict] = {}
    for strategy in strategies:
        summary[strategy] = {
            "generation": {},
            "ragas": {},
            "ragas_context_metric_name": RAGAS_CONTEXT_METRIC_NAME,
        }
        for model_name in models:
            generation_summary = run_generation_for_strategy_and_model(strategy, model_name)
            summary[strategy]["generation"][model_name] = generation_summary
            if run_ragas:
                ragas_summary = compute_ragas_for_strategy_model(strategy, model_name)
                summary[strategy]["ragas"][model_name] = ragas_summary

    write_json(GENERATION_DIR / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run answer generation and optional RAGAS.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["fixed", "structure", "semantic"],
        choices=["fixed", "structure", "semantic"],
        help="Chunking strategies to use.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[OPENAI_GENERATION_MODEL, MISTRAL_GENERATION_MODEL],
        choices=[OPENAI_GENERATION_MODEL, MISTRAL_GENERATION_MODEL],
        help="Generator models to run.",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip RAGAS triad scoring.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_generation(
        strategies=args.strategies,
        models=args.models,
        run_ragas=not args.skip_ragas,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

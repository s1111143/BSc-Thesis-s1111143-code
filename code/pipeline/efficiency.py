"""Measure chunking, indexing, and search latency per strategy."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken

from pipeline.chunkers import chunk_doc, sentence_spans
from common.config import (
    CHUNKS_DIR,
    CHUNKING_STRATEGIES,
    EFFICIENCY_DIR,
    PREPARED_DIR,
    RETRIEVAL_DIR,
    SAMPLE_QUERY_COUNT,
    SEMANTIC_MIN_SENTENCES,
    TOP_K,
    ensure_directories,
    load_environment,
)
from common.io_utils import write_json

try:
    import faiss
except Exception:
    faiss = None


def load_subset_docs(path: Path) -> list[dict]:
    """Load subset corpus docs used for timed chunking."""
    docs: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            docs.append(json.loads(line))
    return docs


def time_chunking_strategy(
    strategy: str, docs: list[dict], encoding: tiktoken.Encoding
) -> dict[str, float | int | None]:
    """Time in-memory chunking for one strategy without writing chunk files."""
    semantic_provider = "local"
    sentences_embedded = None
    if strategy == "semantic":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is required for semantic timing in efficiency.py. "
                "Local fallback is disabled on purpose for this measurement."
            )
        semantic_provider = "openai"
        sentences_embedded = 0

    started = time.perf_counter()
    chunks_produced = 0
    for doc in docs:
        if strategy == "semantic" and sentences_embedded is not None:
            sent_spans = sentence_spans(doc["text"])
            if len(sent_spans) >= SEMANTIC_MIN_SENTENCES:
                sentences_embedded += len(sent_spans)
        try:
            rows = chunk_doc(
                doc,
                strategy=strategy,
                encoding=encoding,
                semantic_provider=semantic_provider,
            )
        except Exception as exc:
            if strategy == "semantic":
                raise RuntimeError(
                    "Semantic timing failed while calling OpenAI sentence embeddings. "
                    "The measurement must use provider='openai' with no local fallback."
                ) from exc
            raise
        chunks_produced += len(rows)
    chunk_seconds = time.perf_counter() - started

    return {
        "docs_processed": int(len(docs)),
        "chunks_produced_timed": int(chunks_produced),
        "sentences_embedded": sentences_embedded,
        "chunk_seconds": float(chunk_seconds),
    }


def chunk_token_totals(strategy: str, encoding: tiktoken.Encoding) -> dict[str, int]:
    """Count chunks and total chunk tokens from existing stored chunk JSONL."""
    path = CHUNKS_DIR / f"{strategy}_chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing chunk file: {path}")

    chunk_count = 0
    total_tokens = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            chunk_count += 1
            total_tokens += len(encoding.encode(row["text"]))
    return {"chunk_count": int(chunk_count), "total_chunk_tokens": int(total_tokens)}


def pick_embedding_cache(strategy_dir: Path, prefix: str) -> Path:
    """Pick one cached embedding matrix, preferring the *_auto.npy variant."""
    candidates = sorted(strategy_dir.glob(f"{prefix}_*.npy"))
    if not candidates:
        raise FileNotFoundError(
            f"No cached {prefix}_*.npy found in {strategy_dir}. "
            "Run retrieval.py with cached embeddings first."
        )
    auto = [path for path in candidates if path.stem.endswith("_auto")]
    return auto[0] if auto else candidates[0]


def load_cached_embeddings(strategy: str) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Load cached chunk/query embeddings that match the retrieval setup."""
    strategy_dir = RETRIEVAL_DIR / strategy
    chunk_path = pick_embedding_cache(strategy_dir, "chunk_embeddings")
    suffix = chunk_path.stem.removeprefix("chunk_embeddings_")
    query_path = strategy_dir / f"query_embeddings_{suffix}.npy"
    if not query_path.exists():
        query_path = pick_embedding_cache(strategy_dir, "query_embeddings")

    chunk_embeddings = np.load(chunk_path).astype(np.float32)
    query_embeddings = np.load(query_path).astype(np.float32)
    if query_embeddings.shape[0] != SAMPLE_QUERY_COUNT:
        raise ValueError(
            f"Expected {SAMPLE_QUERY_COUNT} query embeddings, got {query_embeddings.shape[0]} "
            f"in {query_path}."
        )
    return chunk_embeddings, query_embeddings, {
        "chunk_embeddings_file": str(chunk_path),
        "query_embeddings_file": str(query_path),
    }


def build_index_and_time(chunk_embeddings: np.ndarray) -> tuple["faiss.IndexFlatIP", float]:
    """Build an IndexFlatIP from cached chunk embeddings and return build time."""
    if faiss is None:
        raise ImportError("faiss-cpu is required for efficiency.py.")
    started = time.perf_counter()
    index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
    index.add(chunk_embeddings.astype(np.float32))
    seconds = time.perf_counter() - started
    return index, float(seconds)


def search_latency_summary(index: "faiss.IndexFlatIP", query_embeddings: np.ndarray) -> dict[str, float]:
    """Measure one-by-one top-k search latency over the 400 cached query vectors."""
    latencies_ms: list[float] = []
    for query_vector in query_embeddings:
        query_batch = query_vector.reshape(1, -1).astype(np.float32)
        started = time.perf_counter()
        index.search(query_batch, TOP_K)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "search_ms_mean": float(arr.mean()),
        "search_ms_median": float(np.median(arr)),
        "search_ms_p95": float(np.percentile(arr, 95)),
    }


def run_efficiency(strategies: list[str]) -> dict:
    """Run all efficiency measurements and write JSON + tidy CSV outputs."""
    ensure_directories()
    load_environment()
    EFFICIENCY_DIR.mkdir(parents=True, exist_ok=True)
    encoding = tiktoken.get_encoding("cl100k_base")

    subset_docs_path = PREPARED_DIR / "subset_docs.jsonl"
    if not subset_docs_path.exists():
        raise FileNotFoundError(f"Missing prepared subset docs: {subset_docs_path}")
    docs = load_subset_docs(subset_docs_path)

    summary: dict[str, dict] = {
        "sample_query_count": SAMPLE_QUERY_COUNT,
        "top_k": TOP_K,
        "strategies": {},
    }
    table_rows: list[dict] = []

    for strategy in strategies:
        chunk_timing = time_chunking_strategy(strategy, docs, encoding)
        token_totals = chunk_token_totals(strategy, encoding)
        chunk_embeddings, query_embeddings, cache_files = load_cached_embeddings(strategy)
        index, index_build_seconds = build_index_and_time(chunk_embeddings)
        search_summary = search_latency_summary(index, query_embeddings)

        strategy_summary = {
            **chunk_timing,
            **token_totals,
            "index_build_seconds": index_build_seconds,
            **search_summary,
            "cache_files": cache_files,
        }
        summary["strategies"][strategy] = strategy_summary

        table_rows.append(
            {
                "strategy": strategy,
                "chunk_seconds": strategy_summary["chunk_seconds"],
                "sentences_embedded": strategy_summary["sentences_embedded"],
                "chunk_count": strategy_summary["chunk_count"],
                "total_chunk_tokens": strategy_summary["total_chunk_tokens"],
                "index_build_seconds": strategy_summary["index_build_seconds"],
                "search_ms_mean": strategy_summary["search_ms_mean"],
                "search_ms_median": strategy_summary["search_ms_median"],
                "search_ms_p95": strategy_summary["search_ms_p95"],
            }
        )

    write_json(EFFICIENCY_DIR / "efficiency_summary.json", summary)
    pd.DataFrame(table_rows).to_csv(EFFICIENCY_DIR / "efficiency_table.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure chunking/indexing/retrieval latency by strategy.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=CHUNKING_STRATEGIES,
        choices=CHUNKING_STRATEGIES,
        help="Chunking strategies to measure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_efficiency(args.strategies)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

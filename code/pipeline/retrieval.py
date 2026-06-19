"""Embed chunks, build FAISS indexes, and score Recall@5 and MRR@5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from common.config import (
    CACHE_DIR,
    CHUNKS_DIR,
    DEFAULT_EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    PREPARED_DIR,
    RETRIEVAL_DIR,
    TOP_K,
    ensure_directories,
    load_environment,
)
from common.io_utils import read_tsv, write_json

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import faiss
except Exception:
    faiss = None


def spans_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    """Return True when two half-open spans overlap."""
    return max(start_a, start_b) < min(end_a, end_b)


def chunk_is_relevant(chunk_row: dict, gold_spans_by_doc: dict[str, list[tuple[int, int]]]) -> bool:
    """A chunk is relevant iff it overlaps any gold passage span in the same doc."""
    doc_spans = gold_spans_by_doc.get(chunk_row["doc_id"], [])
    for start_char, end_char in doc_spans:
        if spans_overlap(
            int(chunk_row["start_char"]),
            int(chunk_row["end_char"]),
            int(start_char),
            int(end_char),
        ):
            return True
    return False


def deterministic_local_embedding(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic no-API embedding for offline smoke tests."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16)
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def openai_embed_texts(texts: list[str], model: str, batch_size: int) -> np.ndarray:
    """Batch OpenAI embeddings with lightweight retry."""
    if OpenAI is None:
        raise ImportError("openai package is not available.")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing.")
    client = OpenAI()

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        attempts = 0
        while True:
            try:
                response = client.embeddings.create(model=model, input=batch)
                vectors.extend([row.embedding for row in response.data])
                break
            except Exception:
                attempts += 1
                if attempts >= 3:
                    raise
                time.sleep(2**attempts)
    matrix = np.asarray(vectors, dtype=np.float32)
    return normalize_rows(matrix)


def local_embed_texts(texts: list[str], dim: int = 256) -> np.ndarray:
    """Embed texts deterministically without external API calls."""
    matrix = np.vstack([deterministic_local_embedding(text, dim=dim) for text in texts])
    return matrix.astype(np.float32)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize embedding vectors for cosine/IP FAISS search."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def cache_key(texts: list[str], provider: str, model: str) -> str:
    """Stable key for embedding cache files."""
    payload = {
        "provider": provider,
        "model": model,
        "count": len(texts),
        "digest": hashlib.sha256("||".join(texts).encode("utf-8")).hexdigest(),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def embed_texts_with_cache(
    texts: list[str],
    cache_path: Path,
    provider: str = "auto",
    model: str = EMBEDDING_MODEL,
    force: bool = False,
) -> tuple[np.ndarray, str]:
    """Return embeddings from cache or compute them using selected provider."""
    if cache_path.exists() and not force:
        matrix = np.load(cache_path)
        if matrix.shape[0] == len(texts):
            used_provider = "cached"
            return matrix.astype(np.float32), used_provider

    if provider == "openai":
        matrix = openai_embed_texts(texts, model=model, batch_size=DEFAULT_EMBED_BATCH_SIZE)
        used_provider = "openai"
    elif provider == "local":
        matrix = local_embed_texts(texts)
        used_provider = "local"
    elif provider == "auto":
        if os.getenv("OPENAI_API_KEY"):
            matrix = openai_embed_texts(texts, model=model, batch_size=DEFAULT_EMBED_BATCH_SIZE)
            used_provider = "openai"
        else:
            matrix = local_embed_texts(texts)
            used_provider = "local"
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, matrix)
    return matrix, used_provider


def load_chunks(path: Path) -> list[dict]:
    """Load chunk JSONL rows into memory (subset-scale experiment)."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def load_queries(path: Path) -> list[dict]:
    """Load sampled query JSONL rows."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def build_gold_spans(sampled_qrels_path: Path) -> dict[str, dict[str, list[tuple[int, int]]]]:
    """Map query -> doc -> list of gold passage spans."""
    rows = read_tsv(sampled_qrels_path)
    mapping: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for row in rows:
        query_id = row["query-id"]
        doc_id = row["doc-id"]
        span = (int(row["start-char"]), int(row["end-char"]))
        mapping.setdefault(query_id, {}).setdefault(doc_id, []).append(span)
    return mapping


def write_retrieval_rows(path: Path, rows: list[dict]) -> None:
    """Write retrieval rankings to CSV for analysis and debugging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "rank",
        "chunk_id",
        "doc_id",
        "score",
        "is_relevant",
        "strategy",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_ranked_results(
    rows: list[dict],
    query_ids: list[str],
) -> dict:
    """Compute Recall@5 and MRR@5 from ranked rows."""
    by_query: dict[str, list[dict]] = {qid: [] for qid in query_ids}
    for row in rows:
        by_query[row["query_id"]].append(row)

    recall_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    per_query_rows: list[dict] = []
    for query_id in query_ids:
        ranked = sorted(by_query.get(query_id, []), key=lambda r: int(r["rank"]))
        relevant_ranks = [int(r["rank"]) for r in ranked if int(r["is_relevant"]) == 1]
        recall = 1.0 if relevant_ranks else 0.0
        rr = 1.0 / min(relevant_ranks) if relevant_ranks else 0.0
        recall_scores.append(recall)
        reciprocal_ranks.append(rr)
        per_query_rows.append({"query_id": query_id, "recall_at_5": recall, "mrr_at_5": rr})

    return {
        "recall_at_5": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "mrr_at_5": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "query_count": len(query_ids),
        "per_query": per_query_rows,
    }


def build_and_search_index(
    strategy: str,
    chunk_rows: list[dict],
    query_rows: list[dict],
    gold_spans: dict[str, dict[str, list[tuple[int, int]]]],
    provider: str,
    force: bool,
) -> dict:
    """Embed chunks, build FAISS index, retrieve top-k, and score relevance."""
    if faiss is None:
        raise ImportError("faiss-cpu is required for retrieval indexing.")
    strategy_dir = RETRIEVAL_DIR / strategy
    strategy_dir.mkdir(parents=True, exist_ok=True)

    chunk_texts = [row["text"] for row in chunk_rows]
    query_texts = [row["text"] for row in query_rows]
    query_ids = [row["_id"] for row in query_rows]

    chunk_cache = strategy_dir / f"chunk_embeddings_{provider}.npy"
    query_cache = strategy_dir / f"query_embeddings_{provider}.npy"

    chunk_embeddings, used_provider_chunks = embed_texts_with_cache(
        chunk_texts, chunk_cache, provider=provider, force=force
    )
    query_embeddings, used_provider_queries = embed_texts_with_cache(
        query_texts, query_cache, provider=provider, force=force
    )

    index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
    index.add(chunk_embeddings.astype(np.float32))
    index_path = strategy_dir / "index.faiss"
    faiss.write_index(index, str(index_path))

    scores, indices = index.search(query_embeddings.astype(np.float32), TOP_K)
    ranked_rows: list[dict] = []
    for query_idx, query_id in enumerate(query_ids):
        doc_gold_spans = gold_spans.get(query_id, {})
        for rank in range(TOP_K):
            chunk_idx = int(indices[query_idx, rank])
            if chunk_idx < 0:
                continue
            chunk = chunk_rows[chunk_idx]
            is_rel = int(chunk_is_relevant(chunk, doc_gold_spans))
            ranked_rows.append(
                {
                    "query_id": query_id,
                    "rank": rank + 1,
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "score": float(scores[query_idx, rank]),
                    "is_relevant": is_rel,
                    "strategy": strategy,
                }
            )

    rankings_path = strategy_dir / "retrieval_rankings.csv"
    write_retrieval_rows(rankings_path, ranked_rows)

    metrics = evaluate_ranked_results(ranked_rows, query_ids)
    per_query_path = strategy_dir / "per_query_metrics.csv"
    with per_query_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "recall_at_5", "mrr_at_5"])
        writer.writeheader()
        writer.writerows(metrics["per_query"])

    summary = {
        "strategy": strategy,
        "embedding_provider_chunks": used_provider_chunks,
        "embedding_provider_queries": used_provider_queries,
        "retrieval_metrics": {
            "recall_at_5": metrics["recall_at_5"],
            "mrr_at_5": metrics["mrr_at_5"],
            "query_count": metrics["query_count"],
        },
        "files": {
            "faiss_index": str(index_path),
            "rankings": str(rankings_path),
            "per_query_metrics": str(per_query_path),
        },
    }
    write_json(strategy_dir / "metrics.json", summary)
    return summary


def run_retrieval(strategies: list[str], provider: str = "auto", force: bool = False) -> dict:
    """Run retrieval stage for one or more chunking strategies."""
    ensure_directories()
    load_environment()
    (CACHE_DIR / "retrieval").mkdir(parents=True, exist_ok=True)

    sampled_queries_path = PREPARED_DIR / "sampled_queries.jsonl"
    sampled_qrels_path = PREPARED_DIR / "sampled_qrels.tsv"
    if not sampled_queries_path.exists() or not sampled_qrels_path.exists():
        raise FileNotFoundError("Prepared files missing. Run data_prep.py first.")

    query_rows = load_queries(sampled_queries_path)
    gold_spans = build_gold_spans(sampled_qrels_path)

    overall: dict[str, dict] = {}
    for strategy in strategies:
        chunk_path = CHUNKS_DIR / f"{strategy}_chunks.jsonl"
        if not chunk_path.exists():
            raise FileNotFoundError(f"{chunk_path} is missing. Run chunkers.py first.")
        chunk_rows = load_chunks(chunk_path)
        overall[strategy] = build_and_search_index(
            strategy=strategy,
            chunk_rows=chunk_rows,
            query_rows=query_rows,
            gold_spans=gold_spans,
            provider=provider,
            force=force,
        )

    write_json(RETRIEVAL_DIR / "summary.json", overall)
    return overall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval and compute Recall@5/MRR@5.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["fixed", "structure", "semantic"],
        choices=["fixed", "structure", "semantic"],
        help="Chunking strategies to evaluate.",
    )
    parser.add_argument(
        "--embedding-provider",
        default="auto",
        choices=["auto", "openai", "local"],
        help="Provider for retrieval embeddings.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute embeddings and metrics even if cached.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_retrieval(
        strategies=args.strategies,
        provider=args.embedding_provider,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

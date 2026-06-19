"""Rebuild NQ articles from passages and build the sampled subset corpus."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common.config import (
    CORPUS_PATH,
    NEGATIVE_TO_POSITIVE_RATIO,
    PREPARED_DIR,
    QRELS_PATH,
    QUERIES_PATH,
    SAMPLE_QUERY_COUNT,
    SEED,
    ensure_directories,
    seed_everything,
)
from common.io_utils import read_tsv, write_json, write_jsonl, write_tsv


def suffix_prefix_overlap(left: str, right: str, max_window: int = 300) -> int:
    """Return overlap length where a suffix of left equals a prefix of right."""
    max_len = min(len(left), len(right), max_window)
    for size in range(max_len, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def reconstruct_documents(
    corpus_path: Path,
    reconstructed_path: Path,
    provenance_path: Path,
) -> dict:
    """Rebuild article-level docs by contiguous title runs and save provenance."""
    seen_titles: set[str] = set()

    current_title: str | None = None
    current_doc_id: str | None = None
    current_text = ""
    current_passages: list[dict] = []
    doc_index = -1
    overlap_chars_trimmed = 0
    document_count = 0
    passage_count = 0

    reconstructed_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)

    with reconstructed_path.open("w", encoding="utf-8") as doc_out:
        with provenance_path.open("w", encoding="utf-8") as prov_out:

            def flush_current() -> None:
                nonlocal document_count, passage_count
                if current_title is None or current_doc_id is None:
                    return

                doc_row = {
                    "doc_id": current_doc_id,
                    "title": current_title,
                    "text": current_text,
                    "passages": current_passages,
                }
                doc_out.write(json.dumps(doc_row, ensure_ascii=False) + "\n")
                document_count += 1

                for passage in current_passages:
                    prov_row = {
                        "passage_id": passage["passage_id"],
                        "doc_id": current_doc_id,
                        "title": current_title,
                        "start_char": passage["start_char"],
                        "end_char": passage["end_char"],
                    }
                    prov_out.write(json.dumps(prov_row, ensure_ascii=False) + "\n")
                    passage_count += 1

            with corpus_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    title = row["title"]
                    passage_id = row["_id"]
                    passage_text = row["text"]

                    if title != current_title:
                        flush_current()
                        if title in seen_titles:
                            raise ValueError(
                                f"Title '{title}' appears in multiple non-contiguous blocks."
                            )
                        seen_titles.add(title)
                        doc_index += 1
                        current_title = title
                        current_doc_id = f"article_{doc_index:06d}"
                        current_text = ""
                        current_passages = []

                    overlap = suffix_prefix_overlap(current_text, passage_text)
                    if overlap == 0 and current_text and not current_text.endswith((" ", "\n")):
                        current_text += " "

                    start_char = len(current_text) - overlap
                    if overlap:
                        overlap_chars_trimmed += overlap
                    current_text += passage_text[overlap:]
                    end_char = start_char + len(passage_text)

                    if current_text[start_char:end_char] != passage_text:
                        raise ValueError(
                            f"Passage span mismatch for {passage_id}; reconstruction not lossless."
                        )

                    current_passages.append(
                        {
                            "passage_id": passage_id,
                            "start_char": start_char,
                            "end_char": end_char,
                        }
                    )

            flush_current()

    return {
        "document_count": document_count,
        "passage_count": passage_count,
        "contiguous_titles_verified": True,
        "lossless_span_check_verified": True,
        "overlap_chars_trimmed": overlap_chars_trimmed,
        "reconstructed_path": str(reconstructed_path),
        "provenance_path": str(provenance_path),
    }


def load_queries(queries_path: Path) -> list[dict]:
    """Load all query rows from BEIR query JSONL."""
    rows: list[dict] = []
    with queries_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def sample_queries(all_queries: list[dict], sample_count: int, seed: int) -> list[dict]:
    """Return a deterministic random query sample used by all conditions."""
    if sample_count > len(all_queries):
        raise ValueError(
            f"Requested {sample_count} queries but only {len(all_queries)} available."
        )
    rng = random.Random(seed)
    sampled = rng.sample(all_queries, sample_count)
    sampled.sort(key=lambda row: row["_id"])
    return sampled


def build_passage_lookup(provenance_path: Path, wanted_ids: set[str]) -> dict[str, dict]:
    """Load only needed passage provenance rows (qrels are much smaller than corpus)."""
    found: dict[str, dict] = {}
    with provenance_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pid = row["passage_id"]
            if pid in wanted_ids:
                found[pid] = row
    return found


def select_subset_docs(
    reconstructed_path: Path,
    selected_doc_ids: set[str],
    subset_path: Path,
) -> int:
    """Write subset corpus docs by selected reconstructed doc ids."""
    kept = 0
    with reconstructed_path.open("r", encoding="utf-8") as source:
        with subset_path.open("w", encoding="utf-8") as target:
            for line in source:
                row = json.loads(line)
                if row["doc_id"] in selected_doc_ids:
                    kept += 1
                    target.write(json.dumps(row, ensure_ascii=False) + "\n")
    return kept


def run_data_prep(force: bool = False) -> dict:
    """Run the full preparation stage and return a metadata summary."""
    ensure_directories()
    seed_everything(SEED)

    reconstructed_path = PREPARED_DIR / "reconstructed_docs.jsonl"
    provenance_path = PREPARED_DIR / "passage_provenance.jsonl"
    sampled_queries_path = PREPARED_DIR / "sampled_queries.jsonl"
    sampled_qrels_path = PREPARED_DIR / "sampled_qrels.tsv"
    subset_docs_path = PREPARED_DIR / "subset_docs.jsonl"
    metadata_path = PREPARED_DIR / "prep_metadata.json"

    if force or not reconstructed_path.exists() or not provenance_path.exists():
        reconstruction_summary = reconstruct_documents(
            CORPUS_PATH, reconstructed_path, provenance_path
        )
    else:
        reconstruction_summary = {
            "reused_cached_reconstruction": True,
            "reconstructed_path": str(reconstructed_path),
            "provenance_path": str(provenance_path),
        }

    queries = load_queries(QUERIES_PATH)
    sampled_queries = sample_queries(queries, SAMPLE_QUERY_COUNT, SEED)
    write_jsonl(sampled_queries_path, sampled_queries)
    sampled_query_ids = {q["_id"] for q in sampled_queries}

    qrels_rows = read_tsv(QRELS_PATH)
    sampled_qrels = [
        row
        for row in qrels_rows
        if row["query-id"] in sampled_query_ids and float(row["score"]) > 0
    ]
    if not sampled_qrels:
        raise ValueError("No positive qrels found for sampled query set.")

    wanted_passages = {row["corpus-id"] for row in sampled_qrels}
    passage_lookup = build_passage_lookup(provenance_path, wanted_passages)
    if len(passage_lookup) != len(wanted_passages):
        missing = wanted_passages - set(passage_lookup)
        raise ValueError(
            f"Missing provenance for {len(missing)} gold passages, e.g. {sorted(list(missing))[:5]}"
        )

    enriched_qrels: list[dict] = []
    positive_doc_ids: set[str] = set()
    for row in sampled_qrels:
        prov = passage_lookup[row["corpus-id"]]
        positive_doc_ids.add(prov["doc_id"])
        enriched_qrels.append(
            {
                "query-id": row["query-id"],
                "corpus-id": row["corpus-id"],
                "score": row["score"],
                "doc-id": prov["doc_id"],
                "start-char": prov["start_char"],
                "end-char": prov["end_char"],
            }
        )
    write_tsv(
        sampled_qrels_path,
        enriched_qrels,
        ["query-id", "corpus-id", "score", "doc-id", "start-char", "end-char"],
    )

    all_doc_ids: list[str] = []
    with reconstructed_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            all_doc_ids.append(row["doc_id"])

    candidate_negatives = [doc_id for doc_id in all_doc_ids if doc_id not in positive_doc_ids]
    rng = random.Random(SEED)
    negative_count = min(
        int(len(positive_doc_ids) * NEGATIVE_TO_POSITIVE_RATIO),
        len(candidate_negatives),
    )
    sampled_negatives = set(rng.sample(candidate_negatives, negative_count))
    subset_doc_ids = positive_doc_ids | sampled_negatives
    kept_docs = select_subset_docs(reconstructed_path, subset_doc_ids, subset_docs_path)

    metadata = {
        "seed": SEED,
        "sample_query_count": len(sampled_queries),
        "positive_qrel_rows": len(enriched_qrels),
        "positive_docs": len(positive_doc_ids),
        "sampled_negative_docs": len(sampled_negatives),
        "subset_docs": kept_docs,
        "negative_to_positive_ratio": NEGATIVE_TO_POSITIVE_RATIO,
        "files": {
            "sampled_queries": str(sampled_queries_path),
            "sampled_qrels": str(sampled_qrels_path),
            "subset_docs": str(subset_docs_path),
        },
        "reconstruction": reconstruction_summary,
    }
    write_json(metadata_path, metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare reconstructed NQ subset for RAG study.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild reconstruction outputs even if cached files exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_data_prep(force=args.force)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Fixed-size, structure-aware, and semantic chunkers with provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import tiktoken

from common.config import (
    CHUNKS_DIR,
    FIXED_CHUNK_SIZE_TOKENS,
    OVERLAP_RATIO,
    PREPARED_DIR,
    SEMANTIC_BREAKPOINT_PERCENTILE,
    SEMANTIC_MIN_SENTENCES,
    STRUCTURE_HARD_CAP_TOKENS,
    STRUCTURE_SOFT_TARGET_TOKENS,
    ensure_directories,
    load_environment,
)

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def spans_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    """Return True if two half-open character spans overlap."""
    return max(start_a, start_b) < min(end_a, end_b)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split text into sentence-like spans using lightweight punctuation regex."""
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", text, flags=re.DOTALL):
        start, end = match.span()
        chunk = text[start:end].strip()
        if chunk:
            left_trim = len(text[start:end]) - len(text[start:end].lstrip())
            right_trim = len(text[start:end]) - len(text[start:end].rstrip())
            spans.append((start + left_trim, end - right_trim))
    if not spans and text.strip():
        stripped = text.strip()
        start = text.find(stripped)
        spans = [(start, start + len(stripped))]
    return spans


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Split by paragraph boundaries first, then fallback to sentence boundaries."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for block in re.split(r"\n{2,}", text):
        block = block.strip()
        if not block:
            cursor += 1
            continue
        start = text.find(block, cursor)
        end = start + len(block)
        spans.append((start, end))
        cursor = end
    if len(spans) <= 1:
        return sentence_spans(text)
    return spans


def deterministic_local_embedding(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic no-API embedding used for offline tests and no-key fallback."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16)
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def openai_embeddings(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Fetch embeddings from OpenAI and return normalized vectors."""
    if OpenAI is None:
        raise ImportError("openai package is not available.")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing.")
    client = OpenAI()
    response = client.embeddings.create(model=model, input=texts)
    vectors = np.asarray([row.embedding for row in response.data], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def semantic_sentence_embeddings(sentences: list[str], provider: str = "auto") -> np.ndarray:
    """Choose OpenAI or local deterministic embeddings for semantic chunking."""
    if provider == "openai":
        return openai_embeddings(sentences)
    if provider == "local":
        vectors = np.vstack([deterministic_local_embedding(sentence) for sentence in sentences])
        return vectors.astype(np.float32)
    if provider == "auto":
        if os.getenv("OPENAI_API_KEY"):
            return openai_embeddings(sentences)
        vectors = np.vstack([deterministic_local_embedding(sentence) for sentence in sentences])
        return vectors.astype(np.float32)
    raise ValueError(f"Unknown semantic embedding provider: {provider}")


def prefix_char_positions(tokens: list[int], encoding: tiktoken.Encoding) -> list[int]:
    """Map token boundaries to character positions in decoded text."""
    positions = [0]
    cursor = 0
    for token in tokens:
        piece = encoding.decode([token])
        cursor += len(piece)
        positions.append(cursor)
    return positions


def collect_provenance_for_span(passages: list[dict], start: int, end: int) -> list[dict]:
    """Collect source passages whose spans overlap a chunk span."""
    overlap_rows: list[dict] = []
    for passage in passages:
        p_start = int(passage["start_char"])
        p_end = int(passage["end_char"])
        if spans_overlap(start, end, p_start, p_end):
            overlap_rows.append(
                {
                    "passage_id": passage["passage_id"],
                    "start_char": p_start,
                    "end_char": p_end,
                }
            )
    return overlap_rows


def make_chunk_row(
    strategy: str,
    doc: dict,
    chunk_index: int,
    start_char: int,
    end_char: int,
    encoding: tiktoken.Encoding,
) -> dict:
    """Build one chunk row with text slice, token count, and provenance."""
    text = doc["text"][start_char:end_char]
    source_passages = collect_provenance_for_span(doc["passages"], start_char, end_char)
    return {
        "chunk_id": f"{strategy}_{doc['doc_id']}_{chunk_index:04d}",
        "strategy": strategy,
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "text": text,
        "start_char": start_char,
        "end_char": end_char,
        "token_count": len(encoding.encode(text)),
        "source_passage_ids": [row["passage_id"] for row in source_passages],
        "source_passages": source_passages,
    }


def apply_boundary_overlap(
    spans: list[tuple[int, int]],
    text: str,
    encoding: tiktoken.Encoding,
    overlap_tokens: int,
) -> list[tuple[int, int]]:
    """Extend each non-first span leftward so boundary facts survive in one chunk."""
    if not spans:
        return spans

    overlapped = [spans[0]]
    for idx in range(1, len(spans)):
        start_char, end_char = spans[idx]
        left_text = text[:start_char]
        left_tokens = encoding.encode(left_text)
        if len(left_tokens) <= overlap_tokens:
            new_start = 0
        else:
            trimmed = encoding.decode(left_tokens[:-overlap_tokens])
            new_start = len(trimmed)
        overlapped.append((new_start, end_char))
    return overlapped


def fixed_size_spans(text: str, encoding: tiktoken.Encoding) -> list[tuple[int, int]]:
    """Chunk by fixed token windows with 15% overlap."""
    tokens = encoding.encode(text)
    if not tokens:
        return []
    overlap_tokens = int(FIXED_CHUNK_SIZE_TOKENS * OVERLAP_RATIO)
    stride = max(FIXED_CHUNK_SIZE_TOKENS - overlap_tokens, 1)
    positions = prefix_char_positions(tokens, encoding)

    spans: list[tuple[int, int]] = []
    start_token = 0
    while start_token < len(tokens):
        end_token = min(start_token + FIXED_CHUNK_SIZE_TOKENS, len(tokens))
        spans.append((positions[start_token], positions[end_token]))
        if end_token == len(tokens):
            break
        start_token += stride
    return spans


def split_span_by_token_cap(
    text: str,
    span: tuple[int, int],
    cap_tokens: int,
    encoding: tiktoken.Encoding,
) -> list[tuple[int, int]]:
    """Force split one large coherent unit when it exceeds hard token cap."""
    start, end = span
    part = text[start:end]
    tokens = encoding.encode(part)
    if len(tokens) <= cap_tokens:
        return [span]

    positions = prefix_char_positions(tokens, encoding)
    split_spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(tokens):
        next_cursor = min(cursor + cap_tokens, len(tokens))
        split_spans.append((start + positions[cursor], start + positions[next_cursor]))
        cursor = next_cursor
    return split_spans


def merge_units_to_soft_target(
    text: str,
    units: list[tuple[int, int]],
    soft_target: int,
    hard_cap: int,
    encoding: tiktoken.Encoding,
) -> list[tuple[int, int]]:
    """Merge coherent units toward soft target and only hard split if necessary."""
    if not units:
        return []

    spans: list[tuple[int, int]] = []
    current_start, current_end = units[0]
    current_tokens = len(encoding.encode(text[current_start:current_end]))
    if current_tokens > hard_cap:
        split_first = split_span_by_token_cap(text, (current_start, current_end), hard_cap, encoding)
        spans.extend(split_first[:-1])
        current_start, current_end = split_first[-1]
        current_tokens = len(encoding.encode(text[current_start:current_end]))

    for unit_start, unit_end in units[1:]:
        unit_tokens = len(encoding.encode(text[unit_start:unit_end]))
        if unit_tokens > hard_cap:
            spans.append((current_start, current_end))
            spans.extend(split_span_by_token_cap(text, (unit_start, unit_end), hard_cap, encoding))
            current_start, current_end = unit_end, unit_end
            current_tokens = 0
            continue

        merged_tokens = current_tokens + unit_tokens
        can_still_merge = merged_tokens <= hard_cap
        should_merge = merged_tokens <= soft_target or current_tokens < (0.6 * soft_target)

        if can_still_merge and should_merge:
            current_end = unit_end
            current_tokens = merged_tokens
        else:
            if current_end > current_start:
                spans.append((current_start, current_end))
            current_start, current_end = unit_start, unit_end
            current_tokens = unit_tokens

    if current_end > current_start:
        spans.append((current_start, current_end))

    return spans


def structure_aware_spans(text: str, encoding: tiktoken.Encoding) -> list[tuple[int, int]]:
    """Structure-aware chunking over paragraph/sentence units + boundary overlap."""
    units = paragraph_spans(text)
    base_spans = merge_units_to_soft_target(
        text,
        units,
        STRUCTURE_SOFT_TARGET_TOKENS,
        STRUCTURE_HARD_CAP_TOKENS,
        encoding,
    )
    overlap_tokens = int(STRUCTURE_SOFT_TARGET_TOKENS * OVERLAP_RATIO)
    return apply_boundary_overlap(base_spans, text, encoding, overlap_tokens)


def semantic_units(text: str, provider: str) -> list[tuple[int, int]]:
    """Build semantic units by sentence similarity breakpoints."""
    sent_spans = sentence_spans(text)
    if len(sent_spans) < SEMANTIC_MIN_SENTENCES:
        return sent_spans

    sentences = [text[start:end] for start, end in sent_spans]
    vectors = semantic_sentence_embeddings(sentences, provider=provider)
    similarities = np.sum(vectors[:-1] * vectors[1:], axis=1)
    threshold = float(np.percentile(similarities, SEMANTIC_BREAKPOINT_PERCENTILE))

    units: list[tuple[int, int]] = []
    current_start = sent_spans[0][0]
    current_end = sent_spans[0][1]
    for idx, sim in enumerate(similarities):
        next_span = sent_spans[idx + 1]
        if sim <= threshold:
            units.append((current_start, current_end))
            current_start, current_end = next_span
        else:
            current_end = next_span[1]
    units.append((current_start, current_end))
    return units


def semantic_spans(
    text: str,
    encoding: tiktoken.Encoding,
    provider: str = "auto",
) -> list[tuple[int, int]]:
    """Semantic chunking with embedding breakpoints + soft/hard token controls."""
    units = semantic_units(text, provider=provider)
    base_spans = merge_units_to_soft_target(
        text,
        units,
        STRUCTURE_SOFT_TARGET_TOKENS,
        STRUCTURE_HARD_CAP_TOKENS,
        encoding,
    )
    overlap_tokens = int(STRUCTURE_SOFT_TARGET_TOKENS * OVERLAP_RATIO)
    return apply_boundary_overlap(base_spans, text, encoding, overlap_tokens)


def chunk_doc(doc: dict, strategy: str, encoding: tiktoken.Encoding, semantic_provider: str) -> list[dict]:
    """Chunk one reconstructed document with the selected strategy."""
    text = doc["text"]
    if strategy == "fixed":
        spans = fixed_size_spans(text, encoding)
    elif strategy == "structure":
        spans = structure_aware_spans(text, encoding)
    elif strategy == "semantic":
        spans = semantic_spans(text, encoding, provider=semantic_provider)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    rows: list[dict] = []
    for idx, (start_char, end_char) in enumerate(spans):
        if end_char <= start_char:
            continue
        row = make_chunk_row(strategy, doc, idx, start_char, end_char, encoding)
        if row["text"].strip():
            rows.append(row)
    return rows


def run_chunking(
    strategies: list[str],
    semantic_provider: str = "auto",
    force: bool = False,
) -> dict:
    """Run requested chunkers over subset corpus and write chunk JSONL files."""
    ensure_directories()
    load_environment()
    encoding = tiktoken.get_encoding("cl100k_base")

    subset_docs_path = PREPARED_DIR / "subset_docs.jsonl"
    if not subset_docs_path.exists():
        raise FileNotFoundError(
            f"{subset_docs_path} does not exist. Run data_prep.py first."
        )

    summary: dict[str, dict] = {}
    for strategy in strategies:
        output_path = CHUNKS_DIR / f"{strategy}_chunks.jsonl"
        if output_path.exists() and not force:
            summary[strategy] = {"skipped_cached": True, "output_path": str(output_path)}
            continue

        total_chunks = 0
        total_docs = 0
        with subset_docs_path.open("r", encoding="utf-8") as source:
            with output_path.open("w", encoding="utf-8") as target:
                for line in source:
                    doc = json.loads(line)
                    rows = chunk_doc(doc, strategy, encoding, semantic_provider)
                    total_docs += 1
                    total_chunks += len(rows)
                    for row in rows:
                        target.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary[strategy] = {
            "documents_processed": total_docs,
            "chunks_written": total_chunks,
            "output_path": str(output_path),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chunk corpora for each strategy.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["fixed", "structure", "semantic"],
        choices=["fixed", "structure", "semantic"],
        help="Chunking strategies to run.",
    )
    parser.add_argument(
        "--semantic-provider",
        default="auto",
        choices=["auto", "openai", "local"],
        help="Sentence embedding provider for semantic breakpoints.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute chunk files even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_chunking(
        strategies=args.strategies,
        semantic_provider=args.semantic_provider,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

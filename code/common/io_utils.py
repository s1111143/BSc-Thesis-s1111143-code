"""JSONL, CSV, and JSON read/write helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> list[dict]:
    """Return JSONL rows as a list of dictionaries."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write rows to JSONL with one compact JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    """Append one JSON object to an existing JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a TSV file into a list of string dictionaries."""
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write dictionaries to TSV in the provided column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    """Write a JSON file with readable indentation for auditability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict:
    """Read a JSON dictionary file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stable_hash(payload: dict | str) -> str:
    """Return a deterministic SHA256 hash for cache keys."""
    if isinstance(payload, dict):
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    else:
        raw = payload
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

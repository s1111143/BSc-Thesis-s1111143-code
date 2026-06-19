"""Shared configuration: constants, model names, and artifact paths."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from dotenv import load_dotenv


SEED = 42
SAMPLE_QUERY_COUNT = 400
TOP_K = 5
TEMPERATURE = 0.0


NEGATIVE_TO_POSITIVE_RATIO = 3.0


FIXED_CHUNK_SIZE_TOKENS = 512
OVERLAP_RATIO = 0.15
STRUCTURE_SOFT_TARGET_TOKENS = 512
STRUCTURE_HARD_CAP_TOKENS = 1024
SEMANTIC_BREAKPOINT_PERCENTILE = 25
SEMANTIC_MIN_SENTENCES = 4


EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_GENERATION_MODEL = "gpt-4.1-mini"
MISTRAL_GENERATION_MODEL = "open-mistral-nemo-2407"
RAGAS_JUDGE_MODEL = "claude-haiku-4-5-20251001"

CHUNKING_STRATEGIES = ["fixed", "structure", "semantic"]
GENERATION_MODELS = [OPENAI_GENERATION_MODEL, MISTRAL_GENERATION_MODEL]


MAX_GENERATION_CALLS = 2500
MAX_RAGAS_EVAL_ROWS = 2500
DEFAULT_EMBED_BATCH_SIZE = 128


RAGAS_MAX_WORKERS = 16
RAGAS_BATCH_ROWS = 20
RAGAS_MAX_RETRIES = 20
RAGAS_MAX_WAIT = 60


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"
CORPUS_PATH = DATASET_DIR / "corpus.jsonl"
QUERIES_PATH = DATASET_DIR / "queries.jsonl"
QRELS_PATH = DATASET_DIR / "qrels" / "test.tsv"

CODE_DIR = REPO_ROOT / "code"
ARTIFACTS_DIR = CODE_DIR / "artifacts"
PREPARED_DIR = ARTIFACTS_DIR / "prepared"
CHUNKS_DIR = ARTIFACTS_DIR / "chunks"
RETRIEVAL_DIR = ARTIFACTS_DIR / "retrieval"
GENERATION_DIR = ARTIFACTS_DIR / "generation"
ANALYSIS_DIR = ARTIFACTS_DIR / "analysis"
EFFICIENCY_DIR = ARTIFACTS_DIR / "efficiency"
CACHE_DIR = ARTIFACTS_DIR / "cache"


def load_environment() -> None:
    """Load environment variables from .env in repo root."""
    load_dotenv(REPO_ROOT / ".env")


def seed_everything(seed: int = SEED) -> None:
    """Seed Python and NumPy RNG so sampling and toy embeddings are repeatable."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories() -> None:
    """Create artifact folders once so every script can write safely."""
    for folder in [
        ARTIFACTS_DIR,
        PREPARED_DIR,
        CHUNKS_DIR,
        RETRIEVAL_DIR,
        GENERATION_DIR,
        ANALYSIS_DIR,
        EFFICIENCY_DIR,
        CACHE_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

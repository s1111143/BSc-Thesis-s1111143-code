"""Run the pipeline stages in order: prep, chunk, retrieve, generate, analyse."""

from __future__ import annotations

import argparse
import json
import os

from common.config import (
    MISTRAL_GENERATION_MODEL,
    OPENAI_GENERATION_MODEL,
    ensure_directories,
    load_environment,
    seed_everything,
)


def api_status() -> dict[str, bool]:
    """Return availability flags for paid APIs."""
    return {
        "openai_key_available": bool(os.getenv("OPENAI_API_KEY")),
        "mistral_key_available": bool(os.getenv("MISTRAL_API_KEY")),
    }


def run_pipeline(
    run_data: bool,
    run_chunks: bool,
    run_retrieval_step: bool,
    run_generation_step: bool,
    run_analysis_step: bool,
    embedding_provider: str,
    semantic_provider: str,
    force: bool,
    skip_ragas: bool,
) -> dict:
    """Run selected stages and return a JSON-serializable summary."""
    ensure_directories()
    load_environment()
    seed_everything()

    summary: dict[str, dict] = {"api_status": api_status()}
    if run_data:
        from pipeline.data_prep import run_data_prep

        summary["data_prep"] = run_data_prep(force=force)
    if run_chunks:
        from pipeline.chunkers import run_chunking

        summary["chunkers"] = run_chunking(
            strategies=["fixed", "structure", "semantic"],
            semantic_provider=semantic_provider,
            force=force,
        )
    if run_retrieval_step:
        from pipeline.retrieval import run_retrieval

        summary["retrieval"] = run_retrieval(
            strategies=["fixed", "structure", "semantic"],
            provider=embedding_provider,
            force=force,
        )
    if run_generation_step:
        from pipeline.generation import run_generation

        summary["generation"] = run_generation(
            strategies=["fixed", "structure", "semantic"],
            models=[OPENAI_GENERATION_MODEL, MISTRAL_GENERATION_MODEL],
            run_ragas=not skip_ragas,
        )
    if run_analysis_step:
        from pipeline.analysis import run_analysis

        summary["analysis"] = run_analysis()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full thesis RAG experiment pipeline.")
    parser.add_argument("--skip-data", action="store_true", help="Skip data preparation stage.")
    parser.add_argument("--skip-chunks", action="store_true", help="Skip chunking stage.")
    parser.add_argument("--skip-retrieval", action="store_true", help="Skip retrieval stage.")
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation stage.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis stage.")
    parser.add_argument(
        "--embedding-provider",
        default="auto",
        choices=["auto", "openai", "local"],
        help="Embedding backend for retrieval indexes.",
    )
    parser.add_argument(
        "--semantic-provider",
        default="auto",
        choices=["auto", "openai", "local"],
        help="Embedding backend for semantic chunking breakpoints.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of each stage even if outputs exist.",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip RAGAS evaluation in generation stage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pipeline(
        run_data=not args.skip_data,
        run_chunks=not args.skip_chunks,
        run_retrieval_step=not args.skip_retrieval,
        run_generation_step=not args.skip_generation,
        run_analysis_step=not args.skip_analysis,
        embedding_provider=args.embedding_provider,
        semantic_provider=args.semantic_provider,
        force=args.force,
        skip_ragas=args.skip_ragas,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

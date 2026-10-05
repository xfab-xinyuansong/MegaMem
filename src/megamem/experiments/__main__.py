from __future__ import annotations

import argparse
import json

from .config import load_config
from .protocols import SUITES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="megamem-experiments",
        description="Plan, build, execute, and summarize explicitly configured MegaMem paper experiments.",
        epilog=("Documents require doc_id and content. Questions require question_id and question, with optional gold_answer, "
                "expected_doc_ids, answer_facts, and question_type for evaluation. Enterprise split manifests require "
                "dev_question_ids (100) and validation_question_ids (400). Question-local datasets require corpus_doc_ids. "
                "Gold intervention files require question_id and gold_chunks with chunk_id, doc_id, and content. "
                "Selective oracle requires explicit answerable booleans. Use LLM_API_BASE and LLM_API_KEY for credentials; "
                "EMBEDDING_API_BASE and EMBEDDING_API_KEY can override embedding service settings. "
                "Install the experiments extra before build or run. Plan never starts model calls."),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "build", "run"):
        child = commands.add_parser(command)
        child.add_argument("--config", required=True, help="YAML configuration; paths resolve relative to this file")
        child.add_argument("--suite", nargs="+", choices=[*SUITES, "all"], default=["headline"])
        if command == "run":
            child.add_argument("--resume", action="store_true", help="Reuse only records with identical configuration, inputs, index, and code")
    child = commands.add_parser("summarize")
    child.add_argument("directory")
    child = commands.add_parser("compare")
    child.add_argument("first")
    child.add_argument("second")
    args = parser.parse_args(argv)
    from . import runner

    try:
        if args.command == "summarize":
            result = runner.summarize(args.directory)
        elif args.command == "compare":
            result = runner.compare(args.first, args.second)
        else:
            config = load_config(args.config)
            suites = list(SUITES) if "all" in args.suite else args.suite
            if args.command == "run":
                result = runner.run(config, suites, resume=args.resume)
            else:
                result = getattr(runner, args.command)(config, suites)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

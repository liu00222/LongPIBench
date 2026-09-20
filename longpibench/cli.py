"""Public LongPIBench command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .attacks import ATTACKS, GOALS
from .benchmark import DEFAULT_GOALS, SUITES, prepare_example
from .data import DATA_ROOT, download_dataset
from .runner import run_detector, run_model


def _condition(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--attack", choices=ATTACKS, default="authority_spoof")
    parser.add_argument("--goal", choices=("default", *GOALS), default="default")
    parser.add_argument("--position", default="default")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LongPIBench public benchmark CLI")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser(
        "download-data", help="download and validate all four suites from Hugging Face"
    )
    download.add_argument("--local-dir", type=Path, default=DATA_ROOT)
    download.add_argument("--revision", default="main")
    download.add_argument("--force", action="store_true", help="download files again")

    inspect = commands.add_parser("inspect", help="print one fully prepared example")
    _condition(inspect)
    inspect.add_argument("--item-id", type=int, default=0)

    run = commands.add_parser("run", help="evaluate an LLM")
    _condition(run)
    run.add_argument("--backend", choices=("openai", "hf", "secalign"), required=True)
    run.add_argument("--model", help="API model name or Hugging Face model ID")
    run.add_argument("--max-output-tokens", type=int, default=4096)
    run.add_argument("--start", type=int, default=0)
    run.add_argument("--count", type=int, default=100)
    run.add_argument("--output", type=Path, required=True)

    detect = commands.add_parser("detect", help="evaluate a prompt-injection detector")
    _condition(detect)
    detect.add_argument("--detector", required=True, help="alias or Hugging Face model ID")
    detect.add_argument("--threshold", type=float, default=0.5)
    detect.add_argument("--chunk-tokens", type=int, default=512)
    detect.add_argument("--start", type=int, default=0)
    detect.add_argument("--count", type=int, default=100)
    detect.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "download-data":
        path = download_dataset(args.local_dir, revision=args.revision, force_download=args.force)
        print(f"Dataset ready at {path}")
        return 0
    if args.command == "inspect":
        example = prepare_example(args.suite, args.item_id, args.attack, args.goal, args.position)
        print(json.dumps(example.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        from .models import create_model

        goal = DEFAULT_GOALS[args.suite] if args.goal == "default" else args.goal
        model = create_model(args.backend, args.model, args.max_output_tokens)
        asr = run_model(
            model,
            suite=args.suite,
            attack=args.attack,
            goal=goal,
            position=args.position,
            start=args.start,
            count=args.count,
            output=args.output,
        )
        print(f"ASR={asr:.4f}")
        return 0

    from .detectors import create_detector

    goal = DEFAULT_GOALS[args.suite] if args.goal == "default" else args.goal
    detector = create_detector(args.detector, args.threshold, args.chunk_tokens)
    error_rate = run_detector(
        detector,
        suite=args.suite,
        attack=args.attack,
        goal=goal,
        position=args.position,
        start=args.start,
        count=args.count,
        output=args.output,
    )
    metric = "FPR" if args.attack == "no" else "FNR"
    print(f"{metric}={error_rate:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

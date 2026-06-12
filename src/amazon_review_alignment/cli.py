from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import load_config
from .utils import configure_logging


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/full.yaml", help="Path to YAML config.")
    parser.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review-align",
        description="Amazon review SFT, DPO, PPO, and GRPO alignment pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data")
    _add_common_arguments(prepare)

    pilot = subparsers.add_parser("teacher-pilot")
    _add_common_arguments(pilot)
    pilot.add_argument("--limit", type=int)

    batch = subparsers.add_parser("teacher-batch")
    _add_common_arguments(batch)
    batch.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create and validate the Batch API JSONL without submitting it.",
    )

    sft = subparsers.add_parser("train-sft")
    _add_common_arguments(sft)

    dpo = subparsers.add_parser("train-dpo")
    _add_common_arguments(dpo)

    rm_human = subparsers.add_parser("prepare-rm-human-eval")
    _add_common_arguments(rm_human)
    rm_human.add_argument("--samples", type=int)

    rlhf_data = subparsers.add_parser("build-rlhf-data")
    _add_common_arguments(rlhf_data)
    rlhf_data.add_argument("--responses", type=Path, required=True)

    merge = subparsers.add_parser("merge-sft")
    _add_common_arguments(merge)

    reward = subparsers.add_parser("train-reward")
    _add_common_arguments(reward)

    ppo = subparsers.add_parser("train-ppo")
    _add_common_arguments(ppo)

    grpo = subparsers.add_parser("train-grpo")
    _add_common_arguments(grpo)

    inference = subparsers.add_parser("inference")
    _add_common_arguments(inference)
    inference.add_argument(
        "--variant",
        choices=("base", "sft", "dpo", "ppo", "grpo"),
        required=True,
    )
    inference.add_argument("--force", action="store_true")

    evaluate = subparsers.add_parser("evaluate")
    _add_common_arguments(evaluate)
    evaluate.add_argument("--force-inference", action="store_true")
    evaluate.add_argument(
        "--llm-judge",
        action="store_true",
        help="Override the config and enable OpenAI pairwise judging.",
    )

    human = subparsers.add_parser("human-eval")
    _add_common_arguments(human)
    human.add_argument("--samples", type=int, default=200)
    human.add_argument("--left-variant", default="ppo")
    human.add_argument("--right-variant", default="grpo")
    human.add_argument("--responses", type=Path)

    report = subparsers.add_parser("build-report")
    _add_common_arguments(report)
    return parser


def _print_result(result: Any) -> None:
    if isinstance(result, Path):
        print(result)
    elif result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    config = load_config(args.config)

    command_handlers: dict[str, Callable[[], Any]] = {}
    if args.command == "prepare-data":
        from .data import prepare_data

        command_handlers[args.command] = lambda: prepare_data(config)
    elif args.command == "teacher-pilot":
        from .teacher import run_teacher_pilot

        command_handlers[args.command] = lambda: run_teacher_pilot(config, args.limit)
    elif args.command == "teacher-batch":
        from .teacher import run_teacher_batch

        command_handlers[args.command] = lambda: run_teacher_batch(config, args.prepare_only)
    elif args.command == "train-sft":
        from .train_sft import train_sft

        command_handlers[args.command] = lambda: train_sft(config)
    elif args.command == "train-dpo":
        from .train_dpo import train_dpo

        command_handlers[args.command] = lambda: train_dpo(config)
    elif args.command == "prepare-rm-human-eval":
        from .rlhf_data import prepare_rm_human_eval

        command_handlers[args.command] = lambda: prepare_rm_human_eval(
            config,
            args.samples,
        )
    elif args.command == "build-rlhf-data":
        from .rlhf_data import build_rlhf_data

        command_handlers[args.command] = lambda: build_rlhf_data(
            config,
            args.responses,
        )
    elif args.command == "merge-sft":
        from .merge_sft import merge_sft

        command_handlers[args.command] = lambda: merge_sft(config)
    elif args.command == "train-reward":
        from .train_reward import train_reward

        command_handlers[args.command] = lambda: train_reward(config)
    elif args.command == "train-ppo":
        from .train_ppo import train_ppo

        command_handlers[args.command] = lambda: train_ppo(config)
    elif args.command == "train-grpo":
        from .train_grpo import train_grpo

        command_handlers[args.command] = lambda: train_grpo(config)
    elif args.command == "inference":
        from .inference import run_inference

        command_handlers[args.command] = lambda: run_inference(config, args.variant, args.force)
    elif args.command == "evaluate":
        from .evaluation import run_evaluation

        if args.llm_judge:
            config["evaluation"]["run_llm_judge"] = True
        command_handlers[args.command] = lambda: run_evaluation(config, args.force_inference)
    elif args.command == "human-eval":
        from .human_eval import prepare_human_evaluation, summarize_human_evaluation

        if args.responses:
            command_handlers[args.command] = lambda: summarize_human_evaluation(
                config,
                args.responses,
            )
        else:
            command_handlers[args.command] = lambda: prepare_human_evaluation(
                config,
                args.samples,
                args.left_variant,
                args.right_variant,
            )
    elif args.command == "build-report":
        from .report import build_report

        command_handlers[args.command] = lambda: build_report(config)

    _print_result(command_handlers[args.command]())


if __name__ == "__main__":
    main()

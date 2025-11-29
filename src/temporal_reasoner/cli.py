"""Command-line interface for the temporal reasoning solver."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .parser import parse_observation_action_sequence
from .solver import build_model, debug_summary, run_viterbi, write_state_sequence


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the CS 561 HW3 temporal reasoning solver (Viterbi)."
    )
    parser.add_argument(
        "-s",
        "--state-weights",
        type=Path,
        default=Path("state_weights.txt"),
        help="State prior weights file (default: state_weights.txt).",
    )
    parser.add_argument(
        "-e",
        "--state-observation-weights",
        type=Path,
        default=Path("state_observation_weights.txt"),
        help="State-observation weight table (default: state_observation_weights.txt).",
    )
    parser.add_argument(
        "-t",
        "--state-action-state-weights",
        type=Path,
        default=Path("state_action_state_weights.txt"),
        help=(
            "State-action-state transition weights "
            "(default: state_action_state_weights.txt)."
        ),
    )
    parser.add_argument(
        "-i",
        "--observation-actions",
        type=Path,
        default=Path("observation_actions.txt"),
        help="Observation/action sequence file (default: observation_actions.txt).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("states.txt"),
        help="Destination file for the predicted state sequence (default: states.txt).",
    )
    parser.add_argument(
        "--infer-null-action",
        dest="infer_null_action",
        action="store_true",
        help="Infer missing actions as 'N' when observation rows omit them.",
    )
    parser.add_argument(
        "--no-infer-null-action",
        dest="infer_null_action",
        action="store_false",
        help="Require explicit actions (except possibly the final observation).",
    )
    parser.set_defaults(infer_null_action=True)
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Print debug information about parsed inputs and model tables.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    observations, actions_taken = parse_observation_action_sequence(
        args.observation_actions, infer_null_action=args.infer_null_action
    )
    model = build_model(
        args.state_weights,
        args.state_observation_weights,
        args.state_action_state_weights,
        observations,
        actions_taken,
    )
    if args.debug:
        print("[debug]", debug_summary(observations=observations, actions=actions_taken, model=model))
    sequence = run_viterbi(model, observations, actions_taken)
    write_state_sequence(args.output, sequence)


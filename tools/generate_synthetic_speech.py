#!/usr/bin/env python3
"""Generate synthetic speech-style POMDP datasets for HW3 debugging."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from temporal_reasoner.synthetic import generate_dataset


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic speech-style HW3 inputs.")
    parser.add_argument("--states", type=int, default=12, help="Number of hidden states.")
    parser.add_argument("--observations", type=int, default=8, help="Number of observation symbols.")
    parser.add_argument("--length", type=int, default=25, help="Length of observation sequence.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to place generated files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--observations-only",
        action="store_true",
        help="Emit only observations (no explicit actions) in observation_actions.txt.",
    )
    parser.add_argument(
        "--weight-low",
        type=int,
        default=1,
        help="Minimum integer weight to sample.",
    )
    parser.add_argument(
        "--weight-high",
        type=int,
        default=100,
        help="Maximum integer weight to sample.",
    )
    args = parser.parse_args(argv)

    generate_dataset(
        output_dir=args.output_dir,
        state_count=args.states,
        observation_count=args.observations,
        sequence_length=args.length,
        include_actions=not args.observations_only,
        seed=args.seed,
        weight_low=args.weight_low,
        weight_high=args.weight_high,
    )
    print(f"Wrote synthetic dataset to {args.output_dir}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Generate synthetic speech-style POMDP datasets for HW3 debugging."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _write_state_weights(path: Path, states: Sequence[str], weights: Sequence[int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("state_weights\n")
        handle.write(f"{len(states)} 0\n")
        for state, weight in zip(states, weights):
            handle.write(f"\"{state}\" {weight}\n")


def _write_state_observation_weights(
    path: Path,
    states: Sequence[str],
    observations: Sequence[str],
    weights: Dict[Tuple[str, str], int],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("state_observation_weights\n")
        handle.write(f"{len(states)*len(observations)} {len(states)} {len(observations)} 1\n")
        for state in states:
            for obs in observations:
                handle.write(f"\"{state}\" \"{obs}\" {weights[(state, obs)]}\n")


def _write_state_action_state_weights(
    path: Path,
    states: Sequence[str],
    action: str,
    weights: Dict[Tuple[str, str], int],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("state_action_state_weights\n")
        handle.write(f"{len(states)*len(states)} {len(states)} 1 1\n")
        for state in states:
            for next_state in states:
                handle.write(f"\"{state}\" \"{action}\" \"{next_state}\" {weights[(state, next_state)]}\n")


def _write_observation_actions(
    path: Path,
    observations: Sequence[str],
    action: str,
    include_actions: bool,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("observation_actions\n")
        handle.write(f"{len(observations)}\n")
        for obs in observations:
            if include_actions:
                handle.write(f"\"{obs}\" \"{action}\"\n")
            else:
                handle.write(f"\"{obs}\"\n")


def _write_expected_states(path: Path, states: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("states\n")
        handle.write(f"{len(states)}\n")
        for state in states:
            handle.write(f"\"{state}\"\n")


def _sample_from_weights(labels: Sequence[str], weights: Sequence[int]) -> str:
    total = sum(weights)
    threshold = random.uniform(0, total)
    cumulative = 0.0
    for label, weight in zip(labels, weights):
        cumulative += weight
        if threshold <= cumulative:
            return label
    return labels[-1]


def build_random_weights(rows: Sequence[str], cols: Sequence[str], low: int, high: int) -> Dict[Tuple[str, str], int]:
    return {
        (row, col): random.randint(low, high)
        for row in rows
        for col in cols
    }


def generate_sequence(
    *,
    states: Sequence[str],
    observations: Sequence[str],
    action: str,
    length: int,
    prior_weights: Sequence[int],
    transition_weights: Dict[Tuple[str, str], int],
    emission_weights: Dict[Tuple[str, str], int],
) -> Tuple[List[str], List[str]]:
    hidden: List[str] = []
    emitted: List[str] = []

    current_state = _sample_from_weights(states, prior_weights)
    hidden.append(current_state)
    emitted.append(_sample_from_weights(observations, [emission_weights[(current_state, obs)] for obs in observations]))

    for _ in range(1, length):
        next_state = _sample_from_weights(states, [transition_weights[(current_state, s2)] for s2 in states])
        hidden.append(next_state)
        emitted.append(_sample_from_weights(observations, [emission_weights[(next_state, obs)] for obs in observations]))
        current_state = next_state

    return hidden, emitted


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

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    states = [f"S{i}" for i in range(args.states)]
    observations = [f"O{i}" for i in range(args.observations)]
    action = "N"

    prior_weights = [random.randint(args.weight_low, args.weight_high) for _ in states]
    transition_weights = build_random_weights(states, states, args.weight_low, args.weight_high)
    emission_weights = build_random_weights(states, observations, args.weight_low, args.weight_high)

    hidden_sequence, emitted_observations = generate_sequence(
        states=states,
        observations=observations,
        action=action,
        length=args.length,
        prior_weights=prior_weights,
        transition_weights=transition_weights,
        emission_weights=emission_weights,
    )

    _write_state_weights(args.output_dir / "state_weights.txt", states, prior_weights)
    _write_state_observation_weights(
        args.output_dir / "state_observation_weights.txt",
        states,
        observations,
        emission_weights,
    )
    _write_state_action_state_weights(
        args.output_dir / "state_action_state_weights.txt",
        states,
        action,
        transition_weights,
    )
    _write_observation_actions(
        args.output_dir / "observation_actions.txt",
        emitted_observations,
        action,
        include_actions=not args.observations_only,
    )
    _write_expected_states(args.output_dir / "expected_states.txt", hidden_sequence)

    print(f"Wrote synthetic dataset to {args.output_dir}")


if __name__ == "__main__":
    main()


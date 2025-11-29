"""Synthetic dataset generation for stress testing."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .constants import DEFAULT_NULL_ACTION


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
        handle.write(
            f"{len(states)*len(observations)} {len(states)} {len(observations)} 1\n"
        )
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
                handle.write(
                    f"\"{state}\" \"{action}\" \"{next_state}\" {weights[(state, next_state)]}\n"
                )


def _write_observation_actions(
    path: Path,
    observations: Sequence[str],
    action: str,
    include_actions: bool,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("observation_actions\n")
        handle.write(f"{len(observations)}\n")
        for idx, obs in enumerate(observations):
            if include_actions or idx < len(observations) - 1:
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


def build_random_weights(
    rows: Sequence[str], cols: Sequence[str], low: int, high: int
) -> Dict[Tuple[str, str], int]:
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
    emitted.append(
        _sample_from_weights(
            observations, [emission_weights[(current_state, obs)] for obs in observations]
        )
    )

    for _ in range(1, length):
        next_state = _sample_from_weights(
            states, [transition_weights[(current_state, s2)] for s2 in states]
        )
        hidden.append(next_state)
        emitted.append(
            _sample_from_weights(
                observations, [emission_weights[(next_state, obs)] for obs in observations]
            )
        )
        current_state = next_state

    return hidden, emitted


def generate_dataset(
    *,
    output_dir: Path,
    state_count: int,
    observation_count: int,
    sequence_length: int,
    include_actions: bool,
    seed: int,
    weight_low: int,
    weight_high: int,
) -> None:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    states = [f"S{i}" for i in range(state_count)]
    observations = [f"O{i}" for i in range(observation_count)]
    action = DEFAULT_NULL_ACTION

    prior_weights = [random.randint(weight_low, weight_high) for _ in states]
    transition_weights = build_random_weights(states, states, weight_low, weight_high)
    emission_weights = build_random_weights(states, observations, weight_low, weight_high)

    hidden_sequence, emitted_observations = generate_sequence(
        states=states,
        observations=observations,
        action=action,
        length=sequence_length,
        prior_weights=prior_weights,
        transition_weights=transition_weights,
        emission_weights=emission_weights,
    )

    _write_state_weights(output_dir / "state_weights.txt", states, prior_weights)
    _write_state_observation_weights(
        output_dir / "state_observation_weights.txt",
        states,
        observations,
        emission_weights,
    )
    _write_state_action_state_weights(
        output_dir / "state_action_state_weights.txt",
        states,
        action,
        transition_weights,
    )
    _write_observation_actions(
        output_dir / "observation_actions.txt",
        emitted_observations,
        action,
        include_actions=include_actions,
    )
    _write_expected_states(output_dir / "expected_states.txt", hidden_sequence)


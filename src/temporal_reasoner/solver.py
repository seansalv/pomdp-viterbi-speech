"""Core Viterbi solver and probability table construction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from .constants import DEBUG_SAMPLE_SIZE, LOG_ZERO
from .parser import (
    parse_observation_action_sequence,
    parse_state_action_state_weights,
    parse_state_observation_weights,
    parse_state_weights,
)


@dataclass
class POMDPModel:
    """Container for model probabilities."""

    states: List[str]
    actions: List[str]
    observations: List[str]
    prior: Dict[str, float]
    transition: Dict[str, Dict[str, Dict[str, float]]]
    emission: Dict[str, Dict[str, float]]


def normalize_prior(state_weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(state_weights.values())
    if total <= 0:
        raise ValueError("State weights must sum to a positive value.")
    return {state: weight / total for state, weight in state_weights.items()}


def _normalize_rows(
    rows: Dict[str, Dict[str, float]], identifier: str
) -> Dict[str, Dict[str, float]]:
    normalized: Dict[str, Dict[str, float]] = {}
    for key, columns in rows.items():
        total = sum(columns.values())
        if total <= 0:
            raise ValueError(
                f"No positive weights found while normalizing {identifier} '{key}'."
            )
        normalized[key] = {col: weight / total for col, weight in columns.items()}
    return normalized


def build_emission_probabilities(
    states: Sequence[str],
    observations: Sequence[str],
    raw_weights: Dict[tuple[str, str], float],
    default_weight: int,
) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    for state in states:
        row: Dict[str, float] = {}
        for observation in observations:
            weight = raw_weights.get((state, observation), default_weight)
            row[observation] = float(weight)
        rows[state] = row
    return _normalize_rows(rows, "emission row for state")


def build_transition_probabilities(
    states: Sequence[str],
    actions: Sequence[str],
    raw_weights: Dict[tuple[str, str, str], float],
    default_weight: int,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    action_rows: Dict[str, Dict[str, Dict[str, float]]] = {}
    for action in actions:
        state_rows: Dict[str, Dict[str, float]] = {}
        for state in states:
            row: Dict[str, float] = {}
            for next_state in states:
                weight = raw_weights.get((state, action, next_state), default_weight)
                row[next_state] = float(weight)
            state_rows[state] = row
        action_rows[action] = _normalize_rows(
            state_rows, f"transition row for action '{action}'"
        )
    return action_rows


def safe_log(value: float) -> float:
    return math.log(value) if value > 0 else LOG_ZERO


def run_viterbi(
    model: POMDPModel, observations: Sequence[str], actions_taken: Sequence[str]
) -> List[str]:
    if not observations:
        raise ValueError("Observation sequence must contain at least one element.")
    expected_actions = max(len(observations) - 1, 0)
    if len(actions_taken) != expected_actions:
        raise ValueError(
            "Action sequence must contain one fewer entry than observations."
        )

    delta: Dict[str, float] = {}
    backpointers: List[Dict[str, str | None]] = []
    first_observation = observations[0]
    if first_observation not in model.observations:
        raise ValueError(f"Unknown observation '{first_observation}'.")

    for state in model.states:
        log_prob = safe_log(model.prior[state]) + safe_log(
            model.emission[state][first_observation]
        )
        delta[state] = log_prob
    backpointers.append({state: None for state in model.states})

    for t in range(1, len(observations)):
        obs = observations[t]
        action = actions_taken[t - 1]
        if action not in model.actions:
            raise ValueError(f"Action '{action}' lacks weight data.")
        if obs not in model.observations:
            raise ValueError(f"Observation '{obs}' lacks weight data.")

        prev_delta = delta
        delta = {}
        backpointer_step: Dict[str, str] = {}

        for curr_state in model.states:
            best_prev_state = None
            best_log_prob = LOG_ZERO
            emission_log = safe_log(model.emission[curr_state][obs])
            if emission_log == LOG_ZERO:
                delta[curr_state] = LOG_ZERO
                backpointer_step[curr_state] = model.states[0]
                continue
            for prev_state in model.states:
                transition_log = safe_log(
                    model.transition[action][prev_state][curr_state]
                )
                if transition_log == LOG_ZERO:
                    continue
                candidate = prev_delta[prev_state] + transition_log + emission_log
                if candidate > best_log_prob:
                    best_log_prob = candidate
                    best_prev_state = prev_state
            delta[curr_state] = best_log_prob
            backpointer_step[curr_state] = best_prev_state or model.states[0]
        backpointers.append(backpointer_step)

    final_state = max(delta, key=delta.get)
    sequence = [final_state]
    for step in reversed(backpointers[1:]):
        final_state = step[final_state]
        sequence.append(final_state)
    sequence.reverse()
    return sequence


def write_state_sequence(path: Path, states: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("states\n")
        handle.write(f"{len(states)}\n")
        for state in states:
            handle.write(f"\"{state}\"\n")


def build_model(
    state_weights_path: Path,
    state_observation_path: Path,
    state_transition_path: Path,
    observations: Sequence[str],
    actions: Sequence[str],
) -> POMDPModel:
    states, state_weights = parse_state_weights(state_weights_path)
    (
        raw_obs_weights,
        default_obs_weight,
        observation_names,
    ) = parse_state_observation_weights(state_observation_path)
    (
        raw_transition_weights,
        default_transition_weight,
        action_names,
    ) = parse_state_action_state_weights(state_transition_path)

    obs_union = list(observation_names)
    for obs in observations:
        if obs not in obs_union:
            obs_union.append(obs)

    action_union = list(action_names)
    for action in actions:
        if action not in action_union:
            action_union.append(action)

    prior = normalize_prior(state_weights)
    emission = build_emission_probabilities(
        states, obs_union, raw_obs_weights, default_obs_weight
    )
    transition = build_transition_probabilities(
        states, action_union, raw_transition_weights, default_transition_weight
    )

    return POMDPModel(
        states=states,
        actions=action_union,
        observations=obs_union,
        prior=prior,
        emission=emission,
        transition=transition,
    )


def _format_sample(values: Sequence[str], size: int = DEBUG_SAMPLE_SIZE) -> str:
    if len(values) <= size:
        return ", ".join(values)
    head = ", ".join(values[:size])
    return f"{head}, ..."


def debug_summary(
    *,
    observations: Sequence[str],
    actions: Sequence[str],
    model: POMDPModel,
) -> str:
    sample_action = model.actions[0]
    sample_state = model.states[0]
    lines = [
        f"states: {len(model.states)}",
        f"actions: {len(model.actions)} [{_format_sample(model.actions)}]",
        f"observations: {len(model.observations)} "
        f"[{_format_sample(model.observations)}]",
        "sequence obs/actions: "
        f"{len(observations)} / {len(actions)} "
        f"obs sample [{_format_sample(observations)}] "
        f"actions sample [{_format_sample(actions)}]",
        f"transition row for action '{sample_action}', state '{sample_state}': "
        f"{model.transition[sample_action][sample_state]}",
        f"emission row for state '{sample_state}': "
        f"{model.emission[sample_state]}",
    ]
    return "\n".join(lines)


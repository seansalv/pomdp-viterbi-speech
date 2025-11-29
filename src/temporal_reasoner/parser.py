"""Parsing helpers for HW3 probability tables."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .constants import DEFAULT_NULL_ACTION
from .exceptions import FormatError


def _read_nonempty_lines(path: Path) -> List[str]:
    data = path.read_text(encoding="utf-8").splitlines()
    lines = [line.strip() for line in data if line.strip()]
    if not lines:
        raise FormatError(f"{path} is empty or contains only blank lines.")
    return lines


def _require_indicator(lines: List[str], expected: str, path: Path) -> List[str]:
    indicator = lines[0].strip().lower()
    if indicator != expected:
        raise FormatError(
            f"Expected indicator '{expected}' at the top of {path}, "
            f"but found '{lines[0]}'."
        )
    return lines[1:]


def _parse_int_fields(raw: str, expected_count: int, path: Path) -> List[int]:
    parts = raw.split()
    if len(parts) != expected_count:
        raise FormatError(
            f"Expected {expected_count} integers on line '{raw}' in {path}."
        )
    try:
        return [int(value) for value in parts]
    except ValueError as exc:
        raise FormatError(f"Non-integer count encountered in {path}: {exc}") from exc


def _parse_weight_line(
    raw: str, expected_fields: int, path: Path
) -> Tuple[List[str], int]:
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        raise FormatError(f"Failed to parse quoted tokens in {path}: {exc}") from exc
    if len(tokens) != expected_fields:
        raise FormatError(
            f"Expected {expected_fields} tokens before the weight in {path}, "
            f"but found {len(tokens)} on line '{raw}'."
        )
    try:
        weight = int(tokens[-1])
    except ValueError as exc:
        raise FormatError(f"Weight must be an integer in {path}: {exc}") from exc
    return tokens[:-1], weight


def parse_state_weights(path: Path) -> Tuple[List[str], Dict[str, float]]:
    lines = _read_nonempty_lines(path)
    lines = _require_indicator(lines, "state_weights", path)
    counts_line, *entry_lines = lines
    parts = counts_line.split()
    if len(parts) == 1:
        try:
            num_states = int(parts[0])
        except ValueError as exc:
            raise FormatError(f"Invalid state count in {path}: {exc}") from exc
    elif len(parts) == 2:
        try:
            num_states = int(parts[0])
            int(parts[1])
        except ValueError as exc:
            raise FormatError(f"Invalid header fields in {path}: {exc}") from exc
    else:
        raise FormatError(
            f"Expected one or two integers on line '{counts_line}' in {path}."
        )
    states: List[str] = []
    weights: Dict[str, float] = {}
    for raw in entry_lines:
        tokens, weight = _parse_weight_line(raw, 2, path)
        state = tokens[0]
        if state not in states:
            states.append(state)
        weights[state] = float(weight)
    if len(states) != num_states:
        raise FormatError(
            f"{path} declares {num_states} states but lists {len(states)} entries."
        )
    return states, weights


def parse_state_observation_weights(
    path: Path,
) -> Tuple[Dict[Tuple[str, str], float], int, List[str]]:
    lines = _read_nonempty_lines(path)
    lines = _require_indicator(lines, "state_observation_weights", path)
    counts_line, *entry_lines = lines
    _num_pairs, _num_states, _num_observations, default_weight = _parse_int_fields(
        counts_line, 4, path
    )
    weights: Dict[Tuple[str, str], float] = {}
    observations: List[str] = []
    for raw in entry_lines:
        tokens, weight = _parse_weight_line(raw, 3, path)
        state, observation = tokens[0], tokens[1]
        weights[(state, observation)] = float(weight)
        if observation not in observations:
            observations.append(observation)
    return weights, default_weight, observations


def parse_state_action_state_weights(
    path: Path,
) -> Tuple[Dict[Tuple[str, str, str], float], int, List[str]]:
    lines = _read_nonempty_lines(path)
    lines = _require_indicator(lines, "state_action_state_weights", path)
    counts_line, *entry_lines = lines
    _num_triples, _num_states, _num_actions, default_weight = _parse_int_fields(
        counts_line, 4, path
    )
    weights: Dict[Tuple[str, str, str], float] = {}
    actions: List[str] = []
    for raw in entry_lines:
        tokens, weight = _parse_weight_line(raw, 4, path)
        state, action, next_state = tokens[0], tokens[1], tokens[2]
        weights[(state, action, next_state)] = float(weight)
        if action not in actions:
            actions.append(action)
    return weights, default_weight, actions


def parse_observation_action_sequence(
    path: Path,
    *,
    infer_null_action: bool,
) -> Tuple[List[str], List[str]]:
    lines = _read_nonempty_lines(path)
    lines = _require_indicator(lines, "observation_actions", path)
    counts_line, *entry_lines = lines
    num_pairs = _parse_int_fields(counts_line, 1, path)[0]
    observations: List[str] = []
    actions: List[str] = []
    for idx, raw in enumerate(entry_lines):
        tokens = shlex.split(raw)
        if len(tokens) == 1:
            observation = tokens[0]
            action: str | None
            if idx == len(entry_lines) - 1:
                action = None
            elif infer_null_action:
                action = DEFAULT_NULL_ACTION
            else:
                raise FormatError(
                    f"Missing action for observation '{observation}' in {path}. "
                    "Provide explicit actions or pass --infer-null-action."
                )
        elif len(tokens) == 2:
            observation, action = tokens
        else:
            raise FormatError(
                f"Each entry in {path} must contain one observation "
                "optionally followed by an action."
            )
        observations.append(observation)
        if action is not None:
            actions.append(action)
    if len(observations) != num_pairs:
        raise FormatError(
            f"{path} declares {num_pairs} pairs but lists {len(observations)}."
        )
    expected_actions = max(len(observations) - 1, 0)
    if len(actions) < expected_actions:
        raise FormatError(
            f"{path} needs at least {expected_actions} actions but only "
            f"{len(actions)} were provided."
        )
    if len(actions) > expected_actions:
        actions = actions[:expected_actions]
    return observations, actions


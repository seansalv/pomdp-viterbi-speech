from __future__ import annotations

import argparse
import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


LOG_ZERO = float("-inf")
DEFAULT_NULL_ACTION = "N"
DEBUG_SAMPLE_SIZE = 5


class FormatError(RuntimeError):
    """Raised when an input file does not match the required specification."""


@dataclass
class POMDPModel:
    states: List[str]
    actions: List[str]
    observations: List[str]
    prior: Dict[str, float]
    transition: Dict[str, Dict[str, Dict[str, float]]]
    emission: Dict[str, Dict[str, float]]


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
        except ValueError as exc:
            raise FormatError(f"Invalid state count in {path}: {exc}") from exc
        # The default weight is not used for normalization, but ensure it's numeric.
        try:
            int(parts[1])
        except ValueError as exc:
            raise FormatError(f"Invalid default weight in {path}: {exc}") from exc
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
) -> None:
    print("[debug] states:", len(model.states))
    print("[debug] actions:", len(model.actions), f"[{_format_sample(model.actions)}]")
    print(
        "[debug] observations:",
        len(model.observations),
        f"[{_format_sample(model.observations)}]",
    )
    print(
        "[debug] sequence obs/actions:",
        len(observations),
        "/",
        len(actions),
        f"obs sample [{_format_sample(observations)}]",
        f"actions sample [{_format_sample(actions)}]",
    )
    sample_action = model.actions[0]
    sample_state = model.states[0]
    print(
        f"[debug] transition row for action '{sample_action}', state '{sample_state}':",
        model.transition[sample_action][sample_state],
    )
    print(
        f"[debug] emission row for state '{sample_state}':",
        model.emission[sample_state],
    )


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
            raise ValueError(f"No positive weights found while normalizing {identifier} '{key}'.")
        normalized[key] = {col: weight / total for col, weight in columns.items()}
    return normalized


def build_emission_probabilities(
    states: Sequence[str],
    observations: Sequence[str],
    raw_weights: Dict[Tuple[str, str], float],
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
    raw_weights: Dict[Tuple[str, str, str], float],
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
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
        debug_summary(
            observations=observations,
            actions=actions_taken,
            model=model,
        )
    sequence = run_viterbi(model, observations, actions_taken)
    write_state_sequence(args.output, sequence)


if __name__ == "__main__":
    main()


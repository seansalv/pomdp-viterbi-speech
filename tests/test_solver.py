from __future__ import annotations

from pathlib import Path

from temporal_reasoner.parser import parse_observation_action_sequence
from temporal_reasoner.solver import build_model, run_viterbi
from temporal_reasoner.synthetic import generate_dataset


def _read_states(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip().strip('"') for line in lines[2:] if line.strip()]


def test_little_prince_sequence_matches_expected() -> None:
    base = Path("samples/little_prince")
    obs, actions = parse_observation_action_sequence(
        base / "observation_actions.txt", infer_null_action=False
    )
    model = build_model(
        base / "state_weights.txt",
        base / "state_observation_weights.txt",
        base / "state_action_state_weights.txt",
        obs,
        actions,
    )
    predicted = run_viterbi(model, obs, actions)
    expected = _read_states(base / "expected_states.txt")
    assert predicted == expected


def test_synthetic_dataset_observations_only(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_obs_only"
    generate_dataset(
        output_dir=output_dir,
        state_count=8,
        observation_count=5,
        sequence_length=15,
        include_actions=False,
        seed=123,
        weight_low=1,
        weight_high=5,
    )
    obs, actions = parse_observation_action_sequence(
        output_dir / "observation_actions.txt", infer_null_action=True
    )
    model = build_model(
        output_dir / "state_weights.txt",
        output_dir / "state_observation_weights.txt",
        output_dir / "state_action_state_weights.txt",
        obs,
        actions,
    )
    predicted = run_viterbi(model, obs, actions)
    assert len(predicted) == len(obs)


def test_synthetic_dataset_explicit_actions(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_explicit"
    generate_dataset(
        output_dir=output_dir,
        state_count=8,
        observation_count=5,
        sequence_length=12,
        include_actions=True,
        seed=321,
        weight_low=1,
        weight_high=7,
    )
    obs, actions = parse_observation_action_sequence(
        output_dir / "observation_actions.txt", infer_null_action=False
    )
    model = build_model(
        output_dir / "state_weights.txt",
        output_dir / "state_observation_weights.txt",
        output_dir / "state_action_state_weights.txt",
        obs,
        actions,
    )
    predicted = run_viterbi(model, obs, actions)
    assert len(predicted) == len(obs)


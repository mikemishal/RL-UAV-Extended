import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from visualizer.backend.replay_contract import LRPPOGuidance, PolicyVisible, ReplayBundle


def test_schema_is_valid_and_matches_typed_contract():
    schema = json.loads((Path(__file__).parents[1] / "docs" / "replay_v1.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    assert schema == ReplayBundle.model_json_schema()
    assert schema["properties"]["episode_count"]["default"] == 1
    assert schema["properties"]["episode_count"]["const"] == 1


def test_lrppo_angular_contract_preserves_native_bound():
    lead = (1, 0, 0)
    diagnostic = LRPPOGuidance(lead_direction=lead, native_adjustment_deg=30, signed_horizontal_adjustment_deg=-120)
    assert diagnostic.max_adjustment_deg == 30
    assert diagnostic.residual_action is None
    with pytest.raises(ValidationError):
        LRPPOGuidance(lead_direction=lead, max_adjustment_deg=45)
    with pytest.raises(ValidationError):
        LRPPOGuidance(lead_direction=lead, native_adjustment_deg=31)
    with pytest.raises(ValidationError):
        LRPPOGuidance(lead_direction=(float("nan"), 0, 0))


def test_privileged_data_cannot_enter_policy_visible_contract():
    values = dict(
        estimator="measurement", observation=[0.0] * 16, soldier_pos_m=(0, 0, 0),
        defender_pos_m=(0, 0, 0), defender_vel_mps=(0, 0, 0), enemy_detected=False,
    )
    assert PolicyVisible(**values).e_hat_m is None
    with pytest.raises(ValidationError):
        PolicyVisible(**values, enemy_pos=(10, 10, 10))
    with pytest.raises(ValidationError, match="Kalman-only"):
        PolicyVisible(**values, e_hat_m=(1, 2, 3))


def test_bundle_rejects_missing_recordings():
    with pytest.raises(ValidationError, match="episode_count"):
        ReplayBundle(
            provenance=dict(
                simulation_source_commit="a" * 40, visualizer_source_commit="b" * 40,
                source_dirty=False, visualizer_dirty=False, python_version="test", package_versions={},
                model_device="cpu", registry_sha256="c" * 64,
            ),
            seed_start=100, episodes=[],
        )

import gzip
import hashlib
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

np = pytest.importorskip("numpy")
from visualizer.backend import replay_engine as engine
from visualizer.backend.application_error import ApplicationError
from visualizer.backend.replay_contract import (
    EntityState,
    EnvironmentSpec,
    EpisodeSummary,
    PolicySelection,
    PolicyTelemetry,
    PolicyVisible,
    PrivilegedDiagnostics,
    ReplayBundle,
    ReplayEpisode,
    ReplayFrame,
    ReplayProvenance,
)


def _visible():
    return PolicyVisible(
        estimator="measurement",
        observation=[0.0] * 16,
        soldier_pos_m=(20.0, 10.0, 0.0),
        defender_pos_m=(0.0, 0.0, 10.0),
        defender_vel_mps=(1.0, 0.0, 0.0),
        enemy_detected=False,
    )


def _entity(x, y, z):
    return EntityState(position_m=(x, y, z), velocity_mps=(0.0, 0.0, 0.0), speed_mps=0.0)


def _bundle():
    visible = _visible()
    telemetry = PolicyTelemetry(
        decision_step=0,
        decision_policy_visible=visible,
        requested_action=(1.0, 0.0, 0.0),
        requested_action_norm=1.0,
        defender_policy_input=[0.0] * 16,
        controller_action_executed=True,
        defender_standby=False,
        avoidance_active=False,
        executed_velocity_mps=(1.0, 0.0, 0.0),
    )

    def frame(step, outcome, *, terminal=False):
        return ReplayFrame(
            step=step,
            time_s=step * 0.5,
            defender=_entity(float(step), 0.0, 10.0),
            attacker=_entity(10.0, float(step), 10.0),
            asset=_entity(20.0, 10.0, 0.0),
            moving_obstacles=[],
            policy_visible=visible,
            telemetry=telemetry if step else None,
            privileged_diagnostics=PrivilegedDiagnostics(
                attacker_position_m=(10.0, float(step), 10.0),
                attacker_velocity_mps=(0.0, 0.0, 0.0),
                defender_attacker_range_m=10.0,
                attacker_asset_range_m=10.0,
                simulator_info={},
            ),
            detection_radius_m=100.0,
            defender_action_applied=True if step else None,
            cumulative_reward=0.0,
            reward=0.0 if step else None,
            terminated=terminal,
            truncated=False,
            outcome=outcome,
        )

    defender = PolicySelection(
        policy_kind="analytical",
        policy_id="greedy",
        estimator="measurement",
        parameters={},
    )
    attacker = PolicySelection(
        policy_kind="scripted",
        policy_id="A0",
        estimator="scripted_ground_truth",
        parameters={},
    )
    config = {}
    import hashlib
    import json

    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ReplayBundle(
        provenance=ReplayProvenance(
            simulation_source_commit="a" * 40,
            visualizer_source_commit="b" * 40,
            source_dirty=False,
            visualizer_dirty=False,
            python_version="3.test",
            package_versions={},
            model_device="cpu",
            registry_sha256="c" * 64,
        ),
        episode_count=1,
        seed_start=2,
        episodes=[
            ReplayEpisode(
                episode_id="unit-test",
                simulation_seed=2,
                defender=defender,
                attacker=attacker,
                environment=EnvironmentSpec(
                    clutter="C0",
                    awareness="local",
                    builder="experiments.journal_clutter_protocol.build_canonical_env_config",
                    config=config,
                    config_sha256=config_hash,
                    dt_s=0.5,
                ),
                terminal_flags={"terminated": True, "truncated": False},
                static_obstacles=[],
                frames=[frame(0, "ongoing"), frame(1, "timeout", terminal=True)],
                events=[],
                summary=EpisodeSummary(
                    outcome="timeout",
                    steps=1,
                    duration_s=0.5,
                    final_defender_attacker_distance_m=10.0,
                    final_attacker_asset_distance_m=10.0,
                    total_reward=0.0,
                ),
            )
        ],
    )


@pytest.mark.parametrize(
    "values, message",
    [
        ({"defender_policy_id": "lr_ppo", "attacker_condition": "A0", "clutter_level": "C2", "evaluation_seed": 1},
         "explicit generation and training seed"),
        ({"defender_policy_id": "lr_ppo", "defender_generation": "d0_plus", "defender_training_seed": 77,
          "checkpoint_role": "secondary_best", "attacker_condition": "A0", "clutter_level": "C2", "evaluation_seed": 1},
         "secondary best checkpoints"),
        ({"defender_policy_id": "ppo_kalman", "defender_generation": "d1", "defender_training_seed": 77,
          "attacker_condition": "A0", "clutter_level": "C2", "evaluation_seed": 1},
         "supports seeds"),
        ({"defender_policy_id": "greedy", "defender_generation": "d0", "defender_training_seed": 71,
          "attacker_condition": "A0", "clutter_level": "C0", "evaluation_seed": 1},
         "do not have an architecture"),
        ({"defender_policy_id": "greedy", "attacker_condition": "A1-203", "attacker_model_seed": 202,
          "clutter_level": "C0", "evaluation_seed": 1},
         "requires attacker_model_seed=203"),
    ],
)
def test_request_rejects_incompatible_policy_selection(values, message):
    with pytest.raises(ValidationError, match=message):
        engine.ReplayRequest(**values)


def test_request_accepts_allowlisted_clutter_and_attacker_models():
    request = engine.ReplayRequest(
        defender_policy_id="lr_ppo",
        defender_generation="d1",
        defender_training_seed=77,
        attacker_condition="A1-203",
        attacker_model_seed=203,
        clutter_level="C3",
        evaluation_seed=0,
    )
    assert request.deterministic_inference is True
    assert request.checkpoint_role is None


def test_nonfinite_json_values_are_explicitly_null_with_reason():
    reasons = {}
    result = engine._plain({"valid": np.float32(2.5), "bad": np.float64(np.nan)}, reasons)
    assert result == {"valid": 2.5, "bad": None}
    assert "non-finite" in reasons["$.bad"]
    assert engine._vector((1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)
    assert engine._vector((1.0, float("nan"), 3.0)) is None


def test_checkpoint_integrity_is_required_before_loading(tmp_path):
    artifact = tmp_path / "checkpoint.zip"
    artifact.write_bytes(b"frozen checkpoint test bytes")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    class Store:
        def resolve_file(self, relative_path):
            assert relative_path == "checkpoint.zip"
            return artifact

    checkpoint = engine._ResolvedCheckpoint(
        artifact_id="unit-test",
        relative_path="checkpoint.zip",
        sha256=digest,
        role="final",
        training_seed=77,
    )
    assert engine._verify_artifact(Store(), checkpoint) == artifact
    tampered = artifact.read_bytes() + b"!"
    artifact.write_bytes(tampered)
    with pytest.raises(ApplicationError, match="SHA-256 mismatch"):
        engine._verify_artifact(Store(), checkpoint)


def test_model_observation_and_action_shapes_must_match_catalog():
    model = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(39,)),
        action_space=SimpleNamespace(shape=(3,)),
    )
    engine._checkpoint_model_input(model, (39,), "lrppo-test")
    with pytest.raises(ApplicationError, match="does not match required"):
        engine._checkpoint_model_input(model, (36,), "lrppo-test")
    model.action_space.shape = (2,)
    with pytest.raises(ApplicationError, match="does not match"):
        engine._checkpoint_model_input(model, (39,), "lrppo-test")


def test_trace_order_and_native_3d_validation():
    bundle = _bundle()
    engine._validate_trace(bundle)
    assert len(bundle.episodes[0].frames) == bundle.episodes[0].summary.steps + 1
    assert bundle.episodes[0].frames[0].defender.position_m == (0.0, 0.0, 10.0)
    assert bundle.episodes[0].frames[-1].terminated
    bad = bundle.model_copy(deep=True)
    bad.episodes[0].frames[1].step = 2
    with pytest.raises(engine.ApplicationError, match="indices are not contiguous"):
        engine._validate_trace(bad)


def test_gzip_round_trip_is_validated_and_never_overwrites(tmp_path, monkeypatch):
    bundle = _bundle()
    monkeypatch.setattr(engine, "local_output_root", lambda: tmp_path)
    output = engine.save_replay_bundle(bundle, "roundtrip")
    assert output.parent == tmp_path
    with gzip.open(output, "rt", encoding="utf-8") as source:
        assert '"schema_version":"1.0.0"' in source.read()
    loaded = engine.load_replay_bundle(output)
    assert loaded == bundle
    with pytest.raises(engine.ApplicationError, match="Refusing to overwrite"):
        engine.save_replay_bundle(bundle, "roundtrip")

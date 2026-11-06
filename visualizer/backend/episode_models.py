"""Simplified, user-facing scenario/episode data model (two-option backend, journal_cbf_attacker-era A1-CBF
freeze). SEPARATE and ADDITIVE to the existing `/api/cases`, `/api/replays`, `/api/replay-jobs`, and
`/api/cbf-review/*` routes -- those remain unchanged for historical/research tooling. See
`visualizer/docs/simplified_episode_backend.md` for the full design rationale.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ----------------------------------------------------------------------------------- scenario presets
# The app exposes exactly two scenarios. The attacker is ALWAYS auto-resolved from the scenario -- it is never a
# user-facing selection in this simplified API (brief section 2).
ScenarioId = Literal["no_clutter", "clutter"]

# Every supported defender id, reused verbatim from the existing research/visualizer defender vocabulary
# (`visualizer/backend/replay_requests.py: POLICY_IDS`) -- names are NOT changed here.
DefenderId = Literal["passive", "greedy", "kalman_greedy", "pn", "lead", "rmpc", "ppo", "ppo_kalman", "lr_ppo"]
SUPPORTED_DEFENDERS: tuple[str, ...] = ("passive", "greedy", "kalman_greedy", "pn", "lead", "rmpc", "ppo", "ppo_kalman", "lr_ppo")

A1_CBF_TAG = "journal-cbf-attacker-v1"
A1_CBF_SOURCE_COMMIT = "a989217b38dc0316718d04e41b065943fec29ce7"
A1_CBF_LABEL = "A1-CBF \u2014 Optimization-Based Evasive Attacker"


class ScenarioPreset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario: ScenarioId
    scenario_label: str
    clutter_level: Literal["C0", "C2"]
    resolved_attacker: Literal["A0", "A1-CBF"]
    attacker_label: str
    dynamic_obstacles_enabled: bool
    viability_conditioned_init: bool
    defender_cbf_enabled: bool


# Internal mapping (brief section 2) -- frozen, not user-configurable through the simplified API.
SCENARIO_PRESETS: dict[str, ScenarioPreset] = {
    "no_clutter": ScenarioPreset(
        scenario="no_clutter", scenario_label="No Clutter", clutter_level="C0", resolved_attacker="A0",
        attacker_label="A0 \u2014 Scripted Pursuit Attacker", dynamic_obstacles_enabled=False,
        viability_conditioned_init=False, defender_cbf_enabled=False,
    ),
    "clutter": ScenarioPreset(
        scenario="clutter", scenario_label="Clutter", clutter_level="C2", resolved_attacker="A1-CBF",
        attacker_label=A1_CBF_LABEL, dynamic_obstacles_enabled=False, viability_conditioned_init=True,
        defender_cbf_enabled=True,
    ),
}


# ----------------------------------------------------------------------------------- API request/response models
class EpisodeCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: ScenarioId
    defender: DefenderId
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


class EpisodeMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    created_at: str
    scenario: ScenarioId
    scenario_label: str
    resolved_attacker: Literal["A0", "A1-CBF"]
    defender: DefenderId
    seed: int
    status: Literal["completed", "failed"]
    outcome: str
    duration: float
    steps: int
    asset_captured: bool
    defender_interception: bool
    obstacle_collision: bool
    replay_path: str
    source_commit: str
    controller_version: str
    is_new: bool
    min_defender_separation_m: float | None = None
    min_obstacle_clearance_m: float | None = None
    path_length_m: float | None = None
    error_message: str | None = None


class EpisodeListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episodes: list[EpisodeMetadata]
    total: int
    new_count: int
    saved_count: int


class BulkDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_ids: list[str] = Field(min_length=1, max_length=500)


class DeleteOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    deleted: bool
    reason: str | None = None


class BulkDeleteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[DeleteOutcome]

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CLUTTER_ARCHITECTURES = {
    "ppo": ("ppo_clutter_v1", (71, 72, 73), "measurement", True, False),
    "ppo_kalman": ("ppo_kalman_clutter_v1", (74, 75, 76), "kalman", True, False),
    "lr_ppo": ("lrppo_clutter_v1", (77, 78, 79), "measurement", True, True),
}
CONFERENCE_ARCHITECTURES = {
    "ppo": ("direct", (45, 46, 47), "measurement", 16),
    "ppo_kalman": ("kalman", (45, 46, 47), "kalman", 16),
    "lr_ppo": ("lr_ppo", (55, 56, 57), "measurement", 19),
}
ATTACKER_SEEDS = {"A1-201": 201, "A1-202": 202, "A1-203": 203}
# "passive" is the research validation harness's parked defender (see
# experiments.journal_v2_validation_core: policy None, zero action every step); it is analytical
# in the sense that it has no architecture, generation, seed or checkpoint.
POLICY_IDS = ("greedy", "kalman_greedy", "pn", "lead", "rmpc", "ppo", "ppo_kalman", "lr_ppo", "passive")
CLUTTER_LEVELS = ("C0", "C1", "C2", "C3")
PERCEPTION_MODES = ("local", "map_local")
# Simulation semantics selector. The visualizer only forwards this choice to the
# research repository's own config builders; it never implements the semantics.
#   V1: the original environment (endpoint mission contact, legacy weaving A0).
#   V2: corrected semantics (swept mission contact, direct obstacle-only A0).
SEMANTICS = ("V1", "V2")
DEFAULT_SEMANTICS = "V1"
MAX_EVALUATION_SEED = 2**32 - 1


class ReplayRequest(BaseModel):
    """Allowlisted single-episode request; never contains an artifact path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    defender_policy_id: Literal["greedy", "kalman_greedy", "pn", "lead", "rmpc", "ppo", "ppo_kalman", "lr_ppo", "passive"]
    defender_architecture: str | None = None
    defender_generation: Literal["conference", "d0", "d0_plus", "d1"] | None = None
    defender_training_seed: Annotated[int, Field(ge=0, strict=True)] | None = None
    checkpoint_role: Literal["conference_best", "d0_start", "final", "secondary_best"] | None = None
    attacker_condition: Literal["A0", "A1-201", "A1-202", "A1-203"]
    attacker_model_seed: Annotated[int, Field(ge=0, strict=True)] | None = None
    clutter_level: Literal["C0", "C1", "C2", "C3"]
    perception_mode: Literal["local", "map_local"] = "map_local"
    semantics: Literal["V1", "V2"] = DEFAULT_SEMANTICS
    evaluation_seed: Annotated[int, Field(ge=0, le=MAX_EVALUATION_SEED, strict=True)]
    deterministic_inference: Literal[True] = True

    @model_validator(mode="after")
    def validate_selection(self):
        learned = self.defender_policy_id in CLUTTER_ARCHITECTURES
        if learned:
            if self.defender_generation is None or self.defender_training_seed is None:
                raise ValueError("Learned defenders require an explicit generation and training seed.")
            if self.defender_generation == "conference":
                family, seeds, _track, _shape = CONFERENCE_ARCHITECTURES[self.defender_policy_id]
            else:
                family, seeds, _track, _sectors, _lr = CLUTTER_ARCHITECTURES[self.defender_policy_id]
            if self.defender_training_seed not in seeds:
                raise ValueError(
                    f"{self.defender_policy_id} {self.defender_generation} supports seeds {seeds}, "
                    f"not {self.defender_training_seed}."
                )
            if self.defender_architecture is not None and self.defender_architecture != family:
                raise ValueError(
                    f"{self.defender_policy_id} {self.defender_generation} requires architecture "
                    f"'{family}', not '{self.defender_architecture}'."
                )
            expected_role = {
                "conference": "conference_best",
                "d0": "d0_start",
                "d0_plus": "final",
                "d1": "final",
            }[self.defender_generation]
            if self.checkpoint_role is not None and self.checkpoint_role != expected_role:
                raise ValueError(
                    f"{self.defender_generation.upper()} requires checkpoint role '{expected_role}'; "
                    "secondary best checkpoints are not selectable."
                )
        elif any(value is not None for value in (
            self.defender_architecture, self.defender_generation,
            self.defender_training_seed, self.checkpoint_role,
        )):
            raise ValueError("Analytical defenders do not have an architecture, generation, seed, or checkpoint role.")

        expected_attacker_seed = ATTACKER_SEEDS.get(self.attacker_condition)
        if self.attacker_model_seed != expected_attacker_seed:
            if expected_attacker_seed is None and self.attacker_model_seed is None:
                return self
            raise ValueError(
                f"{self.attacker_condition} requires attacker_model_seed={expected_attacker_seed}."
            )
        return self


class ReplayBatchRequest(BaseModel):
    """A bounded batch of independent deterministic episodes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    defender_policy_id: Literal["greedy", "kalman_greedy", "pn", "lead", "rmpc", "ppo", "ppo_kalman", "lr_ppo", "passive"]
    defender_generation: Literal["conference", "d0", "d0_plus", "d1"] | None = None
    defender_training_seed: Annotated[int, Field(ge=0, strict=True)] | None = None
    attacker_condition: Literal["A0", "A1-201", "A1-202", "A1-203"]
    clutter_level: Literal["C0", "C1", "C2", "C3"]
    perception_mode: Literal["local", "map_local"] = "map_local"
    semantics: Literal["V1", "V2"] = DEFAULT_SEMANTICS
    seed_start: Annotated[int, Field(ge=0, le=MAX_EVALUATION_SEED, strict=True)] = 0
    episode_count: Literal[1, 3, 5, 10] = 5
    deterministic_inference: Literal[True] = True

    @model_validator(mode="after")
    def validate_batch(self):
        last_seed = self.seed_start + self.episode_count - 1
        if last_seed > MAX_EVALUATION_SEED:
            raise ValueError(f"Consecutive evaluation seeds must not exceed {MAX_EVALUATION_SEED}.")
        for seed in range(self.seed_start, last_seed + 1):
            ReplayRequest(
                defender_policy_id=self.defender_policy_id,
                defender_generation=self.defender_generation,
                defender_training_seed=self.defender_training_seed,
                attacker_condition=self.attacker_condition,
                attacker_model_seed=ATTACKER_SEEDS.get(self.attacker_condition),
                clutter_level=self.clutter_level,
                perception_mode=self.perception_mode,
                semantics=self.semantics,
                evaluation_seed=seed,
                deterministic_inference=self.deterministic_inference,
            )
        return self

    def episode_request(self, evaluation_seed: int) -> ReplayRequest:
        return ReplayRequest(
            defender_policy_id=self.defender_policy_id,
            defender_generation=self.defender_generation,
            defender_training_seed=self.defender_training_seed,
            attacker_condition=self.attacker_condition,
            attacker_model_seed=ATTACKER_SEEDS.get(self.attacker_condition),
            clutter_level=self.clutter_level,
            perception_mode=self.perception_mode,
            semantics=self.semantics,
            evaluation_seed=evaluation_seed,
            deterministic_inference=self.deterministic_inference,
        )

"""Versioned execution treatment for one policy proposal per query."""
from dataclasses import dataclass
import json
from pathlib import Path

from src.memory.decision_context import (
    DecisionContext,
    projected_facts,
    relevant_instruction,
    snapshot_facts,
)

DEFAULT_DECISION_PROTOCOL = (
    Path(__file__).resolve().parents[2] / "configs/benchmarks/libero_mem_decision_dev_v1.json"
)


@dataclass(frozen=True)
class ExecutionTreatment:
    mode: str
    exposure: str
    max_queries: int
    periodic_prefix: int
    minimum_prefix: int
    persistence: int

    @property
    def arm(self):
        return f"{self.mode}-{self.exposure}" if self.mode in ("shadow", "event") else self.mode


def load_decision_protocol(path=DEFAULT_DECISION_PROTOCOL):
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if (protocol["protocol_id"] != "libero_mem_decision_dev_v1"
            or protocol["stage"] != "development" or protocol["split"] != "seen"
            or protocol["abstention"] != "terminate_episode_as_failure"):
        raise ValueError("Unknown decision protocol")
    horizon = protocol["parent_execution_horizon"]
    if not (1 <= protocol["minimum_prefix_actions"] <= protocol["periodic_prefix_actions"] <= horizon
            and 1 <= protocol["semantic_persistence_observations"] < horizon
            and protocol["max_policy_queries"] * protocol["periodic_prefix_actions"]
            >= protocol["policy_actions"] > 0):
        raise ValueError("Invalid decision protocol budgets")
    annotation = protocol["annotation_state_indices"]
    if sorted(annotation["development"] + annotation["calibration"]) != list(range(20)):
        raise ValueError("Annotation groups must partition the 20 locked state indices")
    return protocol


def execution_treatment(protocol, mode, exposure):
    if mode not in protocol["modes"] or exposure not in protocol["exposures"]:
        raise ValueError("Mode or exposure is absent from the decision protocol")
    return ExecutionTreatment(mode, exposure, protocol["max_policy_queries"],
                              protocol["periodic_prefix_actions"], protocol["minimum_prefix_actions"],
                              protocol["semantic_persistence_observations"])


class MemorySupervisor:
    def __init__(self, treatment):
        self.treatment = treatment
        self.context = None
        self.pending = {}
        self.reported = set()
        self.last_observation = None

    def _facts(self, instruction, snapshot, raw, renderer, rendered):
        if not relevant_instruction(instruction, raw["profile"]):
            raise ValueError("Instruction is unsupported by the relation decision contract")
        if self.treatment.exposure == "snapshot":
            return snapshot_facts(raw)
        if rendered is None:
            rendered = renderer.render(instruction, snapshot)
        return projected_facts(renderer, snapshot, rendered)

    def bind(self, query_id, instruction, snapshot, raw, renderer=None, rendered=None):
        facts = self._facts(instruction, snapshot, raw, renderer, rendered)
        self.context = DecisionContext(query_id, raw["as_of_observation"], self.treatment.exposure, facts)
        self.last_observation = self.context.observation
        self.pending.clear()
        self.reported.clear()
        return self.context

    def evaluate(self, instruction, snapshot, raw, renderer=None):
        """Report persistent semantic changes relative to the active query context."""
        observation = raw["as_of_observation"]
        prefix = observation - self.context.observation
        if observation != self.last_observation + 1:
            raise ValueError("Decision checks require consecutive post-action observations")
        self.last_observation = observation
        original = {fact.key: fact for fact in self.context.facts}
        current = {fact.key: fact for fact in self._facts(instruction, snapshot, raw, renderer, None)}
        relevant_keys = original.keys() | current.keys()
        for key in self.pending.keys() - relevant_keys:
            del self.pending[key]
        changes = []
        for key in sorted(relevant_keys):
            before, after = original.get(key), current.get(key)
            old_version = before.version if before else None
            new_version = after.version if after else None
            if old_version == new_version:
                self.pending.pop(key, None)
                continue
            # Acquiring an unknown relation conveys no new usable history.
            if before is None and after.validity in ("unknown", "conflicted"):
                self.pending.pop(key, None)
                continue
            previous, start = self.pending.get(key, (None, observation))
            if previous != new_version:
                start = observation
            self.pending[key] = (new_version, start)
            marker = (key, new_version)
            if (prefix >= self.treatment.minimum_prefix
                    and observation - start + 1 >= self.treatment.persistence
                    and marker not in self.reported):
                self.reported.add(marker)
                changes.append(dict(key=key, first_observation=start,
                                    before=before.trace() if before else None,
                                    after=after.trace() if after else None))
        return changes

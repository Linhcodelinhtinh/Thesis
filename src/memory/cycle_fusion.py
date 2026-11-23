"""Count bowl returns per grasp cycle from two sources: a vision writer and the proprioceptive touchdown cue.

A grasp cycle is one closed-gripper phase (aperture <= closed, then >= released). Each source may report the same
return at different times: the touchdown cue fires inside the closed phase, while a vision writer confirms the return
up to ~30 steps later, often after the gripper has opened. Merging by a fixed frame gap double counts such pairs, so
every event is credited to a cycle instead, and a cycle counts once whichever sources reported it:

- a touchdown event belongs to the closed phase it fired in;
- a vision return belongs to the latest cycle that started at or before the event's `occurred_at` step
  (or to a "pre-grasp" slot when no cycle had started yet).

Streaming and causal: `step` only uses the current frame and the vision events published on it.
It edits no writer, so the memory prompt text is unchanged.
"""

from dataclasses import dataclass

from src.memory.proprio_cycle_cue import ProprioCueConfig, ProprioCycleCue


@dataclass(frozen=True)
class CycleCredit:
    cycle: int          # index of the closed phase; -1 means before any grasp
    source: str         # "proprio" or "vision"
    step: int           # frame at which the credit became available


class CycleFusionCounter:
    def __init__(self, cue_config=None):
        self.cue_config = cue_config or ProprioCueConfig.load()
        self.cue = ProprioCycleCue(self.cue_config)
        self.cycle_starts = []      # start frame of every closed phase seen so far
        self.closed = False
        self.credits = {}           # cycle -> first CycleCredit
        self.duplicates = []        # credits that landed on an already credited cycle
        self._last_step = None

    def count_at(self, step):
        """Cycles credited by `step` (a touchdown is only available after its confirmation frames)."""
        return sum(credit.step <= step for credit in self.credits.values())

    def sources(self):
        return sorted({credit.source for credit in self.credits.values()})

    def _credit(self, cycle, source, step):
        credit = CycleCredit(cycle, source, step)
        if cycle in self.credits:
            self.duplicates.append(credit)
        else:
            self.credits[cycle] = credit

    def _cycle_at(self, occurred_at):
        started = [index for index, start in enumerate(self.cycle_starts) if start <= occurred_at]
        return started[-1] if started else -1

    def step(self, step, aperture, eef_pos, vision_returns=()):
        """Advance one frame. `vision_returns` are the writer's new return events (dicts with `occurred_at`)."""
        if self._last_step is not None and step != self._last_step + 1:
            raise ValueError("CycleFusionCounter needs consecutive frames")
        self._last_step = step
        config = self.cue_config
        if not self.closed and aperture <= config.closed_aperture_meters:
            self.closed = True
            self.cycle_starts.append(step)
        elif self.closed and aperture >= config.released_aperture_meters:
            self.closed = False
        touchdown = self.cue.step(step, aperture, eef_pos)
        if touchdown is not None:
            # The cue resets with each closed phase, so its event belongs to the current one.
            self._credit(len(self.cycle_starts) - 1, "proprio", touchdown["available_at"])
        for event in vision_returns:
            self._credit(self._cycle_at(int(event["occurred_at"])), "vision", step)
        return self.count_at(step)

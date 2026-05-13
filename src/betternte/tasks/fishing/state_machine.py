from __future__ import annotations

import time

from betternte.tasks.fishing.models import FishingState, Observation, Suggestion

_MAX_EMPTY_CYCLES = 3

# 时间阈值（秒）
_CASTING_MIN_WAIT = 0.8
_CASTING_MAX_WAIT = 5.0
_WAITING_BITE_TIMEOUT = 15.0
_HOOKING_TIMEOUT = 2.0
_CONTROLLING_MAX_DURATION = 60.0
_MISSING_BAR_TIMEOUT = 0.4
_WAITING_BAR_TIMEOUT = 0.1
_HOOKING_BAR_TIMEOUT = 0.06
_FINISHED_CLICK_DELAY = 3.0
_FINISHED_CAST_DELAY = 5.0


class FishingStateMachine:
    def __init__(self) -> None:
        self.state = FishingState.IDLE
        self._entered_at = time.monotonic()
        self._f_triggered = False
        self._bar_missing_since: float | None = None
        self._waiting_bar_since: float | None = None
        self._hooking_bar_since: float | None = None
        self._empty_cycles = 0
        self._ever_saw_bar = False

    def reset(self) -> None:
        self.state = FishingState.IDLE
        self._entered_at = time.monotonic()
        self._f_triggered = False
        self._bar_missing_since = None
        self._waiting_bar_since = None
        self._hooking_bar_since = None
        self._empty_cycles = 0
        self._ever_saw_bar = False

    def start(self) -> None:
        self._transition(FishingState.CASTING)

    def _transition(self, new_state: FishingState) -> None:
        self.state = new_state
        self._entered_at = time.monotonic()
        if new_state == FishingState.CASTING:
            self._f_triggered = False
        if new_state == FishingState.HOOKING:
            self._f_triggered = False
            self._hooking_bar_since = None
        if new_state == FishingState.FINISHED:
            self._f_triggered = False
        if new_state != FishingState.CONTROLLING:
            self._bar_missing_since = None

    def step(self, observation: Observation) -> tuple[FishingState, Suggestion]:
        now = time.monotonic()
        elapsed = now - self._entered_at
        suggestion = Suggestion.NONE

        if self.state == FishingState.IDLE:
            return self.state, suggestion

        if self.state == FishingState.CASTING:
            if not self._f_triggered:
                suggestion = Suggestion.PRESS_F
                self._f_triggered = True
            if self._f_triggered and elapsed >= _CASTING_MIN_WAIT:
                self._transition(FishingState.WAITING_BITE)
            elif elapsed >= _CASTING_MAX_WAIT:
                self._transition(FishingState.WAITING_BITE)

        elif self.state == FishingState.WAITING_BITE:
            if observation.blue_circle is not None and observation.blue_circle.found:
                self._transition(FishingState.HOOKING)
            elif observation.bar_visible:
                if self._waiting_bar_since is None:
                    self._waiting_bar_since = now
                elif now - self._waiting_bar_since >= _WAITING_BAR_TIMEOUT:
                    self._ever_saw_bar = True
                    self._empty_cycles = 0
                    self._transition(FishingState.CONTROLLING)
            else:
                self._waiting_bar_since = None
                if elapsed >= _WAITING_BITE_TIMEOUT:
                    self._empty_cycles += 1
                    if self._empty_cycles >= _MAX_EMPTY_CYCLES:
                        self._transition(FishingState.IDLE)
                    else:
                        self._transition(FishingState.CASTING)

        elif self.state == FishingState.HOOKING:
            if not self._f_triggered:
                suggestion = Suggestion.PRESS_F
                self._f_triggered = True
            if self._f_triggered:
                if observation.bar_visible:
                    if self._hooking_bar_since is None:
                        self._hooking_bar_since = now
                    elif now - self._hooking_bar_since >= _HOOKING_BAR_TIMEOUT:
                        self._ever_saw_bar = True
                        self._empty_cycles = 0
                        self._transition(FishingState.CONTROLLING)
                else:
                    self._hooking_bar_since = None
                    if elapsed >= _HOOKING_TIMEOUT:
                        self._empty_cycles += 1
                        if self._empty_cycles >= _MAX_EMPTY_CYCLES:
                            self._transition(FishingState.IDLE)
                        else:
                            self._transition(FishingState.CASTING)

        elif self.state == FishingState.CONTROLLING:
            if elapsed >= _CONTROLLING_MAX_DURATION:
                self._transition(FishingState.FINISHED)
            elif observation.bar_visible:
                self._bar_missing_since = None
            else:
                if self._bar_missing_since is None:
                    self._bar_missing_since = now
                elif now - self._bar_missing_since >= _MISSING_BAR_TIMEOUT:
                    if self._ever_saw_bar:
                        self._transition(FishingState.FINISHED)
                    else:
                        self._empty_cycles += 1
                        if self._empty_cycles >= _MAX_EMPTY_CYCLES:
                            self._transition(FishingState.IDLE)
                        else:
                            self._transition(FishingState.CASTING)

        elif self.state == FishingState.FINISHED:
            if elapsed >= _FINISHED_CLICK_DELAY and not self._f_triggered:
                suggestion = Suggestion.CLICK_SCREEN
                self._f_triggered = True
            if self._f_triggered and elapsed >= _FINISHED_CAST_DELAY:
                self._empty_cycles = 0
                self._transition(FishingState.CASTING)

        return self.state, suggestion

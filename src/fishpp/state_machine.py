from __future__ import annotations

import time

from fishpp.models import FishingState, Observation, Suggestion

# 防止空转：如果连续 N 次 FINISHED→CASTING 循环都没见到 bar，自动暂停
_MAX_EMPTY_CYCLES = 3


class FishingStateMachine:
    def __init__(self) -> None:
        self.state = FishingState.IDLE
        self._entered_at = time.monotonic()
        self._f_triggered = False
        self._missing_bar_frames = 0
        self._waiting_hook_bar_frames = 0
        self._waiting_bar_frames = 0
        self._empty_cycles = 0
        self._ever_saw_bar = False

    def reset(self) -> None:
        self.state = FishingState.IDLE
        self._entered_at = time.monotonic()
        self._f_triggered = False
        self._missing_bar_frames = 0
        self._waiting_hook_bar_frames = 0
        self._waiting_bar_frames = 0
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
            self._waiting_hook_bar_frames = 0
        if new_state == FishingState.FINISHED:
            self._f_triggered = False
        if new_state != FishingState.CONTROLLING:
            self._missing_bar_frames = 0

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
            if self._f_triggered and elapsed >= 0.8:
                self._transition(FishingState.WAITING_BITE)
            elif elapsed >= 5.0:
                self._transition(FishingState.WAITING_BITE)

        elif self.state == FishingState.WAITING_BITE:
            if observation.blue_circle is not None and observation.blue_circle.found:
                self._transition(FishingState.HOOKING)
            elif observation.bar_visible:
                # 已经在钓鱼中（手动 F 或漏掉了蓝圈）→ 直接接管操控
                self._waiting_bar_frames += 1
                if self._waiting_bar_frames >= 5:
                    self._ever_saw_bar = True
                    self._empty_cycles = 0
                    self._transition(FishingState.CONTROLLING)
            else:
                self._waiting_bar_frames = 0
                if elapsed >= 15.0:
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
                    self._ever_saw_bar = True
                    self._waiting_hook_bar_frames += 1
                    if self._waiting_hook_bar_frames >= 3:
                        self._empty_cycles = 0
                        self._transition(FishingState.CONTROLLING)
                elif elapsed >= 2.0:
                    # F 键按了但 bar 没出现，说明没生效
                    self._empty_cycles += 1
                    if self._empty_cycles >= _MAX_EMPTY_CYCLES:
                        self._transition(FishingState.IDLE)
                    else:
                        self._transition(FishingState.CASTING)

        elif self.state == FishingState.CONTROLLING:
            if observation.bar_visible:
                self._missing_bar_frames = 0
            else:
                self._missing_bar_frames += 1
                if self._missing_bar_frames >= 15:
                    if self._ever_saw_bar:
                        # 之前见过 bar，现在消失了 → 钓到了
                        self._transition(FishingState.FINISHED)
                    else:
                        # 从来没见过 bar → 异常，回到 CASTING
                        self._empty_cycles += 1
                        if self._empty_cycles >= _MAX_EMPTY_CYCLES:
                            self._transition(FishingState.IDLE)
                        else:
                            self._transition(FishingState.CASTING)

        elif self.state == FishingState.FINISHED:
            if elapsed >= 3.0 and not self._f_triggered:
                suggestion = Suggestion.CLICK_SCREEN
                self._f_triggered = True
            if self._f_triggered and elapsed >= 5.0:
                self._empty_cycles = 0
                self._transition(FishingState.CASTING)

        return self.state, suggestion

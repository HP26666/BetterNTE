from __future__ import annotations

import queue
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from betternte.core.task_executor import TaskExecutor
from betternte.core.input import InputController, is_admin
from betternte.tasks.base import TriggerTask
from betternte.tasks.fishing.controller import apply_suggestion, compute_ad_pulse
from betternte.tasks.fishing.models import AppConfig, FishingState, Observation, ResultPacket, ROI, Suggestion
from betternte.tasks.fishing.state_machine import FishingStateMachine
from betternte.tasks.fishing.vision import analyze_frame


class FishingTask(TriggerTask):
    """由 TaskExecutor 驱动的自动钓鱼任务。"""

    name = "fishing"
    display_name = "自动钓鱼"
    description = "HSV 检测 + 状态机 + PD 控制的自动钓鱼任务"
    requires_input = True

    def __init__(self, config: AppConfig, result_queue: queue.Queue[ResultPacket]) -> None:
        self.config = config.clone()
        self.result_queue = result_queue
        self.state_machine = FishingStateMachine()
        self._running = False
        self._dot_x_smoothed: float | None = None
        self._bar_center_smoothed: float | None = None
        self._prev_error_ratio: float | None = None
        self._prev_bar_center: float | None = None
        self._last_logged_state: FishingState | None = None
        self._frame_count = 0
        self._last_stop_reason = ""
        self._suppress_next_stop_log = False
        self._sync_trigger_interval()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_stop_reason(self) -> str:
        return self._last_stop_reason

    def suppress_next_stop_log(self) -> None:
        self._suppress_next_stop_log = True

    def update_config(self, config: AppConfig) -> None:
        self.config = config.clone()
        self._sync_trigger_interval()
        if self.input_ctrl is not None and hasattr(self.input_ctrl, "set_screen_size"):
            self.input_ctrl.set_screen_size((self.config.screen.width, self.config.screen.height))

    def start(self) -> None:
        self._running = True
        self._last_stop_reason = ""
        self.state_machine.reset()
        self.state_machine.start()
        self._dot_x_smoothed = None
        self._bar_center_smoothed = None
        self._prev_error_ratio = None
        self._prev_bar_center = None
        self._last_logged_state = None
        self._frame_count = 0
        if self.input_ctrl is not None and hasattr(self.input_ctrl, "set_screen_size"):
            self.input_ctrl.set_screen_size((self.config.screen.width, self.config.screen.height))
        self._emit_log(
            f"[Worker] 启动"
            f" · 显示器={self.config.capture.monitor_index}"
            f" · 分辨率={self.config.screen.width}x{self.config.screen.height}"
            f" · 目标FPS={self.config.control.capture_fps()}"
            f" · 实际约={self.config.control.actual_capture_fps()}"
            f" · 管理员={'是' if is_admin() else '否'}"
        )

    def stop(self, reason: str = "用户停止") -> None:
        self._running = False
        self._last_stop_reason = reason
        if self.input_ctrl is not None:
            self.input_ctrl.release_all()
        if reason and not self._suppress_next_stop_log:
            self._emit_log(reason)
        self._suppress_next_stop_log = False

    def run(self, frame) -> None:
        if not self._running:
            return
        if self.input_ctrl is None:
            raise RuntimeError("FishingTask 未注入 input_ctrl")

        source_roi = self.latest_source_roi
        if source_roi is None:
            source_roi = ROI(name="screen", x=0, y=0, w=frame.shape[1], h=frame.shape[0])

        observation = self._sanitize_observation(analyze_frame(frame, source_roi, self.config))
        state, suggestion = self.state_machine.step(observation)

        self._dot_x_smoothed = self._smooth(self._dot_x_smoothed, observation.dot.x if observation.dot else None)
        self._bar_center_smoothed = self._smooth(self._bar_center_smoothed, observation.bar.center if observation.bar else None)

        if state == FishingState.CONTROLLING:
            bar_velocity = 0.0
            if self._bar_center_smoothed is not None and self._prev_bar_center is not None:
                bar_velocity = abs(self._bar_center_smoothed - self._prev_bar_center)
            self._prev_bar_center = self._bar_center_smoothed

            suggestion, pulse_duration = compute_ad_pulse(
                observation.bar,
                observation.dot,
                self.config.control,
                dot_x=self._dot_x_smoothed,
                bar_center=self._bar_center_smoothed,
                prev_error_ratio=self._prev_error_ratio,
            )

            should_hold = False
            if suggestion in (Suggestion.A, Suggestion.D) and observation.bar is not None:
                half = (observation.bar.right - observation.bar.left) / 2.0
                dot_error = abs(self._dot_x_smoothed - self._bar_center_smoothed) if (self._dot_x_smoothed is not None and self._bar_center_smoothed is not None) else 0.0
                if bar_velocity > 1.5 or (half > 0 and dot_error > half * 0.25):
                    should_hold = True

            if observation.bar is not None:
                half = (observation.bar.right - observation.bar.left) / 2.0
                if half > 0 and self._dot_x_smoothed is not None and self._bar_center_smoothed is not None:
                    self._prev_error_ratio = (self._dot_x_smoothed - self._bar_center_smoothed) / half
                else:
                    self._prev_error_ratio = None
            else:
                self._prev_error_ratio = None
        else:
            pulse_duration = 0.0
            should_hold = False
            self._prev_error_ratio = None
            self._prev_bar_center = None

        if state != self._last_logged_state:
            self._emit_log(f"[状态] {self._last_logged_state.value if self._last_logged_state else 'START'} → {state.value}")
            self._last_logged_state = state

        self._frame_count += 1
        if self._frame_count % 60 == 0:
            bar_ok = observation.bar is not None
            dot_ok = observation.dot is not None
            blue_ok = observation.blue_circle is not None and observation.blue_circle.found
            self._emit_log(
                f"[帧{self._frame_count}] {state.value} · bar={'Y' if bar_ok else 'N'} line={'Y' if dot_ok else 'N'} blue={'Y' if blue_ok else 'N'}"
            )

        if suggestion != Suggestion.NONE:
            mode_str = "HOLD" if should_hold else "tap"
            self._emit_log(f"[动作] suggestion={suggestion.value} · {mode_str}")

        packet = ResultPacket(
            frame=frame,
            source_roi=source_roi,
            observation=observation,
            state=state,
            suggestion=suggestion,
            message=self._build_message(observation, state, suggestion),
            dot_x_smoothed=self._dot_x_smoothed,
            bar_center_smoothed=self._bar_center_smoothed,
        )
        self._push_result(packet)

        click_roi = None
        if suggestion == Suggestion.CLICK_SCREEN:
            click_roi = self.config.rois.cast_click_area
            if click_roi is None or not click_roi.valid():
                cx = source_roi.x + source_roi.w // 2
                cy = source_roi.y + source_roi.h // 2
                click_roi = ROI(name="screen_center", x=cx - 50, y=cy - 25, w=100, h=50)
                self._emit_log(f"[动作] cast_click_area 未配置，使用屏幕中心 ({cx},{cy})")
        apply_suggestion(self.input_ctrl, suggestion, click_roi, pulse_duration, should_hold)

        if self.input_ctrl.emergency_corner_stop(source_roi):
            self.stop("鼠标移动到屏幕角落，已触发紧急停止")

    def _sync_trigger_interval(self) -> None:
        self.trigger_interval = max(0.01, self.config.control.loop_interval_ms / 1000.0)

    def _sanitize_observation(self, observation: Observation) -> Observation:
        bar = observation.bar
        dot = observation.dot

        if bar is not None and self._bar_center_smoothed is not None:
            half = max(1.0, (bar.right - bar.left) / 2.0)
            max_bar_jump = max(80.0, half * 3.0)
            if abs(bar.center - self._bar_center_smoothed) > max_bar_jump:
                bar = None

        if dot is not None and bar is not None:
            half = max(1.0, (bar.right - bar.left) / 2.0)
            if dot.x < bar.left - half * 1.3 or dot.x > bar.right + half * 1.3:
                dot = None
            elif dot.y < bar.bbox.y - max(10, bar.bbox.h * 2) or dot.y > bar.bbox.y + bar.bbox.h + max(10, bar.bbox.h * 2):
                dot = None

        if dot is not None and self._dot_x_smoothed is not None:
            reference_half = 40.0
            if bar is not None:
                reference_half = max(reference_half, (bar.right - bar.left) / 2.0)
            max_dot_jump = max(90.0, reference_half * 3.0)
            if abs(dot.x - self._dot_x_smoothed) > max_dot_jump:
                dot = None

        return Observation(
            bar=bar,
            dot=dot,
            blue_circle=observation.blue_circle,
            bar_visible=bar is not None,
            dot_visible=dot is not None,
        )

    def _emit_log(self, message: str) -> None:
        if self.executor is not None:
            self.executor.log_message.emit(message)

    def _smooth(self, previous: float | None, current: int | None) -> float | None:
        if current is None:
            return previous
        if previous is None:
            return float(current)
        alpha = min(max(self.config.control.smoothing, 0.0), 1.0)
        return previous * (1.0 - alpha) + float(current) * alpha

    def _push_result(self, packet: ResultPacket) -> None:
        try:
            self.result_queue.put_nowait(packet)
        except queue.Full:
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.result_queue.put_nowait(packet)
            except queue.Full:
                pass

    def _build_message(self, observation: Observation, state: FishingState, suggestion: Suggestion) -> str:
        bar_text = f"bar={observation.bar.center}" if observation.bar else "bar=None"
        dot_text = f"line={observation.dot.x}" if observation.dot else "line=None"
        blue_text = "blue=1" if observation.blue_circle and observation.blue_circle.found else "blue=0"
        return f"state={state.value} suggestion={suggestion.value} {bar_text} {dot_text} {blue_text}"


class VisionWorker(QObject):
    log_message = Signal(str)
    stopped = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        base_path: Path,
        config: AppConfig,
        result_queue: queue.Queue[ResultPacket],
        input_controller: Any = None,  # InputController 或 PostMessageController
    ) -> None:
        super().__init__()
        self.base_path = base_path
        if input_controller is None:
            input_controller = InputController()
        self.task = FishingTask(config, result_queue)
        self.executor = TaskExecutor(
            input_controller=input_controller,
            capture_config=self.task.config.capture,
            frame_interval_ms=self.task.config.control.loop_interval_ms,
        )
        self.executor.register_trigger_task(self.task)
        self.executor.log_message.connect(self.log_message.emit)
        self.executor.task_failed.connect(self._on_task_failed)
        self.executor.task_state_changed.connect(self._on_task_state_changed)

    def update_config(self, config: AppConfig) -> None:
        self.task.update_config(config)
        self.executor.set_capture_config(self.task.config.capture)
        self.executor.frame_interval_ms = max(1, self.task.config.control.loop_interval_ms)

    def stop(self, reason: str = "用户停止") -> None:
        self.task.suppress_next_stop_log()
        self.executor.stop_task(self.task.name, reason)
        self.executor.stop()

    def start(self) -> None:
        self.executor.start()
        self.executor.start_task(self.task.name)

    def wait(self, timeout: int = 0) -> bool:
        return self.executor.wait(timeout)

    @property
    def is_running(self) -> bool:
        return self.task.is_running

    def _on_task_failed(self, name: str, error_message: str) -> None:
        if name == self.task.name:
            self.failed.emit(error_message)

    def _on_task_state_changed(self, name: str, running: bool) -> None:
        if name == self.task.name and not running:
            self.stopped.emit(self.task.last_stop_reason or "任务停止")


__all__ = ["FishingTask", "VisionWorker"]

from __future__ import annotations

import queue
import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from betternte.capture import ScreenCapture
from betternte.control import InputController, compute_ad_pulse, is_admin
from betternte.models import AppConfig, FishingState, ResultPacket, ROI, Suggestion
from betternte.state_machine import FishingStateMachine
from betternte.vision import analyze_frame


class VisionWorker(QThread):
    log_message = Signal(str)
    stopped = Signal(str)
    failed = Signal(str)

    def __init__(self, base_path: Path, config: AppConfig, result_queue: queue.Queue[ResultPacket]) -> None:
        super().__init__()
        self.base_path = base_path
        self.config = config.clone()
        self.result_queue = result_queue
        self.capture = ScreenCapture()
        self.input_controller = InputController()
        self.input_controller.set_screen_size((config.screen.width, config.screen.height))
        self.state_machine = FishingStateMachine()
        self._running = False
        self._dot_x_smoothed: float | None = None
        self._bar_center_smoothed: float | None = None
        self._prev_error_ratio: float | None = None
        self._prev_bar_center: float | None = None
        self._last_logged_state: FishingState | None = None
        self._frame_count = 0

    def update_config(self, config: AppConfig) -> None:
        self.config = config.clone()
        self.input_controller.set_screen_size((config.screen.width, config.screen.height))

    def stop(self, reason: str = "用户停止") -> None:
        self._running = False
        self.input_controller.release_all()
        self.stopped.emit(reason)

    def run(self) -> None:
        self._running = True
        self.state_machine.reset()
        self.state_machine.start()
        self._frame_count = 0
        self._last_logged_state = None

        self.log_message.emit(
            f"[Worker] 启动"
            f" · 显示器={self.config.capture.monitor_index}"
            f" · 分辨率={self.config.screen.width}x{self.config.screen.height}"
            f" · 管理员={'是' if is_admin() else '否'}"
        )
        try:
            while self._running:
                started_at = time.monotonic()
                frame, source_roi = self.capture.grab_source(self.config.capture)
                observation = analyze_frame(frame, source_roi, self.config)
                state, suggestion = self.state_machine.step(observation)

                self._dot_x_smoothed = self._smooth(self._dot_x_smoothed, observation.dot.x if observation.dot else None)
                self._bar_center_smoothed = self._smooth(self._bar_center_smoothed, observation.bar.center if observation.bar else None)

                if state == FishingState.CONTROLLING:
                    # 计算绿条移动速度
                    bar_velocity = 0.0
                    if self._bar_center_smoothed is not None and self._prev_bar_center is not None:
                        bar_velocity = abs(self._bar_center_smoothed - self._prev_bar_center)
                    self._prev_bar_center = self._bar_center_smoothed

                    suggestion, pulse_duration = compute_ad_pulse(
                        observation.bar, observation.dot, self.config.control,
                        dot_x=self._dot_x_smoothed,
                        bar_center=self._bar_center_smoothed,
                        prev_error_ratio=self._prev_error_ratio,
                    )

                    # 决定长按还是点按：绿条快速移动或黄点离中心较远时长按
                    should_hold = False
                    if suggestion in (Suggestion.A, Suggestion.D) and observation.bar is not None:
                        half = (observation.bar.right - observation.bar.left) / 2.0
                        dot_error = abs(self._dot_x_smoothed - self._bar_center_smoothed) if (self._dot_x_smoothed is not None and self._bar_center_smoothed is not None) else 0.0
                        # 条移动快（>1.5px/帧）或黄点偏离中心超过 25% 半宽 → 长按
                        if bar_velocity > 1.5 or (half > 0 and dot_error > half * 0.25):
                            should_hold = True

                    # 为下一帧保存误差比
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

                # 状态变化时日志
                if state != self._last_logged_state:
                    self.log_message.emit(f"[状态] {self._last_logged_state.value if self._last_logged_state else 'START'} → {state.value}")
                    self._last_logged_state = state

                # 每 60 帧输出一次检测摘要（避免刷屏）
                self._frame_count += 1
                if self._frame_count % 60 == 0:
                    bar_ok = observation.bar is not None
                    dot_ok = observation.dot is not None
                    blue_ok = observation.blue_circle is not None and observation.blue_circle.found
                    self.log_message.emit(
                        f"[帧{self._frame_count}] {state.value} · bar={'Y' if bar_ok else 'N'} dot={'Y' if dot_ok else 'N'} blue={'Y' if blue_ok else 'N'}"
                    )

                # 有非空建议时记录
                if suggestion != Suggestion.NONE:
                    mode_str = "HOLD" if should_hold else "tap"
                    self.log_message.emit(f"[动作] suggestion={suggestion.value} · {mode_str}")

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
                        # 回退到屏幕中心
                        cx = self.config.screen.width // 2
                        cy = self.config.screen.height // 2
                        click_roi = ROI(name="screen_center", x=cx - 50, y=cy - 25, w=100, h=50)
                        self.log_message.emit(f"[动作] cast_click_area 未配置，使用屏幕中心 ({cx},{cy})")
                self.input_controller.apply(suggestion, self.config.control, click_roi, pulse_duration, should_hold)

                if self.input_controller.emergency_corner_stop((self.config.screen.width, self.config.screen.height)):
                    self.stop("鼠标移动到屏幕角落，已触发紧急停止")
                    break

                elapsed = (time.monotonic() - started_at) * 1000.0
                wait_ms = max(1.0, self.config.control.loop_interval_ms - elapsed)
                self.msleep(int(wait_ms))
        except Exception as exc:
            self.input_controller.release_all()
            self.failed.emit(str(exc))
        finally:
            self.input_controller.release_all()

    def _smooth(self, previous: float | None, current: int | None) -> float | None:
        if current is None:
            return previous
        if previous is None:
            return float(current)
        alpha = min(max(self.config.control.smoothing, 0.0), 1.0)
        return previous * (1.0 - alpha) + float(current) * alpha

    def _push_result(self, packet: ResultPacket) -> None:
        while True:
            try:
                self.result_queue.put_nowait(packet)
                return
            except queue.Full:
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    return

    def _build_message(self, observation, state: FishingState, suggestion: Suggestion) -> str:
        bar_text = f"bar={observation.bar.center}" if observation.bar else "bar=None"
        dot_text = f"dot={observation.dot.x}" if observation.dot else "dot=None"
        blue_text = "blue=1" if observation.blue_circle and observation.blue_circle.found else "blue=0"
        return f"state={state.value} suggestion={suggestion.value} {bar_text} {dot_text} {blue_text}"

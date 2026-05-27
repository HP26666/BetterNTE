from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Any

import numpy as np
from PySide6.QtCore import QThread, Signal

from betternte.core.capture import ScreenCapture
from betternte.core.models import CaptureConfig, ROI
from betternte.tasks.base import BaseTask, OneTimeTask, TriggerTask


class TaskExecutor(QThread):
    """统一管理帧获取、任务调度和输入所有权。"""

    status_changed = Signal(str)
    paused_changed = Signal(bool)
    task_state_changed = Signal(str, bool)
    task_failed = Signal(str, str)
    log_message = Signal(str)
    frame_updated = Signal(object, object)

    def __init__(
        self,
        capture: ScreenCapture | None = None,
        input_controller: Any | None = None,
        capture_config: CaptureConfig | None = None,
        frame_interval_ms: int = 50,
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self.capture = capture or ScreenCapture()
        self.input_controller = input_controller
        self.capture_config = capture_config or CaptureConfig()
        self.frame_interval_ms = max(1, int(frame_interval_ms))

        self._lock = Lock()
        self._trigger_tasks: dict[str, TriggerTask] = {}
        self._onetime_tasks: dict[str, OneTimeTask] = {}
        self._trigger_last_run: dict[str, float] = {}
        self._pending_onetime: deque[str] = deque()
        self._active_onetime: str | None = None
        self._input_owner: str | None = None
        self._running = False
        self._paused = False
        self._latest_frame: np.ndarray | None = None
        self._latest_source_roi: ROI | None = None

    @property
    def latest_frame(self) -> np.ndarray | None:
        return self._latest_frame

    @property
    def latest_source_roi(self) -> ROI | None:
        return self._latest_source_roi

    def register_trigger_task(self, task: TriggerTask) -> None:
        name = self._task_name(task)
        with self._lock:
            self._ensure_unique(name)
            task.bind_runtime(capture=self.capture, input_ctrl=self.input_controller, executor=self)
            self._trigger_tasks[name] = task
            self._trigger_last_run[name] = 0.0

    def register_onetime_task(self, task: OneTimeTask) -> None:
        name = self._task_name(task)
        with self._lock:
            self._ensure_unique(name)
            task.bind_runtime(capture=self.capture, input_ctrl=self.input_controller, executor=self)
            self._onetime_tasks[name] = task

    def set_capture_config(self, capture_config: CaptureConfig) -> None:
        self.capture_config = capture_config

    def set_input_controller(self, input_controller: Any | None) -> None:
        with self._lock:
            self.input_controller = input_controller
            for task in [*self._trigger_tasks.values(), *self._onetime_tasks.values()]:
                task.bind_runtime(input_ctrl=input_controller)

    def start(self) -> None:  # type: ignore[override]
        if self.isRunning():
            self._paused = False
            self.paused_changed.emit(False)
            return
        self._running = True
        self._paused = False
        self.status_changed.emit("running")
        super().start()

    def stop(self) -> None:
        self._running = False
        self._paused = False
        with self._lock:
            trigger_tasks = list(self._trigger_tasks.values())
            onetime_tasks = list(self._onetime_tasks.values())
        for task in [*trigger_tasks, *onetime_tasks]:
            if task.is_running:
                task.stop("执行器停止")
        self._release_input_owner(None)
        self.status_changed.emit("stopped")

    def pause(self) -> None:
        self._paused = True
        self.paused_changed.emit(True)
        self.status_changed.emit("paused")

    def resume(self) -> None:
        self._paused = False
        self.paused_changed.emit(False)
        self.status_changed.emit("running")

    def start_task(self, name: str) -> bool:
        with self._lock:
            task = self._trigger_tasks.get(name) or self._onetime_tasks.get(name)
        if task is None:
            return False
        if isinstance(task, TriggerTask):
            task.start()
            self.task_state_changed.emit(name, task.is_running)
            return True
        return self.start_onetime_task(name)

    def stop_task(self, name: str, reason: str = "用户停止") -> bool:
        with self._lock:
            task = self._trigger_tasks.get(name) or self._onetime_tasks.get(name)
        if task is None:
            return False
        task.stop(reason)
        self._release_input_owner(name)
        self.task_state_changed.emit(name, False)
        return True

    def start_onetime_task(self, name: str) -> bool:
        with self._lock:
            task = self._onetime_tasks.get(name)
            if task is None:
                return False
            if name not in self._pending_onetime and self._active_onetime != name:
                self._pending_onetime.append(name)
        self.task_state_changed.emit(name, True)
        return True

    def run(self) -> None:
        try:
            while self._running:
                loop_started = time.monotonic()
                if not self._paused:
                    self._capture_frame()
                    self._run_trigger_tasks(loop_started)
                    self._run_onetime_task()
                elapsed_ms = (time.monotonic() - loop_started) * 1000.0
                self.msleep(max(1, int(self.frame_interval_ms - elapsed_ms)))
        finally:
            if self.input_controller is not None:
                self.input_controller.release_all()

    def _capture_frame(self) -> None:
        try:
            frame, source_roi = self.capture.grab_source(self.capture_config)
        except Exception as exc:
            self.log_message.emit(f"截图失败: {exc}")
            return
        self._latest_frame = frame
        self._latest_source_roi = source_roi
        self.frame_updated.emit(frame, source_roi)

    def _run_trigger_tasks(self, now: float) -> None:
        frame = self._latest_frame
        source_roi = self._latest_source_roi
        if frame is None or source_roi is None:
            return
        with self._lock:
            tasks = list(self._trigger_tasks.items())
        for name, task in tasks:
            if not task.is_running:
                self._release_input_owner(name)
                continue
            interval = max(0.01, float(task.trigger_interval))
            last_run = self._trigger_last_run.get(name, 0.0)
            if now - last_run < interval:
                continue
            if not self._claim_input(task):
                continue
            task.latest_source_roi = source_roi
            try:
                task.run(frame)
                self._trigger_last_run[name] = now
            except Exception as exc:
                task.stop(f"任务异常: {exc}")
                self.task_failed.emit(name, str(exc))
            if not task.is_running:
                self._release_input_owner(name)
                self.task_state_changed.emit(name, False)

    def _run_onetime_task(self) -> None:
        if self._active_onetime is not None:
            return
        with self._lock:
            if not self._pending_onetime:
                return
            name = self._pending_onetime.popleft()
            task = self._onetime_tasks.get(name)
        if task is None:
            return
        if not self._claim_input(task):
            with self._lock:
                self._pending_onetime.appendleft(name)
            return
        self._active_onetime = name
        try:
            task.start()
            task.latest_source_roi = self._latest_source_roi
            task.run()
        except Exception as exc:
            task.stop(f"任务异常: {exc}")
            self.task_failed.emit(name, str(exc))
        finally:
            if task.is_running:
                task.stop("任务完成")
            self._active_onetime = None
            self._release_input_owner(name)
            self.task_state_changed.emit(name, False)

    def _task_name(self, task: BaseTask) -> str:
        return task.name or task.__class__.__name__

    def _ensure_unique(self, name: str) -> None:
        if name in self._trigger_tasks or name in self._onetime_tasks:
            raise ValueError(f"任务已注册: {name}")

    def _claim_input(self, task: BaseTask) -> bool:
        if not task.requires_input:
            return True
        name = self._task_name(task)
        if self._input_owner in (None, name):
            self._input_owner = name
            return True
        return False

    def _release_input_owner(self, name: str | None) -> None:
        if name is None or self._input_owner == name:
            self._input_owner = None
            if self.input_controller is not None:
                self.input_controller.release_all()


__all__ = ["TaskExecutor"]
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from betternte.core.models import ROI

if TYPE_CHECKING:
    from betternte.core.capture import ScreenCapture
    from betternte.core.input import InputController
    from betternte.core.task_executor import TaskExecutor


class BaseTask(ABC):
    """所有自动化任务的基类。"""

    name: str = ""
    display_name: str = ""
    description: str = ""
    default_config: dict[str, Any] = {}
    config_type: dict[str, dict[str, Any]] = {}
    requires_input: bool = False

    # 运行时由 TaskExecutor 注入
    capture: ScreenCapture | None = None
    input_ctrl: InputController | None = None
    executor: TaskExecutor | None = None
    latest_source_roi: ROI | None = None

    def bind_runtime(
        self,
        *,
        capture: ScreenCapture | None = None,
        input_ctrl: InputController | None = None,
        executor: TaskExecutor | None = None,
    ) -> None:
        if capture is not None:
            self.capture = capture
        if input_ctrl is not None:
            self.input_ctrl = input_ctrl
        if executor is not None:
            self.executor = executor

    @abstractmethod
    def start(self) -> None:
        """启动任务。"""

    @abstractmethod
    def stop(self, reason: str = "") -> None:
        """停止任务。"""

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """任务是否正在运行。"""


class TriggerTask(BaseTask):
    """持续轮询的后台任务。框架按 interval 反复调用 run()。"""

    trigger_interval: float = 0.1

    @abstractmethod
    def run(self, frame: np.ndarray) -> None:
        """每帧调用，frame 为最新截图。"""


class OneTimeTask(BaseTask):
    """用户手动触发的流程任务，跑完一次就结束。"""

    @abstractmethod
    def run(self) -> None:
        """执行一次性任务流程。"""


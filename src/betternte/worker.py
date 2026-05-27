"""向后兼容的 re-export 层。旧代码 from betternte.worker import VisionWorker 仍可用。"""
from betternte.tasks.fishing.worker import VisionWorker

__all__ = ["VisionWorker"]

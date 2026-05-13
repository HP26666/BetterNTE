"""向后兼容的 re-export 层。旧代码 from betternte.state_machine import X 仍可用。"""
from betternte.tasks.fishing.state_machine import FishingStateMachine

__all__ = ["FishingStateMachine"]

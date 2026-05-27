"""向后兼容的 re-export 层。旧代码 from betternte.control import X 仍可用。"""
from betternte.core.input import InputController, PostMessageController, is_admin, admin_status_text
from betternte.tasks.fishing.controller import compute_ad_pulse

__all__ = [
    "InputController",
    "PostMessageController",
    "is_admin",
    "admin_status_text",
    "compute_ad_pulse",
]

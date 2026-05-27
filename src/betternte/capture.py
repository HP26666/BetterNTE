"""向后兼容的 re-export 层。旧代码 from betternte.capture import ScreenCapture 仍可用。"""
from betternte.core.capture import ScreenCapture, find_game_window, list_game_windows, grab_window_bitblt

__all__ = ["ScreenCapture", "find_game_window", "list_game_windows", "grab_window_bitblt"]

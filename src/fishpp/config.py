from __future__ import annotations

import json
from pathlib import Path

from fishpp.models import AppConfig

DEFAULT_CONFIG_PATH = Path("config.json")
CONFIGS_DIR = Path("configs")


def ensure_configs_dir(base_path: Path) -> Path:
    configs = base_path / CONFIGS_DIR
    configs.mkdir(parents=True, exist_ok=True)
    return configs


def list_configs(base_path: Path) -> list[Path]:
    configs = ensure_configs_dir(base_path)
    return sorted(configs.glob("*.json"))


def load_config(config_path: Path, screen_size: tuple[int, int]) -> AppConfig:
    if not config_path.exists():
        return AppConfig.from_dict({}, screen_size)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return AppConfig.from_dict(payload, screen_size)


def save_config(config: AppConfig, config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

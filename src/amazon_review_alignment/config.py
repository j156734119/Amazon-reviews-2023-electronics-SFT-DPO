from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config = deepcopy(config)
    config["_config_path"] = str(config_path)
    return config


def output_root(config: dict[str, Any]) -> Path:
    return Path(config["project"]["output_dir"]).resolve()


def resolve_output_path(config: dict[str, Any], *parts: str) -> Path:
    path = output_root(config).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

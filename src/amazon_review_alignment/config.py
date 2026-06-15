from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_config(path: Path, chain: tuple[Path, ...]) -> dict[str, Any]:
    config_path = path.resolve()
    if config_path in chain:
        cycle = " -> ".join(str(item) for item in (*chain, config_path))
        raise ValueError(f"Configuration inheritance cycle detected: {cycle}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    parent = config.pop("extends", None)
    if parent is not None:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        config = _deep_merge(
            _load_config(parent_path, (*chain, config_path)),
            config,
        )
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = _load_config(config_path, ())
    config["_config_path"] = str(config_path)
    return config


def output_root(config: dict[str, Any]) -> Path:
    return Path(config["project"]["output_dir"]).resolve()

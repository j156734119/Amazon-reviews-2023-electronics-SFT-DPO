from pathlib import Path

import pytest

from amazon_review_alignment.config import load_config


def test_config_inheritance_deep_merges_relative_parent(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    parent.write_text(
        "project:\n  seed: 42\n  output_dir: outputs\n"
        "training:\n  ppo:\n    batch_size: 1\n    epochs: 4\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "extends: parent.yaml\ntraining:\n  ppo:\n    epochs: 1\n",
        encoding="utf-8",
    )

    config = load_config(child)

    assert config["project"]["seed"] == 42
    assert config["training"]["ppo"]["batch_size"] == 1
    assert config["training"]["ppo"]["epochs"] == 1
    assert config["_config_path"] == str(child.resolve())


def test_config_inheritance_rejects_cycles(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\nvalue: 1\n", encoding="utf-8")
    second.write_text("extends: first.yaml\nvalue: 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cycle"):
        load_config(first)

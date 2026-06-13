from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

REPO_URL = "https://github.com/j156734119/Amazon-reviews-2023-electronics-SFT-DPO.git"


def _source(text: str) -> list[str]:
    return (textwrap.dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def markdown(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def common_setup_cells(gpu_name: str, config_path: str) -> list[dict[str, Any]]:
    return [
        markdown(
            f"""
            ## 1. 挂载 Drive 并加载项目

            本 Notebook 将仓库放在 Google Drive，训练输出和 checkpoint 会在断开
            Colab 后保留。目标 GPU：`{gpu_name}`。

            开始前必须先将本地最新代码和本 Notebook 提交并推送到 GitHub `main`
            分支，否则下方 `git clone` 会获取旧版本。
            """
        ),
        code(
            f"""
            import os
            import subprocess
            from pathlib import Path

            from google.colab import drive

            drive.mount("/content/drive")

            REPO_URL = "{REPO_URL}"
            REPO_DIR = Path(
                "/content/drive/MyDrive/amazon-review-alignment-workspace/repo"
            )
            REPO_DIR.parent.mkdir(parents=True, exist_ok=True)

            if (REPO_DIR / ".git").exists():
                subprocess.run(
                    ["git", "-C", str(REPO_DIR), "pull", "--ff-only"],
                    check=True,
                )
            else:
                subprocess.run(
                    ["git", "clone", REPO_URL, str(REPO_DIR)],
                    check=True,
                )

            os.chdir(REPO_DIR)
            print("Repository:", REPO_DIR)
            subprocess.run(["git", "log", "-1", "--oneline"], check=True)
            """
        ),
        markdown(
            """
            ## 2. 安装依赖

            执行后使用 Colab 菜单 **运行时 -> 重新启动会话**。重启后从下一单元格
            继续，不需要再次执行安装。
            """
        ),
        code(
            """
            import subprocess
            import sys

            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                check=False,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "--upgrade",
                    "pip",
                    "setuptools",
                    "wheel",
                ],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "-e", ".[train,eval,dev]"],
                check=True,
            )
            print("Installation complete. Restart the Colab runtime now.")
            """
        ),
        markdown(
            """
            ## 3. 重启后恢复目录、加载 Secrets

            在 Colab 左侧钥匙图标中添加 `OPENAI_API_KEY`。`HF_TOKEN` 对公开模型
            是可选的，但能提高 Hugging Face 下载限额。
            """
        ),
        code(
            f"""
            import os
            import subprocess
            import sys
            from pathlib import Path

            from google.colab import drive, userdata

            drive.mount("/content/drive", force_remount=False)
            REPO_DIR = Path(
                "/content/drive/MyDrive/amazon-review-alignment-workspace/repo"
            )
            os.chdir(REPO_DIR)
            CONFIG = "{config_path}"

            for secret_name in ("OPENAI_API_KEY", "HF_TOKEN"):
                try:
                    value = userdata.get(secret_name)
                except Exception:
                    value = None
                if value:
                    os.environ[secret_name] = value

            def cli(*arguments: str, check: bool = True) -> subprocess.CompletedProcess:
                command = [
                    sys.executable,
                    "-m",
                    "amazon_review_alignment.cli",
                    *arguments,
                ]
                print("\\n$", " ".join(command))
                return subprocess.run(command, cwd=REPO_DIR, check=check, text=True)

            print("Config:", CONFIG)
            print("OpenAI key loaded:", bool(os.getenv("OPENAI_API_KEY")))
            print("HF token loaded:", bool(os.getenv("HF_TOKEN")))
            """
        ),
    ]


def environment_cell(expected_gpu: str, minimum_gib: int, require_bf16: bool) -> dict[str, Any]:
    return code(
        f"""
        import importlib.metadata

        import torch
        import transformers
        import trl

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select a Colab GPU runtime.")

        gpu_name = torch.cuda.get_device_name(0)
        total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print("GPU:", gpu_name)
        print(f"VRAM: {{total_gib:.2f}} GiB")
        print("PyTorch:", torch.__version__)
        print("Transformers:", transformers.__version__)
        print("TRL:", trl.__version__)
        print("PEFT:", importlib.metadata.version("peft"))

        if "{expected_gpu}".lower() not in gpu_name.lower():
            raise RuntimeError("Expected {expected_gpu}, but Colab assigned: " + gpu_name)
        if total_gib < {minimum_gib}:
            raise RuntimeError("Insufficient GPU memory for this profile.")
        if {require_bf16!r} and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This profile requires BF16 support.")
        """
    )


def test_data_baseline_cells() -> list[dict[str, Any]]:
    return [
        markdown("## 4. 测试、准备数据并生成 Base baseline"),
        code(
            """
            subprocess.run(
                [sys.executable, "-m", "pytest"],
                cwd=REPO_DIR,
                check=True,
            )
            cli("prepare-data", "--config", CONFIG)
            """
        ),
        code(
            """
            cli(
                "evaluate",
                "--config",
                CONFIG,
                "--variants",
                "base",
                "--force-inference",
            )

            import pandas as pd

            output_root = (
                Path("outputs/smoke")
                if "smoke" in CONFIG
                else Path("outputs/a100-qwen3.5-2b")
            )
            display(pd.read_csv(output_root / "evaluation" / "metrics.csv"))
            """
        ),
    ]


def teacher_cells() -> list[dict[str, Any]]:
    return [
        markdown(
            """
            ## 5. 教师数据

            Pilot 会立即调用 OpenAI API。Batch 提交后可能需要等待；提交成功后可以
            关闭 GPU Runtime，稍后重新连接并重复“检查 Batch”单元格。
            """
        ),
        code(
            """
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("Add OPENAI_API_KEY to Colab Secrets first.")
            cli("teacher-pilot", "--config", CONFIG)
            """
        ),
        code(
            """
            # 第一次执行会提交 Batch；后续重复执行会查询并下载结果。
            cli("teacher-batch", "--config", CONFIG)
            """
        ),
        code(
            """
            output_root = (
                Path("outputs/smoke")
                if "smoke" in CONFIG
                else Path("outputs/a100-qwen3.5-2b")
            )
            train_preferences = output_root / "teacher" / "preferences_train.jsonl"
            validation_preferences = (
                output_root / "teacher" / "preferences_validation.jsonl"
            )
            if not train_preferences.exists() or not validation_preferences.exists():
                raise RuntimeError(
                    "Batch is not complete. Re-run the previous cell later; "
                    "do not start training yet."
                )
            print("Teacher train rows:", sum(1 for _ in train_preferences.open()))
            print(
                "Teacher validation rows:",
                sum(1 for _ in validation_preferences.open()),
            )
            """
        ),
    ]


def sft_dpo_cells() -> list[dict[str, Any]]:
    return [
        markdown("## 6. SFT、合并权重与 DPO"),
        code('cli("train-sft", "--config", CONFIG)'),
        code('cli("merge-sft", "--config", CONFIG)'),
        code('cli("train-dpo", "--config", CONFIG)'),
        code(
            """
            cli(
                "evaluate",
                "--config",
                CONFIG,
                "--variants",
                "base",
                "sft",
                "dpo",
                "--force-inference",
            )
            display(pd.read_csv(output_root / "evaluation" / "metrics.csv"))
            """
        ),
    ]


def smoke_human_cells() -> list[dict[str, Any]]:
    return [
        markdown(
            """
            ## 7. Smoke 人工偏好校准

            对 8 组回答逐条输入 `A`、`B` 或 `tie`。至少保留两条非 tie 记录。
            """
        ),
        code(
            """
            cli("prepare-rm-human-eval", "--config", CONFIG, "--samples", "8")

            import pandas as pd

            responses_path = output_root / "rlhf" / "rm_human_responses.csv"
            frame = pd.read_csv(responses_path, keep_default_na=False)

            for index, row in frame.iterrows():
                print("\\n" + "=" * 80)
                print("Review:", row["review_text"])
                print("\\nA:", row["response_a"])
                print("\\nB:", row["response_b"])
                while True:
                    choice = input("选择 A / B / tie: ").strip().lower()
                    if choice in {"a", "b", "tie"}:
                        break
                frame.at[index, "choice"] = choice

            frame.to_csv(responses_path, index=False)
            print("Saved:", responses_path)
            """
        ),
        code(
            """
            cli(
                "build-rlhf-data",
                "--config",
                CONFIG,
                "--responses",
                str(output_root / "rlhf" / "rm_human_responses.csv"),
            )
            """
        ),
    ]


def a100_human_cells() -> list[dict[str, Any]]:
    return [
        markdown(
            """
            ## 7. A100 正式人工偏好校准

            生成 200 条盲评 CSV。下载后填写 `choice` 列，只允许 `A`、`B`、`tie`，
            再上传回原路径。不要查看 `rm_human_key.jsonl`。
            """
        ),
        code(
            """
            cli("prepare-rm-human-eval", "--config", CONFIG, "--samples", "200")

            from google.colab import files

            responses_path = output_root / "rlhf" / "rm_human_responses.csv"
            print("Fill the choice column, then upload the completed CSV.")
            files.download(str(responses_path))
            """
        ),
        code(
            """
            import shutil

            from google.colab import files

            uploaded = files.upload()
            if len(uploaded) != 1:
                raise ValueError("Upload exactly one completed CSV.")
            uploaded_name = next(iter(uploaded))
            shutil.copyfile(uploaded_name, responses_path)
            print("Restored:", responses_path)
            """
        ),
        code(
            """
            cli(
                "build-rlhf-data",
                "--config",
                CONFIG,
                "--responses",
                str(responses_path),
            )
            """
        ),
    ]


def online_training_cells() -> list[dict[str, Any]]:
    return [
        markdown(
            """
            ## 8. Reward Model、PPO 与 GRPO

            每个阶段是独立单元格。阶段失败时先处理报错，不要跳过并继续。
            """
        ),
        code('cli("train-reward", "--config", CONFIG)'),
        code('cli("train-ppo", "--config", CONFIG)'),
        code('cli("train-grpo", "--config", CONFIG)'),
        markdown("## 9. 五模型统一评估和报告"),
        code(
            """
            cli(
                "evaluate",
                "--config",
                CONFIG,
                "--variants",
                "base",
                "sft",
                "dpo",
                "ppo",
                "grpo",
                "--force-inference",
            )
            cli("build-report", "--config", CONFIG)

            metrics_path = output_root / "evaluation" / "metrics.csv"
            report_path = output_root / "report.md"
            display(pd.read_csv(metrics_path))
            print("Report:", report_path.resolve())
            """
        ),
    ]


def notebook(cells: list[dict[str, Any]], gpu_type: str) -> dict[str, Any]:
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": gpu_type, "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_t4() -> dict[str, Any]:
    cells = [
        markdown(
            """
            # Amazon Review Alignment: T4 Smoke

            使用 `Qwen/Qwen3-0.6B` 在 Colab T4 上执行：

            `Base -> SFT -> DPO -> Reward Model -> PPO -> GRPO -> Evaluation`

            所有训练阶段都是极小规模 smoke，只验证链路与产物，不代表收敛。
            """
        ),
        *common_setup_cells("T4", "configs/rlhf_smoke.yaml"),
        markdown("## T4 环境检查"),
        environment_cell("T4", 14, False),
        *test_data_baseline_cells(),
        *teacher_cells(),
        *sft_dpo_cells(),
        *smoke_human_cells(),
        *online_training_cells(),
    ]
    return notebook(cells, "T4")


def build_a100() -> dict[str, Any]:
    cells = [
        markdown(
            """
            # Amazon Review Alignment: A100 Formal Pipeline

            使用 `Qwen/Qwen3.5-2B`、BF16 与 4-bit QLoRA 执行正式的五模型实验。
            该配置控制了 PPO/GRPO prompt 数与评估规模，但 Colab Compute Unit
            消耗是动态的，不能保证固定在 100 CU 内。
            """
        ),
        *common_setup_cells("A100", "configs/rlhf_a100.yaml"),
        markdown("## A100 环境检查"),
        environment_cell("A100", 38, True),
        *test_data_baseline_cells(),
        *teacher_cells(),
        *sft_dpo_cells(),
        *a100_human_cells(),
        *online_training_cells(),
    ]
    return notebook(cells, "A100")


def write_notebook(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    outputs = {
        root / "Amazon_Review_Alignment_T4_Smoke.ipynb": build_t4(),
        root / "Amazon_Review_Alignment_A100.ipynb": build_a100(),
    }
    for path, value in outputs.items():
        write_notebook(path, value)
        print(path)


if __name__ == "__main__":
    main()

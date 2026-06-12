from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any


def _source(value: str) -> list[str]:
    text = textwrap.dedent(value).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def _markdown(value: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source(value),
    }


def _code(value: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(value),
    }


def build_notebook() -> dict[str, Any]:
    cells = [
        _markdown(
            """
            # Amazon Review Alignment: SFT, DPO, PPO, and GRPO

            这是一个独立、可在 Google Colab 上运行的对齐实验。
            它不依赖原 Python 工程，默认用 32 条离线合成评论完成：

            `Base -> SFT -> DPO / PPO / GRPO -> 五模型统一评估`

            Smoke 模式只用于验证完整链路，不代表模型收敛或性能提升。
            PPO/GRPO 的偏好主要来自 AI/规则，因此属于 RLAIF；只有导入人工
            A/B/tie 校准后才称为 human-calibrated RLAIF。
            """
        ),
        _code(
            """
            import subprocess
            import sys

            # Colab currently preinstalls an old torchao build that is incompatible
            # with recent PEFT. This project does not use torchao quantization.
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                check=False,
            )

            PACKAGES = [
                "trl==1.5.1",
                "transformers==5.5.4",
                "accelerate>=1.4",
                "peft>=0.18",
                "bitsandbytes>=0.45",
                "datasets>=3.2",
                "openai>=2.0",
                "pydantic>=2.8",
                "pandas>=2.2",
                "matplotlib>=3.8",
                "pillow>=10",
            ]
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", *PACKAGES]
            )
            print("Dependencies installed. Restart the runtime only if Colab requests it.")
            """
        ),
        _markdown(
            """
            ## 1. 全局配置与 Google Drive

            - `PROFILE="smoke"`：32 条离线数据，所有训练阶段只跑极少更新。
            - `PROFILE="a100"`：使用 Qwen3.5-2B 与 BF16 的正式实验。
            - `TEACHER_MODE="offline"`：不需要 API key。
            - OpenAI 模式从 Colab Secrets 的 `OPENAI_API_KEY` 读取密钥。
            """
        ),
        _code(
            """
            import gc
            import hashlib
            import json
            import os
            import random
            import re
            import time
            from pathlib import Path

            import numpy as np
            import pandas as pd
            import torch

            PROFILE = "smoke"  # smoke | a100
            DATA_MODE = "offline"  # offline | amazon
            TEACHER_MODE = "offline"  # offline | openai | upload
            MOUNT_DRIVE = True
            USE_HUMAN_CALIBRATION = False
            SEED = 42

            if MOUNT_DRIVE:
                try:
                    from google.colab import drive

                    drive.mount("/content/drive")
                    OUTPUT_ROOT = Path(
                        "/content/drive/MyDrive/amazon-review-alignment-colab"
                    )
                except ImportError:
                    OUTPUT_ROOT = Path("outputs/colab")
            else:
                OUTPUT_ROOT = Path("/content/amazon-review-alignment-colab")
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            DRIVE_OPENAI_KEY_FILE = (
                OUTPUT_ROOT / ".secrets" / "openai_api_key.txt"
            )

            try:
                from google.colab import userdata
                from huggingface_hub import login

                hf_token = userdata.get("HF_TOKEN")
                if hf_token:
                    login(token=hf_token, add_to_git_credential=False)
                    print("Authenticated with Hugging Face Hub.")
                else:
                    print("HF_TOKEN is optional for the public Qwen model.")
            except Exception:
                print("HF_TOKEN is optional for the public Qwen model.")

            PROFILE_CONFIG = {
                "smoke": {
                    "base_model": "Qwen/Qwen3-0.6B",
                    "run_name": "smoke-qwen3-0.6b",
                    "max_length": 256,
                    "max_new_tokens": 96,
                    "sft_steps": 1,
                    "dpo_steps": 1,
                    "rm_steps": 1,
                    "ppo_episodes": 8,
                    "grpo_steps": 1,
                    "grpo_generations": 2,
                    "grpo_generation_batch": 2,
                    "lora_r": 4,
                    "lora_targets": ["q_proj", "v_proj"],
                    "online_lora_targets": ["q_proj", "v_proj"],
                },
                "a100": {
                    "base_model": "Qwen/Qwen3.5-2B",
                    "run_name": "a100-qwen3.5-2b",
                    "max_length": 768,
                    "max_new_tokens": 128,
                    "sft_steps": -1,
                    "dpo_steps": -1,
                    "rm_steps": -1,
                    "ppo_episodes": 128,
                    "grpo_steps": -1,
                    "grpo_generations": 4,
                    "grpo_generation_batch": 4,
                    "lora_r": 16,
                    "lora_targets": [
                        "in_proj_qkv",
                        "in_proj_z",
                        "out_proj",
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                    "online_lora_targets": [
                        "in_proj_qkv",
                        "out_proj",
                        "q_proj",
                        "v_proj",
                    ],
                },
            }
            CFG = PROFILE_CONFIG[PROFILE]
            BASE_MODEL = CFG["base_model"]
            RUN_ROOT = OUTPUT_ROOT / "runs" / CFG["run_name"]
            RUN_ROOT.mkdir(parents=True, exist_ok=True)
            USE_BF16 = PROFILE == "a100" and torch.cuda.is_bf16_supported()
            USE_FP16 = not USE_BF16
            COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

            def set_seed(seed=SEED):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            def release_cuda(*objects):
                for obj in objects:
                    del obj
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            def upcast_trainable_params(model):
                if USE_BF16:
                    count = sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    )
                    print(f"BF16 trainable parameters retained: {count:,}")
                    return count
                count = 0
                for parameter in model.parameters():
                    if not parameter.requires_grad:
                        continue
                    count += parameter.numel()
                    if parameter.dtype != torch.float32:
                        parameter.data = parameter.data.to(torch.float32)
                print(f"Trainable parameters kept in FP32: {count:,}")
                return count

            def latest_checkpoint(output_dir):
                output_dir = Path(output_dir)
                checkpoints = sorted(
                    output_dir.glob("checkpoint-*"),
                    key=lambda path: int(path.name.rsplit("-", 1)[-1]),
                )
                return str(checkpoints[-1]) if checkpoints else None

            def save_history(name, trainer):
                metrics_dir = RUN_ROOT / "metrics"
                metrics_dir.mkdir(parents=True, exist_ok=True)
                (metrics_dir / f"{name}_log.json").write_text(
                    json.dumps(trainer.state.log_history, indent=2),
                    encoding="utf-8",
                )

            set_seed()
            print("Persistent root:", OUTPUT_ROOT)
            print("Current run:", RUN_ROOT)
            print("Profile/model:", PROFILE, BASE_MODEL)
            """
        ),
        _code(
            """
            if not torch.cuda.is_available():
                raise RuntimeError("Select Runtime > Change runtime type > GPU.")
            properties = torch.cuda.get_device_properties(0)
            total_gib = properties.total_memory / 1024**3
            free_bytes, _ = torch.cuda.mem_get_info()
            print("GPU:", properties.name)
            print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
            print(f"VRAM total/free: {total_gib:.2f}/{free_bytes / 1024**3:.2f} GiB")
            required_gib = 38 if PROFILE == "a100" else 14
            if total_gib < required_gib:
                raise RuntimeError(
                    f"{PROFILE} profile requires at least {required_gib} GiB VRAM."
                )
            if PROFILE == "a100" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("The A100 profile requires CUDA BF16 support.")
            """
        ),
        _markdown(
            """
            ## 2. Schema、提示词与模型工具

            Qwen3 thinking mode 被关闭。SFT 采用 prompt/completion 数据，只对
            completion 计算 loss。
            """
        ),
        _code(
            """
            from peft import LoraConfig, PeftModel, TaskType
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSequenceClassification,
                AutoTokenizer,
                BitsAndBytesConfig,
            )

            SYSTEM_PROMPT = '''Analyze the Amazon review and return only JSON:
            {"sentiment":"negative|neutral|positive",
             "evidence":["1-2 exact spans copied from the review"],
             "analysis":"faithful analysis of at most 80 words"}
            Do not use star ratings or unsupported facts.'''

            def user_prompt(text):
                return f"Review:\\n{text}\\nReturn the required JSON."

            def stable_id(text):
                return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

            def normalize(text):
                return re.sub(r"\\s+", " ", str(text)).strip().casefold()

            def grounded(review, span):
                return normalize(span) in normalize(review)

            def raw_schema_payload(raw):
                try:
                    payload = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    return None
                if not isinstance(payload, dict):
                    return None
                if set(payload) != {"sentiment", "evidence", "analysis"}:
                    return None
                evidence = payload["evidence"]
                valid = (
                    payload["sentiment"] in {"negative", "neutral", "positive"}
                    and isinstance(evidence, list)
                    and 1 <= len(evidence) <= 2
                    and all(isinstance(x, str) and x.strip() for x in evidence)
                    and isinstance(payload["analysis"], str)
                    and payload["analysis"].strip()
                )
                return payload if valid else None

            def render_chat(tokenizer, messages, add_generation_prompt):
                kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": add_generation_prompt,
                    "enable_thinking": False,
                }
                try:
                    return tokenizer.apply_chat_template(messages, **kwargs)
                except TypeError:
                    kwargs.pop("enable_thinking")
                    return tokenizer.apply_chat_template(messages, **kwargs)

            def analysis_prompt(tokenizer, text):
                return render_chat(
                    tokenizer,
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt(text)},
                    ],
                    True,
                )

            def tokenizer_for(model_path=BASE_MODEL, padding_side="right"):
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True
                )
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = padding_side
                return tokenizer

            def quantization():
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
                    bnb_4bit_use_double_quant=True,
                )

            def causal_model(model_path=BASE_MODEL, train=True):
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    quantization_config=quantization(),
                    device_map="auto",
                )
                model.config.use_cache = not train
                return model

            def causal_lora(r=None, online=False):
                rank = int(r or CFG["lora_r"])
                return LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=rank,
                    lora_alpha=rank * 2,
                    lora_dropout=0.05,
                    target_modules=(
                        CFG["online_lora_targets"]
                        if online
                        else CFG["lora_targets"]
                    ),
                    bias="none",
                )
            """
        ),
        _markdown(
            """
            ## 3. 离线数据与可选教师

            默认数据是 32 条原创合成评论，用于无密钥 smoke test。正式实验可将
            `TEACHER_MODE` 改成 `upload` 并上传与当前 schema 相同的 JSONL，或使用
            OpenAI Structured Outputs 生成小规模教师数据。
            """
        ),
        _code(
            """
            POSITIVE = [
                "The battery lasts all day.",
                "Setup took less than five minutes.",
                "The screen is bright and sharp.",
                "Sound quality is clear at normal volume.",
                "The cable feels durable.",
                "It connected immediately.",
                "The keyboard is comfortable to type on.",
                "Charging is fast and reliable.",
                "The case fits perfectly.",
                "The controls are simple to use.",
                "It works exactly as advertised.",
            ]
            NEUTRAL = [
                "The product works but the instructions are brief.",
                "Performance is acceptable for the price.",
                "The design is plain but functional.",
                "It arrived on time and works normally.",
                "Battery life is average.",
                "The buttons work but feel ordinary.",
                "The picture quality is adequate.",
                "Installation required one restart.",
                "The cable length is sufficient.",
                "It meets basic needs without extra features.",
            ]
            NEGATIVE = [
                "The device stopped working after two days.",
                "The battery drains within an hour.",
                "The connector is loose.",
                "Setup repeatedly failed.",
                "The screen flickers during use.",
                "The fan is extremely loud.",
                "It disconnects every few minutes.",
                "The case arrived cracked.",
                "Charging is slow and unreliable.",
                "The buttons frequently miss inputs.",
                "The advertised feature is missing.",
            ]

            def chosen_json(sentiment, evidence):
                analysis = {
                    "positive": "The reviewer reports a clearly favorable experience.",
                    "neutral": "The review is mixed or factual without strong praise or criticism.",
                    "negative": "The reviewer reports a concrete product problem.",
                }[sentiment]
                return json.dumps(
                    {
                        "sentiment": sentiment,
                        "evidence": [evidence],
                        "analysis": analysis,
                    },
                    ensure_ascii=False,
                )

            def rejected_json(sentiment, evidence):
                wrong = {
                    "positive": "negative",
                    "neutral": "positive",
                    "negative": "positive",
                }[sentiment]
                return json.dumps(
                    {
                        "sentiment": wrong,
                        "evidence": [evidence],
                        "analysis": "This response intentionally uses the wrong sentiment.",
                    },
                    ensure_ascii=False,
                )

            def offline_rows():
                rows = []
                for sentiment, phrases in [
                    ("positive", POSITIVE),
                    ("neutral", NEUTRAL),
                    ("negative", NEGATIVE),
                ]:
                    for phrase in phrases:
                        text = phrase + " This is my experience with the electronics item."
                        rows.append(
                            {
                                "id": stable_id(text),
                                "text": text,
                                "chosen": chosen_json(sentiment, phrase),
                                "rejected": rejected_json(sentiment, phrase),
                                "defect_type": "wrong_sentiment",
                            }
                        )
                return rows[:32]

            def amazon_review_rows(limit):
                import requests

                url = (
                    "https://huggingface.co/datasets/McAuley-Lab/"
                    "Amazon-Reviews-2023/resolve/main/raw/review_categories/"
                    "Electronics.jsonl"
                )
                rows = []
                seen_text = set()
                seen_products = set()
                base_quota, remainder = divmod(limit, 5)
                quotas = {
                    rating: base_quota + int(rating <= remainder)
                    for rating in range(1, 6)
                }
                counts = {rating: 0 for rating in range(1, 6)}
                with requests.get(url, stream=True, timeout=120) as response:
                    response.raise_for_status()
                    for raw_line in response.iter_lines(decode_unicode=True):
                        if not raw_line:
                            continue
                        item = json.loads(raw_line)
                        rating = int(item.get("rating", 0))
                        text = re.sub(r"\\s+", " ", str(item.get("text", ""))).strip()
                        parent_asin = str(item.get("parent_asin", "")).strip()
                        key = normalize(text)
                        if (
                            not text
                            or rating not in quotas
                            or counts[rating] >= quotas[rating]
                            or key in seen_text
                            or not parent_asin
                            or parent_asin in seen_products
                        ):
                            continue
                        seen_text.add(key)
                        seen_products.add(parent_asin)
                        counts[rating] += 1
                        rows.append(
                            {
                                "id": stable_id(parent_asin + "\\0" + text),
                                "text": text,
                            }
                        )
                        if len(rows) >= limit:
                            break
                return rows

            def write_jsonl(path, rows):
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\\n")

            def read_jsonl(path):
                if not Path(path).exists():
                    return []
                return [
                    json.loads(line)
                    for line in Path(path).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]

            def optional_openai_teacher(review_rows):
                from openai import OpenAI
                from typing import Literal

                from pydantic import BaseModel, Field

                class Analysis(BaseModel):
                    sentiment: Literal["negative", "neutral", "positive"]
                    evidence: list[str] = Field(min_length=1, max_length=2)
                    analysis: str

                class Preference(BaseModel):
                    chosen: Analysis
                    rejected: str
                    defect_type: Literal[
                        "wrong_sentiment",
                        "fabricated_evidence",
                        "over_inference",
                        "format_error",
                        "irrelevant_verbose",
                    ]

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    try:
                        from google.colab import userdata

                        api_key = userdata.get("OPENAI_API_KEY")
                    except Exception:
                        api_key = None
                if not api_key and DRIVE_OPENAI_KEY_FILE.exists():
                    api_key = DRIVE_OPENAI_KEY_FILE.read_text(
                        encoding="utf-8"
                    ).strip()
                if not api_key:
                    raise RuntimeError(
                        "OPENAI_API_KEY is missing from Colab Secrets and "
                        f"{DRIVE_OPENAI_KEY_FILE}."
                    )
                client = OpenAI(api_key=api_key)
                output = []
                for row in review_rows:
                    teacher_instruction = (
                        "Create a preference pair for the supplied Amazon review. "
                        "chosen must follow the analysis schema, quote exact evidence, "
                        "and remain faithful. rejected must be a plausible answer with "
                        "exactly one controlled defect. Return the requested structured "
                        "Preference object."
                    )
                    response = client.responses.parse(
                        model="gpt-5.4-mini-2026-03-17",
                        input=[
                            {"role": "system", "content": teacher_instruction},
                            {"role": "user", "content": user_prompt(row["text"])},
                        ],
                        text_format=Preference,
                    )
                    parsed = response.output_parsed
                    if parsed is None:
                        raise ValueError("Teacher returned no structured output.")
                    output.append(
                        {
                            **row,
                            "chosen": parsed.chosen.model_dump_json(),
                            "rejected": parsed.rejected,
                            "defect_type": parsed.defect_type,
                        }
                    )
                return output

            if TEACHER_MODE == "upload":
                uploaded = OUTPUT_ROOT / "uploaded_preferences.jsonl"
                rows = read_jsonl(uploaded)
                if not rows:
                    raise RuntimeError(f"Upload preference JSONL to {uploaded}")
            else:
                if PROFILE == "a100" and DATA_MODE == "offline":
                    raise ValueError(
                        "A100 profile requires DATA_MODE='amazon' with OpenAI "
                        "teacher or TEACHER_MODE='upload'."
                    )
                target_size = 32 if PROFILE == "smoke" else 5000
                rows = (
                    offline_rows()
                    if DATA_MODE == "offline"
                    else amazon_review_rows(target_size)
                )
                if TEACHER_MODE == "openai":
                    rows = optional_openai_teacher(rows)
                elif DATA_MODE == "amazon":
                    raise ValueError(
                        "Amazon rows require TEACHER_MODE='openai' or 'upload'."
                    )

            random.Random(SEED).shuffle(rows)
            if PROFILE == "smoke":
                train_rows, validation_rows, test_rows = (
                    rows[:20],
                    rows[20:24],
                    rows[24:32],
                )
                rm_rows = train_rows[:12]
                online_rows = train_rows[12:20]
            else:
                train_rows = rows[:3500]
                validation_rows = rows[3500:4000]
                test_rows = rows[4000:5000]
                rm_rows = train_rows[:800]
                online_rows = train_rows[800:1056]
            data_dir = RUN_ROOT / "data"
            write_jsonl(data_dir / "train.jsonl", train_rows)
            write_jsonl(data_dir / "validation.jsonl", validation_rows)
            write_jsonl(data_dir / "test.jsonl", test_rows)
            write_jsonl(data_dir / "rm_train.jsonl", rm_rows)
            write_jsonl(data_dir / "online_prompts.jsonl", online_rows)
            assert not {x["id"] for x in online_rows} & {x["id"] for x in test_rows}
            print(
                {
                    "train": len(train_rows),
                    "validation": len(validation_rows),
                    "test": len(test_rows),
                    "rm": len(rm_rows),
                    "ppo_grpo_shared_prompts": len(online_rows),
                }
            )
            """
        ),
        _markdown(
            """
            ## 4. 可选人工 A/B/tie 校准

            默认跳过并保持 RLAIF。启用后先导出 CSV，人工填写 `choice` 后重新运行
            本单元格；`tie` 会被丢弃，A/B 会修正 chosen/rejected 方向。
            """
        ),
        _code(
            """
            calibration_path = RUN_ROOT / "data" / "rm_human_calibration.csv"
            if USE_HUMAN_CALIBRATION:
                if not calibration_path.exists():
                    calibration = []
                    rng = random.Random(SEED)
                    calibration_count = 8 if PROFILE == "smoke" else 200
                    for row in rm_rows[: min(calibration_count, len(rm_rows))]:
                        swapped = bool(rng.getrandbits(1))
                        a = row["rejected"] if swapped else row["chosen"]
                        b = row["chosen"] if swapped else row["rejected"]
                        calibration.append(
                            {
                                "id": row["id"],
                                "review_text": row["text"],
                                "response_a": a,
                                "response_b": b,
                                "choice": "",
                            }
                        )
                    pd.DataFrame(calibration).to_csv(calibration_path, index=False)
                    raise RuntimeError(
                        f"Fill A/B/tie in {calibration_path}, then rerun this cell."
                    )
                frame = pd.read_csv(calibration_path, keep_default_na=False)
                choices = frame["choice"].str.strip().str.lower()
                if not choices.isin({"a", "b", "tie"}).all():
                    raise ValueError("Every choice must be A, B, or tie.")
                calibrated = []
                source = {row["id"]: row for row in rm_rows}
                for _, item in frame.iterrows():
                    if item["choice"].lower() == "tie":
                        continue
                    row = source[item["id"]]
                    choose_a = item["choice"].lower() == "a"
                    calibrated.append(
                        {
                            **row,
                            "chosen": item["response_a"] if choose_a else item["response_b"],
                            "rejected": item["response_b"] if choose_a else item["response_a"],
                        }
                    )
                rm_rows = calibrated + [
                    row for row in rm_rows if row["id"] not in set(frame["id"])
                ]
                print("Human-calibrated RM pairs:", len(calibrated))
            else:
                print("Human calibration disabled: experiment is reported as RLAIF.")
            """
        ),
        _markdown("## 5. SFT：completion-only QLoRA"),
        _code(
            """
            from datasets import Dataset
            from trl import SFTConfig, SFTTrainer

            tokenizer = tokenizer_for()
            def sft_records(items):
                return [
                    {
                        "prompt": analysis_prompt(tokenizer, row["text"]),
                        "completion": row["chosen"],
                    }
                    for row in items
                ]

            sft_dir = RUN_ROOT / "models" / "sft"
            sft_model = causal_model(train=True)
            sft_args = SFTConfig(
                output_dir=str(sft_dir),
                learning_rate=2e-4,
                num_train_epochs=1,
                max_steps=CFG["sft_steps"],
                per_device_train_batch_size=1 if PROFILE == "smoke" else 4,
                gradient_accumulation_steps=1 if PROFILE == "smoke" else 4,
                max_length=CFG["max_length"],
                completion_only_loss=True,
                gradient_checkpointing=True,
                fp16=USE_FP16,
                bf16=USE_BF16,
                logging_steps=1,
                save_steps=1 if PROFILE == "smoke" else 50,
                report_to="none",
                seed=SEED,
            )
            sft_trainer = SFTTrainer(
                model=sft_model,
                args=sft_args,
                train_dataset=Dataset.from_list(sft_records(train_rows)),
                eval_dataset=Dataset.from_list(sft_records(validation_rows)),
                processing_class=tokenizer,
                peft_config=causal_lora(),
            )
            upcast_trainable_params(sft_trainer.model)
            sft_trainer.train(resume_from_checkpoint=latest_checkpoint(sft_dir))
            sft_trainer.save_model(str(sft_dir))
            tokenizer.save_pretrained(sft_dir)
            save_history("sft", sft_trainer)
            release_cuda(sft_trainer, sft_model)
            """
        ),
        _markdown("## 6. 合并 SFT，并训练单主干 DPO"),
        _code(
            """
            merged_dir = RUN_ROOT / "models" / "sft-merged"
            merge_base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                trust_remote_code=True,
                dtype=COMPUTE_DTYPE,
                device_map="auto",
            )
            merged = PeftModel.from_pretrained(merge_base, str(sft_dir)).merge_and_unload()
            merged.save_pretrained(merged_dir, safe_serialization=True)
            tokenizer.save_pretrained(merged_dir)
            release_cuda(merged, merge_base)

            from trl import DPOConfig, DPOTrainer
            from peft import prepare_model_for_kbit_training

            dpo_dir = RUN_ROOT / "models" / "dpo"
            dpo_tokenizer = tokenizer_for(merged_dir, "left")
            dpo_model = prepare_model_for_kbit_training(
                causal_model(merged_dir, train=True),
                use_gradient_checkpointing=True,
            )
            dpo_data = [
                {
                    "prompt": analysis_prompt(dpo_tokenizer, row["text"]),
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                }
                for row in train_rows
            ]
            dpo_args = DPOConfig(
                output_dir=str(dpo_dir),
                learning_rate=5e-5,
                beta=0.1,
                num_train_epochs=1,
                max_steps=CFG["dpo_steps"],
                per_device_train_batch_size=1 if PROFILE == "smoke" else 2,
                gradient_accumulation_steps=1 if PROFILE == "smoke" else 8,
                max_length=CFG["max_length"],
                max_prompt_length=CFG["max_length"] // 2,
                gradient_checkpointing=True,
                fp16=USE_FP16,
                bf16=USE_BF16,
                logging_steps=1,
                save_steps=1 if PROFILE == "smoke" else 50,
                report_to="none",
                seed=SEED,
            )
            dpo_trainer = DPOTrainer(
                model=dpo_model,
                ref_model=None,
                args=dpo_args,
                train_dataset=Dataset.from_list(dpo_data),
                processing_class=dpo_tokenizer,
                peft_config=causal_lora(),
            )
            upcast_trainable_params(dpo_trainer.model)
            dpo_trainer.train(resume_from_checkpoint=latest_checkpoint(dpo_dir))
            dpo_trainer.save_model(str(dpo_dir))
            dpo_tokenizer.save_pretrained(dpo_dir)
            save_history("dpo", dpo_trainer)
            release_cuda(dpo_trainer, dpo_model)
            """
        ),
        _markdown("## 7. Reward Model"),
        _code(
            """
            from trl import RewardConfig, RewardTrainer

            rm_dir = RUN_ROOT / "models" / "reward"
            rm_tokenizer = tokenizer_for(merged_dir)
            rm_model = AutoModelForSequenceClassification.from_pretrained(
                merged_dir,
                num_labels=1,
                trust_remote_code=True,
                quantization_config=quantization(),
                device_map="auto",
            )
            rm_model.config.pad_token_id = rm_tokenizer.pad_token_id
            def reward_records(items):
                records = []
                for row in items:
                    prompt = analysis_prompt(rm_tokenizer, row["text"])
                    records.append(
                        {
                            "chosen": prompt + row["chosen"],
                            "rejected": prompt + row["rejected"],
                        }
                    )
                return records

            rm_args = RewardConfig(
                output_dir=str(rm_dir),
                learning_rate=1e-4,
                num_train_epochs=1,
                max_steps=CFG["rm_steps"],
                per_device_train_batch_size=1 if PROFILE == "smoke" else 4,
                gradient_accumulation_steps=1 if PROFILE == "smoke" else 4,
                max_length=CFG["max_length"],
                gradient_checkpointing=True,
                fp16=USE_FP16,
                bf16=USE_BF16,
                logging_steps=1,
                save_steps=1 if PROFILE == "smoke" else 50,
                report_to="none",
                seed=SEED,
            )
            rm_lora = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=CFG["lora_r"],
                lora_alpha=CFG["lora_r"] * 2,
                lora_dropout=0.05,
                target_modules=CFG["online_lora_targets"],
                modules_to_save=["score"],
                bias="none",
            )
            rm_trainer = RewardTrainer(
                model=rm_model,
                args=rm_args,
                train_dataset=Dataset.from_list(reward_records(rm_rows)),
                eval_dataset=Dataset.from_list(reward_records(validation_rows)),
                processing_class=rm_tokenizer,
                peft_config=rm_lora,
            )
            upcast_trainable_params(rm_trainer.model)
            rm_trainer.train(resume_from_checkpoint=latest_checkpoint(rm_dir))
            rm_trainer.save_model(str(rm_dir))
            rm_tokenizer.save_pretrained(rm_dir)
            save_history("reward", rm_trainer)
            release_cuda(rm_trainer, rm_model)
            """
        ),
        _markdown(
            """
            ## 8. PPO smoke

            PPO 同时装配 policy、reference、reward 和 value。reference 通过禁用
            PPO adapter 获得；该实验只证明流程可运行。
            """
        ),
        _code(
            """
            from trl.experimental.ppo import PPOConfig, PPOTrainer

            def load_reward_adapter(trainable=False):
                model = AutoModelForSequenceClassification.from_pretrained(
                    merged_dir,
                    num_labels=1,
                    trust_remote_code=True,
                    quantization_config=quantization(),
                    device_map="auto",
                )
                model.config.pad_token_id = rm_tokenizer.pad_token_id
                if trainable:
                    model = prepare_model_for_kbit_training(
                        model, use_gradient_checkpointing=True
                    )
                model = PeftModel.from_pretrained(
                    model, str(rm_dir), is_trainable=trainable
                )
                if not trainable:
                    for parameter in model.parameters():
                        parameter.requires_grad = False
                return model

            ppo_dir = RUN_ROOT / "models" / "ppo"
            ppo_tokenizer = tokenizer_for(merged_dir, "left")
            ppo_policy = prepare_model_for_kbit_training(
                causal_model(merged_dir, train=True),
                use_gradient_checkpointing=True,
            )
            reward_model = load_reward_adapter(False)
            value_model = load_reward_adapter(True)
            ppo_dataset = Dataset.from_list(
                [
                    {
                        "input_ids": ppo_tokenizer(
                            analysis_prompt(ppo_tokenizer, row["text"])
                        )["input_ids"][-192:]
                    }
                    for row in online_rows
                ]
            )
            ppo_args = PPOConfig(
                output_dir=str(ppo_dir),
                total_episodes=CFG["ppo_episodes"],
                response_length=64 if PROFILE == "smoke" else 128,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=2 if PROFILE == "smoke" else 8,
                local_rollout_forward_batch_size=1,
                num_mini_batches=1,
                num_ppo_epochs=1,
                learning_rate=3e-6,
                temperature=0.7,
                kl_coef=0.05,
                cliprange=0.2,
                missing_eos_penalty=1.0,
                gradient_checkpointing=True,
                fp16=USE_FP16,
                bf16=USE_BF16,
                report_to="none",
                logging_steps=1,
                save_steps=1 if PROFILE == "smoke" else 8,
                seed=SEED,
            )
            ppo_trainer = PPOTrainer(
                args=ppo_args,
                processing_class=ppo_tokenizer,
                model=ppo_policy,
                ref_model=None,
                reward_model=reward_model,
                value_model=value_model,
                train_dataset=ppo_dataset,
                eval_dataset=ppo_dataset.select(range(1)),
                peft_config=causal_lora(online=True),
            )
            upcast_trainable_params(ppo_trainer.model)
            torch.cuda.reset_peak_memory_stats()
            ppo_trainer.train(resume_from_checkpoint=latest_checkpoint(ppo_dir))
            ppo_peak = torch.cuda.max_memory_reserved() / 1024**3
            ppo_trainer.save_model(str(ppo_dir))
            ppo_tokenizer.save_pretrained(ppo_dir)
            save_history("ppo", ppo_trainer)
            print(f"PPO peak reserved VRAM: {ppo_peak:.2f} GiB")
            release_cuda(ppo_trainer, ppo_policy, reward_model, value_model)
            """
        ),
        _markdown(
            """
            ## 9. GRPO：Reward Model + 可验证规则奖励

            每个 prompt 生成多个候选，奖励权重为 RM `1.0`、schema `1.0`、
            evidence `1.5`、长度 `0.25`。
            """
        ),
        _code(
            """
            from trl import GRPOConfig, GRPOTrainer

            def schema_reward(completions, **kwargs):
                return [1.0 if raw_schema_payload(x) else 0.0 for x in completions]

            def evidence_reward(completions, text, **kwargs):
                rewards = []
                for completion, review in zip(completions, text):
                    payload = raw_schema_payload(completion)
                    ok = payload is not None and all(
                        grounded(review, span) for span in payload["evidence"]
                    )
                    rewards.append(1.0 if ok else 0.0)
                return rewards

            def length_reward(completions, **kwargs):
                rewards = []
                for completion in completions:
                    payload = raw_schema_payload(completion)
                    ok = payload is not None and len(payload["analysis"].split()) <= 80
                    rewards.append(1.0 if ok else 0.0)
                return rewards

            grpo_dir = RUN_ROOT / "models" / "grpo"
            grpo_tokenizer = tokenizer_for(merged_dir, "left")
            grpo_policy = prepare_model_for_kbit_training(
                causal_model(merged_dir, train=True),
                use_gradient_checkpointing=True,
            )
            grpo_reward_model = load_reward_adapter(False)
            grpo_dataset = Dataset.from_list(
                [
                    {
                        "id": row["id"],
                        "text": row["text"],
                        "prompt": analysis_prompt(grpo_tokenizer, row["text"]),
                    }
                    for row in online_rows
                ]
            )
            grpo_args = GRPOConfig(
                output_dir=str(grpo_dir),
                learning_rate=3e-6,
                num_train_epochs=1,
                max_steps=CFG["grpo_steps"],
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1 if PROFILE == "smoke" else 4,
                generation_batch_size=CFG["grpo_generation_batch"],
                num_generations=CFG["grpo_generations"],
                num_iterations=1,
                max_prompt_length=192 if PROFILE == "smoke" else 384,
                max_completion_length=64 if PROFILE == "smoke" else 128,
                temperature=0.7,
                beta=0.02,
                epsilon=0.2,
                loss_type="dapo",
                scale_rewards="group",
                reward_weights=[1.0, 1.0, 1.5, 0.25],
                use_vllm=False,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                fp16=USE_FP16,
                bf16=USE_BF16,
                logging_steps=1,
                save_steps=1 if PROFILE == "smoke" else 8,
                report_to="none",
                seed=SEED,
            )
            grpo_trainer = GRPOTrainer(
                model=grpo_policy,
                args=grpo_args,
                reward_funcs=[
                    grpo_reward_model,
                    schema_reward,
                    evidence_reward,
                    length_reward,
                ],
                reward_processing_classes=[
                    grpo_tokenizer,
                    None,
                    None,
                    None,
                ],
                train_dataset=grpo_dataset,
                processing_class=grpo_tokenizer,
                peft_config=causal_lora(online=True),
            )
            upcast_trainable_params(grpo_trainer.model)
            torch.cuda.reset_peak_memory_stats()
            grpo_trainer.train(resume_from_checkpoint=latest_checkpoint(grpo_dir))
            grpo_peak = torch.cuda.max_memory_reserved() / 1024**3
            grpo_trainer.save_model(str(grpo_dir))
            grpo_tokenizer.save_pretrained(grpo_dir)
            save_history("grpo", grpo_trainer)
            print(f"GRPO peak reserved VRAM: {grpo_peak:.2f} GiB")
            release_cuda(grpo_trainer, grpo_policy, grpo_reward_model)
            """
        ),
        _markdown("## 10. 五模型统一推理、指标与报告"),
        _code(
            """
            def load_variant(name):
                if name == "base":
                    return causal_model(BASE_MODEL, train=False), tokenizer_for(BASE_MODEL, "left")
                if name == "sft":
                    base = causal_model(BASE_MODEL, train=False)
                    return (
                        PeftModel.from_pretrained(base, str(sft_dir)),
                        tokenizer_for(BASE_MODEL, "left"),
                    )
                adapter = {"dpo": dpo_dir, "ppo": ppo_dir, "grpo": grpo_dir}[name]
                base = causal_model(merged_dir, train=False)
                return (
                    PeftModel.from_pretrained(base, str(adapter)),
                    tokenizer_for(merged_dir, "left"),
                )

            def generate(model, tokenizer, text):
                prompt = analysis_prompt(tokenizer, text)
                inputs = tokenizer(prompt, return_tensors="pt")
                device = next(model.parameters()).device
                inputs = {key: value.to(device) for key, value in inputs.items()}
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=CFG["max_new_tokens"],
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                tokens = output[0, inputs["input_ids"].shape[1]:]
                return tokenizer.decode(tokens, skip_special_tokens=True).strip()

            def evaluate(raw, review):
                payload = raw_schema_payload(raw)
                if payload is None:
                    return {
                        "schema_valid": 0,
                        "evidence_grounded": 0,
                        "word_limit_ok": 0,
                        "instruction_following": 0,
                    }
                evidence_ok = all(grounded(review, x) for x in payload["evidence"])
                length_ok = len(payload["analysis"].split()) <= 80
                return {
                    "schema_valid": 1,
                    "evidence_grounded": int(evidence_ok),
                    "word_limit_ok": int(length_ok),
                    "instruction_following": int(evidence_ok and length_ok),
                }

            prediction_dir = RUN_ROOT / "predictions"
            prediction_dir.mkdir(exist_ok=True)
            metric_rows = []
            for variant in ["base", "sft", "dpo", "ppo", "grpo"]:
                model, variant_tokenizer = load_variant(variant)
                predictions = []
                for row in test_rows:
                    raw = generate(model, variant_tokenizer, row["text"])
                    predictions.append(
                        {
                            "id": row["id"],
                            "text": row["text"],
                            "variant": variant,
                            "raw_output": raw,
                            **evaluate(raw, row["text"]),
                        }
                    )
                write_jsonl(prediction_dir / f"{variant}.jsonl", predictions)
                metric_rows.append(
                    {
                        "variant": variant,
                        "examples": len(predictions),
                        **{
                            f"{key}_rate": float(
                                np.mean([item[key] for item in predictions])
                            )
                            for key in [
                                "schema_valid",
                                "evidence_grounded",
                                "word_limit_ok",
                                "instruction_following",
                            ]
                        },
                    }
                )
                release_cuda(model)

            metrics = pd.DataFrame(metric_rows)
            metrics.to_csv(RUN_ROOT / "metrics" / "five_model_metrics.csv", index=False)
            display(metrics)
            report = [
                "# Amazon Review Alignment Colab Report",
                "",
                f"- Profile: {PROFILE}",
                f"- Base model: {BASE_MODEL}",
                f"- Human calibration: {USE_HUMAN_CALIBRATION}",
                "",
                metrics.to_markdown(index=False),
                "",
                "## Limitations",
                "",
                "- Smoke results validate wiring only and do not establish convergence.",
                (
                    "- Offline preferences are synthetic and cannot substitute "
                    "for formal human evaluation."
                ),
                "- PPO and GRPO use the same prompts but different rollout counts.",
            ]
            report_path = RUN_ROOT / "report.md"
            report_path.write_text("\\n".join(report), encoding="utf-8")
            print("Report:", report_path)
            """
        ),
        _markdown(
            """
            ## 11. 下一步

            1. 确认 smoke 全链路完成且各项 loss/reward 为有限值。
            2. 在 A100 Runtime 中将 `PROFILE` 改为 `a100`，上传正式教师数据。
            3. 完成 PPO vs GRPO 的独立人工盲评。
            4. 不预设 GRPO、PPO 或 DPO 必然优于 SFT。
            """
        ),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()

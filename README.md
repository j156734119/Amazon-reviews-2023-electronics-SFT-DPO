# Amazon Review Alignment: SFT, DPO, PPO, and GRPO

This project trains and evaluates small language models for evidence-grounded
Amazon Electronics review analysis. The current maintained workflow is the
single Colab A100 notebook:

```text
Amazon_Review_Alignment_A100.ipynb
```

The model must return only this JSON shape:

```json
{
  "sentiment": "negative | neutral | positive",
  "evidence": ["one or two exact spans copied from the review"],
  "analysis": "a concise explanation grounded only in the review"
}
```

## Current Algorithm Flow

1. **Prepare data**
   - Download Amazon Reviews 2023 Electronics reviews from Hugging Face.
   - Normalize, deduplicate, keep at most one review per `parent_asin`.
   - Build deterministic stratified train/validation/test splits.
   - Star rating is used only for sampling/diagnostics; it is not shown to the model.

2. **Teacher preference generation**
   - Use the OpenAI teacher model configured in `configs/full.yaml`.
   - Generate structured chosen/rejected preference pairs.
   - Validate JSON schema and exact evidence grounding.
   - Submit/reuse Batch API jobs with cost caps and resume metadata.

3. **SFT**
   - Fine-tune `Qwen/Qwen3.5-2B` with QLoRA on teacher chosen responses.
   - Save the SFT LoRA adapter.
   - Merge SFT into a standalone policy for downstream DPO/PPO/GRPO.

4. **DPO**
   - Train a DPO adapter from teacher preference pairs.
   - Current formal profile uses the v2 DPO output path from
     `configs/rlhf_a100_dpo_v2.yaml`.

5. **Reward model and online RLAIF data**
   - Split preference data into reward-model training/validation pairs.
   - Sample shared prompts for PPO and GRPO from training data only.
   - Train a reward model used by PPO and as one component of GRPO rewards.

6. **PPO**
   - Start from the merged SFT policy.
   - Train a PPO LoRA adapter with KL control against the SFT reference policy.
   - Current formal v2 output path is `models/ppo-v2`.

7. **GRPO**
   - Start from the merged SFT policy.
   - Generate grouped completions on the same prompt set as PPO.
   - Reward = frozen reward model + schema validity + grounded evidence + length control.
   - Current formal v2 output path is `models/grpo-v2`.

8. **Evaluation**
   - Evaluate `base`, `sft`, `dpo`, `ppo`, `grpo` on the same test examples.
   - Local metrics: schema validity, evidence grounding, word-limit compliance,
     and instruction following.
   - Optional blinded LLM judge compares randomly swapped model outputs pairwise.

9. **Baselines**
   - `qwen35_2b_fewshot`: Qwen3.5-2B prompt-only few-shot baseline.
   - `nlptown_template`: product-review sentiment classifier plus grounded template.
   - `deepseek_v4_pro_fewshot`: optional DeepSeek API few-shot baseline.

For the rationale behind these metrics, how to interpret a fine-tuned 2B model
against larger prompt-only models, and how to scale the method to Qwen 10B+,
see [Evaluation Metrics And Scaling Strategy](docs/evaluation_and_scaling.md).

## Maintained Files

- `Amazon_Review_Alignment_A100.ipynb`: end-to-end Colab workflow.
- `configs/full.yaml`: base formal A100 data/model/training profile.
- `configs/rlhf_a100.yaml`: reward/PPO/GRPO profile.
- `configs/rlhf_a100_dpo_v2.yaml`: DPO v2 override.
- `configs/rlhf_a100_online_v2.yaml`: current formal online PPO/GRPO + baseline evaluation profile.
- `src/amazon_review_alignment/`: package code and CLI.
- `tools/deepseek_judge.py`: optional independent DeepSeek blind judge.
- `legacy/`: archived old configs/scripts that are no longer part of the maintained run path.

## Google Colab Full Run

Use an A100 runtime. T4 is no longer the maintained path for the full experiment.

### 1. Push local code first

Commit and push the latest repo state to GitHub before opening Colab. The
notebook clones or fast-forwards from GitHub into Google Drive, so unpushed
local changes will not be visible in Colab.

### 2. Open the notebook

Open:

```text
Amazon_Review_Alignment_A100.ipynb
```

In Colab, select:

```text
Runtime -> Change runtime type -> A100 GPU
```

### 3. Add Colab secrets

Required:

```text
OPENAI_API_KEY
```

Recommended:

```text
HF_TOKEN
```

Optional:

```text
DEEPSEEK_API_KEY
```

`DEEPSEEK_API_KEY` is needed only for the DeepSeek baseline or DeepSeek
cross-judge.

### 4. Run notebook cells in order

The notebook is intentionally split into resumable stages:

1. Mount Drive and sync the repo.
2. Install dependencies.
3. Restart the Colab runtime when the install cell tells you to.
4. Restore environment and check A100/BF16.
5. Build an isolated A100 smoke config under `/content/`.
6. Run the full smoke pipeline.
7. Prepare formal data and teacher preferences.
8. Train formal SFT.
9. Merge SFT.
10. Train DPO v2.
11. Build reward/PPO/GRPO data.
12. Train reward model.
13. Train PPO v2.
14. Train GRPO v2.
15. Evaluate trained variants.
16. Run configured baselines.
17. Optionally enable blind LLM judge.

If Colab disconnects, reconnect, rerun the restore/helper cells, then continue
from the last incomplete stage. The formal training cells are separated so you
do not need to rerun earlier stages after a late failure.

### 5. Outputs

The formal run writes to:

```text
outputs/a100-qwen3.5-2b/
```

Important outputs:

```text
outputs/a100-qwen3.5-2b/data/
outputs/a100-qwen3.5-2b/teacher/
outputs/a100-qwen3.5-2b/models/
outputs/a100-qwen3.5-2b/rlhf/
outputs/a100-qwen3.5-2b/evaluation/metrics.csv
outputs/a100-qwen3.5-2b/evaluation/report.md
outputs/a100-qwen3.5-2b/evaluation/judge_pairwise_summary.csv
```

## Local CLI Reference

The notebook is the recommended runner, but the same stages are available via CLI:

```powershell
review-align prepare-data --config configs/rlhf_a100_online_v2.yaml
review-align teacher-pilot --config configs/rlhf_a100_online_v2.yaml
review-align teacher-batch --config configs/rlhf_a100_online_v2.yaml
review-align train-sft --config configs/rlhf_a100_online_v2.yaml
review-align merge-sft --config configs/rlhf_a100_online_v2.yaml
review-align train-dpo --config configs/rlhf_a100_online_v2.yaml
review-align build-rlhf-data --config configs/rlhf_a100_online_v2.yaml
review-align train-reward --config configs/rlhf_a100_online_v2.yaml
review-align train-ppo --config configs/rlhf_a100_online_v2.yaml
review-align train-grpo --config configs/rlhf_a100_online_v2.yaml
review-align evaluate --config configs/rlhf_a100_online_v2.yaml
review-align build-report --config configs/rlhf_a100_online_v2.yaml
```

Baseline-only commands:

```powershell
review-align inference --config configs/rlhf_a100_online_v2.yaml --variant qwen35_2b_fewshot
review-align inference --config configs/rlhf_a100_online_v2.yaml --variant nlptown_template

$env:DEEPSEEK_API_KEY="..."
review-align inference --config configs/rlhf_a100_online_v2.yaml --variant deepseek_v4_pro_fewshot
```

Optional DeepSeek cross-judge:

```powershell
python tools/deepseek_judge.py `
  --config configs/rlhf_a100_online_v2.yaml `
  --root outputs/a100-qwen3.5-2b `
  --samples-per-pair 100 `
  --model deepseek-v4-pro
```

## Tests

```powershell
python -m pip install -e ".[eval,dev]"
pytest
```

Full training and model inference require a Linux CUDA GPU environment. Local
unit tests do not require API keys or a GPU.

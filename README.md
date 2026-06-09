# Amazon Review Alignment: SFT + DPO

An engineering-oriented research pipeline for testing whether supervised
fine-tuning (SFT) followed by Direct Preference Optimization (DPO) improves
small language models at producing structured, faithful, evidence-grounded
analysis of Amazon Electronics reviews.

The original DistilBERT classification notebook and course report are retained
under `legacy/` as project history. The current task is generative:

```json
{
  "sentiment": "negative | neutral | positive",
  "evidence": ["an exact span copied from the review"],
  "analysis": "a concise analysis grounded only in the review"
}
```

## Research comparison

The pipeline evaluates three policies using the same prompts, decoding settings,
and held-out test reviews:

1. Base: `Qwen/Qwen3-0.6B`
2. SFT: 4-bit QLoRA on teacher-generated structured analyses
3. SFT + DPO: preference optimization from the SFT policy

Primary outcomes are schema validity, exact evidence grounding, instruction
following, and blinded pairwise preference. Rating agreement is retained only
as a weak-label diagnostic, not as the main objective.

## Installation

Python 3.10+ is required. Core data, teacher, evaluation, and test tooling:

```powershell
python -m pip install -e ".[eval,dev]"
```

Training dependencies:

```powershell
python -m pip install -e ".[train,eval,dev]"
```

`bitsandbytes` support depends on the CUDA, PyTorch, and operating-system
combination. Cloud Linux with a CUDA GPU is the recommended environment for the
full SFT and DPO runs. The local smoke configuration is intended to validate
the pipeline and tiny training runs, not to establish research results.

## Pipeline

```powershell
review-align prepare-data --config configs/full.yaml

$env:OPENAI_API_KEY = "..."
review-align teacher-pilot --config configs/full.yaml --limit 100
review-align teacher-batch --config configs/full.yaml

review-align train-sft --config configs/full.yaml
review-align train-dpo --config configs/full.yaml
review-align evaluate --config configs/full.yaml

review-align human-eval --config configs/full.yaml --samples 200
review-align build-report --config configs/full.yaml
```

Start with `configs/smoke.yaml` to exercise each stage on a small sample.
Generated datasets, API records, adapters, predictions, metrics, plots, and
reports are written below `outputs/`.

## Teacher generation and cost control

Teacher data uses the OpenAI Responses API with Structured Outputs and the
pinned `gpt-5.4-mini-2026-03-17` snapshot. Credentials are read only from
`OPENAI_API_KEY`.

`teacher-pilot` records actual token usage and projects the Batch API cost.
`teacher-batch` refuses to submit when the projected total exceeds the
configured `$12` cap. Re-running `teacher-batch` checks or downloads an existing
batch instead of submitting a duplicate.

## Evaluation

Local evaluation reports:

- valid JSON and schema rate;
- exact evidence substring match rate;
- analysis word-limit compliance;
- overall instruction-following rate;
- failure examples.

Optional OpenAI judging compares anonymized model outputs using the same
faithfulness, evidence, concision, and usefulness rubric used by the human
evaluation form. Pairwise win rates include bootstrap 95% confidence intervals.

Human evaluation creates a blinded CSV. For each row, select `A`, `B`, or
`tie` in the `choice` column, then summarize it with:

```powershell
review-align human-eval --config configs/full.yaml `
  --responses outputs/evaluation/human_eval_responses.csv
```

## Reproducibility and safety

- Reviews are normalized and deduplicated.
- At most one review is retained per `parent_asin`, preventing product leakage.
- Splits are deterministic and stratified by the hidden star-rating metadata.
- Rating is never included in teacher or student prompts.
- Every stage records its configuration, seed, model identifiers, Git commit,
  source information, and available token usage.
- Original source data and legacy artifacts are not overwritten.

## Tests

```powershell
pytest
```

Tests do not require an API key, network access, or a GPU. Training data
formatters and trainer wiring are exercised with tiny mocks; actual model
downloads and CUDA execution are environment-dependent integration steps.

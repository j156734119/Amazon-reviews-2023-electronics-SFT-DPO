# Evaluation Metrics And Scaling Strategy

This project should be evaluated as a constrained, evidence-grounded generation
task rather than as an open-ended chat or generic summarization task.

The task is:

```text
Amazon review -> sentiment + verbatim evidence + grounded analysis
```

The intended output is a machine-parseable JSON object:

```json
{
  "sentiment": "negative | neutral | positive",
  "evidence": ["one or two exact spans copied from the review"],
  "analysis": "a concise explanation grounded only in the review"
}
```

## What Better Means

The project should not claim that a fine-tuned 2B model is generally more
capable than a 10B model or a GPT-class model. The defensible claim is narrower:

> A smaller model, after task-specific SFT/preference/RLAIF alignment, can be
> more reliable than a larger prompt-only model on this constrained Amazon
> review-analysis protocol.

Here, "better" means better task reliability:

- stable JSON output;
- exact evidence copying from the review;
- fewer unsupported claims;
- concise analysis;
- better blinded preference on the same held-out reviews.

This is a domain/task alignment claim, not a general intelligence claim.

## Primary Metrics

Use task-specific reliability metrics as the main evidence.

| Metric | Role | Why it matters |
|---|---|---|
| `schema_valid_rate` | Format reliability | The output must be parseable and usable by downstream systems. |
| `evidence_grounded_rate` | Hallucination control | Evidence must be copied from the source review, not invented or paraphrased. |
| `word_limit_ok_rate` | Concision | The model must follow the requested analysis budget. |
| `instruction_following_rate` | Main automatic metric | A response only counts as following instructions when schema, evidence, and length constraints all pass. |
| Blind pairwise LLM judge | Semantic quality | Rule metrics cannot fully judge whether the analysis is faithful, useful, and less over-inferred. |

Generic overlap metrics such as BLEU or ROUGE should not be primary metrics
because the task is not reference-style summarization. There can be multiple
valid analyses for the same review as long as they are schema-valid, grounded,
and concise.

## How To Explain 2B Tuned vs Larger Prompt-Only Models

Larger prompt-only models often have stronger general language ability, but
they are not optimized for this exact output contract. They may:

- produce prose instead of strict JSON;
- paraphrase evidence instead of copying a substring;
- infer facts not stated in the review;
- write a longer, more fluent answer that violates the task constraints.

Fine-tuning directly optimizes the behavioral contract:

- produce the exact JSON schema;
- choose evidence spans from the review;
- keep analysis short;
- avoid using hidden star ratings;
- avoid unsupported claims.

Therefore, a tuned 2B model can beat a larger prompt-only model on
task-specific reliability metrics while still being weaker in broad language
understanding or general reasoning.

Suggested wording:

> The result should be interpreted as task-specific alignment outperforming raw
> scale under constrained evaluation, not as evidence that the smaller model has
> stronger general capability.

## Improvement Priorities

Hyperparameters matter, but they are not the main source of vertical task
improvement. The practical priority order is:

```text
teacher/data quality > hard negatives > reward fidelity > scaling base model > hyperparameters
```

### 1. Improve Teacher Data Quality

The current strict validation can quarantine many teacher outputs when evidence
is not an exact substring. Improving teacher data quality is likely more useful
than small learning-rate changes.

Teacher prompt improvements should emphasize:

- evidence must be a contiguous substring from the review;
- do not add quotation marks unless they appear in the original review;
- do not paraphrase evidence;
- choose short but complete spans;
- keep chosen analysis under the word limit.

The target is not just more teacher rows, but more validated rows that satisfy
the same constraints used at evaluation time.

### 2. Build Hard Negatives

Preference training is most useful when rejected answers resemble real model
failures. Strong rejected examples include:

- incorrect sentiment;
- fabricated evidence not present in the review;
- evidence that is present but does not support the sentiment;
- over-inferred analysis;
- malformed or extra-field JSON;
- valid-looking but overly verbose analysis.

This makes DPO/PPO/GRPO learn the actual boundary between a compliant and a
non-compliant answer.

### 3. Align Rewards With Metrics

PPO/GRPO rewards should directly reflect the evaluation contract:

- schema-valid reward;
- exact evidence grounding reward;
- concise length reward;
- sentiment consistency reward;
- unsupported-inference penalty where possible;
- frozen reward-model score as a semantic preference component.

If reward components do not match the final metrics, online optimization can
improve training reward while hurting evaluation reliability.

### 4. Use Error-Driven Retraining

After each run, inspect failure cases before changing parameters. Segment errors
by:

- schema failures;
- evidence grounding failures;
- over-inference;
- sentiment mistakes;
- long-review vs short-review failures;
- negative/neutral/positive class behavior.

Use these failure modes to add targeted teacher examples, hard negatives, or
reward checks.

## Scaling To Qwen 10B Or Larger

Do not jump directly from 2B full RLAIF to 10B full RLAIF. Use a staged scaling
matrix:

| Model | Training | Purpose |
|---|---|---|
| Qwen3.5-2B | few-shot | Small prompt-only baseline |
| Qwen3.5-2B | SFT | Supervised vertical adaptation |
| Qwen3.5-2B | DPO/PPO/GRPO | Preference/RLAIF gain |
| Qwen3.5-10B+ | few-shot | Larger prompt-only baseline |
| Qwen3.5-10B+ | SFT-only | Lowest-cost scaling test |
| Qwen3.5-10B+ | DPO or GRPO | Only run if SFT shows enough gain |

Separate the effects:

```text
fine-tuning gain = tuned model - same base prompt-only
scale gain = larger prompt-only - smaller prompt-only
alignment gain = DPO/PPO/GRPO - SFT
```

This avoids confusing "bigger model" gains with "better alignment" gains.

## Capacity Limits

Vertical fine-tuning has a ceiling. It shifts model behavior toward a narrow
task, but it does not create unlimited reasoning ability.

Limits come from:

- base model capacity;
- ambiguity or missing information in the review;
- teacher noise;
- reward-model errors;
- insufficient hard negatives;
- strict exact-substring evidence constraints;
- limited training coverage.

The expected gain is largest when the task is narrow, structured, and
verifiable. Gains shrink when the task requires broad reasoning, external
knowledge, or ambiguous judgment.

## Recommended Claim

Use this claim in reports or presentations:

> This project evaluates task-specific reliability rather than general language
> ability. Since the target output is structured and evidence-grounded, the main
> metrics are schema validity, exact evidence grounding, instruction following,
> and blinded pairwise preference. A fine-tuned 2B model may outperform a larger
> prompt-only model because fine-tuning directly optimizes the behavioral
> constraints of this vertical task. The result should be interpreted as
> evidence that domain alignment improves reliability under constrained
> review-analysis settings, not as a claim that the smaller model is generally
> more capable.

# Amazon Review Alignment Results

## Research question

How do SFT, DPO, PPO, and GRPO affect a small model's structured, evidence-grounded Amazon review analyses under a constrained compute budget?

The comparison uses identical prompts, decoding settings, and held-out reviews. The previous DistilBERT classification result is historical context and is not treated as a directly comparable baseline.

## Local evaluation

| Variant | Examples | Schema valid | Evidence grounded | Word limit | Instruction following |
|---|---:|---:|---:|---:|---:|
| base | 500 | 95.4% | 84.2% | 95.4% | 84.2% |
| sft | 500 | 100.0% | 96.0% | 100.0% | 96.0% |
| dpo | 500 | 84.6% | 80.0% | 84.6% | 80.0% |
| ppo | 500 | 100.0% | 95.2% | 100.0% | 95.2% |
| grpo | 500 | 100.0% | 94.8% | 100.0% | 94.8% |

![Evaluation metrics](metrics.png)

## Pairwise evaluation

### LLM judge

| Comparison | N | Right-model win rate | 95% CI | Ties |
|---|---:|---:|---:|---:|
| base_vs_sft | 100 | 52.5% | 43.0%–61.5% | 5 |
| base_vs_dpo | 100 | 57.5% | 48.0%–67.0% | 5 |
| base_vs_ppo | 100 | 45.0% | 35.5%–54.5% | 4 |
| base_vs_grpo | 100 | 49.5% | 40.0%–59.0% | 7 |
| sft_vs_dpo | 100 | 60.5% | 51.5%–69.5% | 13 |
| sft_vs_ppo | 100 | 53.5% | 46.0%–61.0% | 43 |
| sft_vs_grpo | 100 | 48.0% | 40.0%–56.0% | 30 |
| dpo_vs_ppo | 100 | 47.5% | 38.5%–56.0% | 21 |
| dpo_vs_grpo | 100 | 41.0% | 32.5%–49.5% | 22 |
| ppo_vs_grpo | 100 | 50.5% | 44.0%–57.0% | 57 |

## Reward Model

| Split | Examples | Preference accuracy | Mean reward margin |
|---|---:|---:|---:|
| ai_validation | 130 | 90.0% | 2.3178 |
| human_held_out | 0 | 0.0% | 0.0000 |

## PPO feasibility metrics

- Episodes: 1024
- Unique prompts: 1024
- Runtime: 176.8 minutes
- Peak allocated CUDA memory: 13.62 GiB
- Peak reserved CUDA memory: 13.82 GiB
- Reference policy: merged SFT policy with the PPO adapter disabled.
- `objective/kl`: -0.011823
- `objective/rlhf_reward`: 3.593987
- `objective/scores`: 3.593750
- `loss/policy_avg`: 0.056113
- `loss/value_avg`: 3.856393

## GRPO feasibility metrics

- Unique prompts: 1024
- Generations per prompt: 4
- Expected completions per epoch: 4096
- Runtime: 315.8 minutes
- Peak allocated CUDA memory: 8.49 GiB
- Peak reserved CUDA memory: 9.37 GiB
- Reference policy: merged SFT policy with the GRPO adapter disabled.
- Reward weights: `{"reward_model": 2.0, "schema": 0.25, "evidence": 0.5, "length": 0.1}`
- `completions/mean_length`: 63.750000
- `completions/clipped_ratio`: 0.000000
- `rewards/sft-merged/mean`: -1.158203
- `rewards/sft-merged/std`: 0.963662
- `rewards/schema_reward/mean`: 1.000000
- `rewards/schema_reward/std`: 0.000000
- `rewards/evidence_reward/mean`: 1.000000
- `rewards/evidence_reward/std`: 0.000000
- `rewards/length_reward/mean`: 1.000000
- `rewards/length_reward/std`: 0.000000
- `reward`: -1.466406
- `reward_std`: 1.927324
- `frac_reward_zero_std`: 0.000000
- `kl`: 0.029740
- `entropy`: 0.588886
- `clip_ratio/region_mean`: 0.000000

## Interpretation constraints

- Star ratings are hidden from the teacher and student and are not the main target.
- Teacher-generated preferences can encode teacher bias despite validation.
- Exact substring matching measures evidence traceability, not full causal faithfulness.
- A null or negative DPO result is a valid outcome and must not be reframed as success.
- PPO and GRPO use AI-generated preferences, so they are reported as RLAIF rather than RLHF.
- The PPO policy receives only 256 episodes in the T4 configuration; it is a resource-constrained baseline, not a convergence claim.
- GRPO uses 256 prompts with four sampled completions each; it is a resource-constrained RLAIF baseline, not a convergence claim.
- Full conclusions require the configured cloud training runs and blinded evaluation.

# A100 Qwen3.5-2B Experiment Analysis

## Main findings

- SFT improved evidence grounding from 84.2% to 96.0% and schema validity from 95.4% to 100.0%.
- DPO v2 reduced schema validity to 84.6% and evidence grounding to 80.0%; its semantic AI-judge result must be interpreted together with this structural regression.
- PPO and GRPO preserved high evidence grounding (95.4% and 95.0%) but did not exceed SFT locally.
- The Reward Model reached 90.0% preference accuracy on 130 AI-generated validation pairs, but no held-out human preferences were available.
- GRPO rule rewards were fully saturated: schema, evidence, and length reward standard deviations were all zero. KL was 0.000557 and clip ratio was zero, indicating very limited policy movement.
- PPO and GRPO produced exactly identical outputs on 66.4% of test reviews and the same sentiment on 99.8%. This supports the observation that the two online-RL policies remained very close.
- SFT and PPO agreed on sentiment for 99.6% of test reviews, which further suggests that the PPO budget mostly preserved the SFT policy.
- PPO used 128 episodes. Its value loss remained 5.7251, so the value function was not yet a strong estimator under this limited rollout budget.

## Instruction-following failures

- base: 79 / 500
- sft: 20 / 500
- dpo: 100 / 500
- ppo: 23 / 500
- grpo: 25 / 500

## AI blind judge

The archived judge log contains 350 decisions across 7 pairwise comparisons.

- base_vs_sft: sft win rate 57.0% (N=50, ties=3).
- sft_vs_dpo: dpo win rate 75.0% (N=50, ties=7).
- sft_vs_ppo: ppo win rate 41.0% (N=50, ties=19).
- sft_vs_grpo: grpo win rate 48.0% (N=50, ties=20).
- dpo_vs_ppo: ppo win rate 36.0% (N=50, ties=10).
- dpo_vs_grpo: grpo win rate 40.0% (N=50, ties=12).
- ppo_vs_grpo: grpo win rate 46.0% (N=50, ties=44).

## Reproducibility note

The local metrics, complete AI blind-judge decision log, aggregate pairwise summary, and training diagnostics are archived under this output directory. Raw review text, full model predictions, adapters, and checkpoints remain ignored.

## Generated tables

- `metric_deltas.csv`: percentage-point changes against Base and SFT.
- `failure_summary.csv`: failed instruction-following examples by model.
- `prediction_summary.csv`: output lengths, JSON parsing, and sentiment counts.
- `pairwise_output_agreement.csv`: exact output and sentiment agreement.
- `training_summary.csv`: Reward Model, PPO, and GRPO diagnostics.

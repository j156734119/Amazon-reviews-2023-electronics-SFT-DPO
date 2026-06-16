# A100 Qwen3.5-2B Experiment Analysis

## Main findings

- SFT improved evidence grounding from 84.2% to 96.0% and schema validity from 95.4% to 100.0%.
- DPO v2 reduced schema validity to 84.6% and evidence grounding to 80.0%; its semantic AI-judge result must be interpreted together with this structural regression.
- PPO and GRPO preserved high evidence grounding (95.2% and 94.8%) but did not exceed SFT locally.
- The Reward Model reached 90.0% preference accuracy on 130 AI-generated validation pairs, but no held-out human preferences were available.
- GRPO rule rewards were fully saturated: schema, evidence, and length reward standard deviations were 0.000, 0.000, and 0.000. KL was 0.029740 and clip ratio was 0.000000.
- PPO and GRPO produced exactly identical outputs on 66.4% of test reviews and the same sentiment on 99.8%. This supports the observation that the two online-RL policies remained very close.
- SFT and PPO agreed on sentiment for 99.6% of test reviews, which further suggests that the PPO budget mostly preserved the SFT policy.
- PPO used 1024 episodes over 1024 unique prompts. Its final value loss was 3.8564, so the value function should still be treated as a limited-budget estimator rather than a fully converged critic.

## Instruction-following failures

- base: 79 / 500
- sft: 20 / 500
- dpo: 100 / 500
- ppo: 23 / 500
- grpo: 25 / 500

## AI blind judge

The archived judge log contains 1000 decisions across 10 pairwise comparisons.

- base_vs_sft: sft win rate 52.5% (N=100, ties=5).
- base_vs_dpo: dpo win rate 57.5% (N=100, ties=5).
- base_vs_ppo: ppo win rate 45.0% (N=100, ties=4).
- base_vs_grpo: grpo win rate 49.5% (N=100, ties=7).
- sft_vs_dpo: dpo win rate 60.5% (N=100, ties=13).
- sft_vs_ppo: ppo win rate 53.5% (N=100, ties=43).
- sft_vs_grpo: grpo win rate 48.0% (N=100, ties=30).
- dpo_vs_ppo: ppo win rate 47.5% (N=100, ties=21).
- dpo_vs_grpo: grpo win rate 41.0% (N=100, ties=22).
- ppo_vs_grpo: grpo win rate 50.5% (N=100, ties=57).

## Reproducibility note

The local metrics, complete AI blind-judge decision log, aggregate pairwise summary, and training diagnostics are archived under this output directory. Raw review text, full model predictions, adapters, and checkpoints remain ignored.

## Generated tables

- `metric_deltas.csv`: percentage-point changes against Base and SFT.
- `failure_summary.csv`: failed instruction-following examples by model.
- `prediction_summary.csv`: output lengths, JSON parsing, and sentiment counts.
- `pairwise_output_agreement.csv`: exact output and sentiment agreement.
- `training_summary.csv`: Reward Model, PPO, and GRPO diagnostics.

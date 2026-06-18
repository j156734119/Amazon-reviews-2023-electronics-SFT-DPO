# DeepSeek LLM Judge Summary

Judge model: `deepseek-v4-flash`

| Comparison | N | Right-model win rate | 95% CI | Ties |
|---|---:|---:|---:|---:|
| base_vs_sft | 100 | 42.0% | 33.0%-51.0% | 14 |
| base_vs_dpo | 100 | 52.5% | 42.5%-62.0% | 3 |
| base_vs_ppo | 100 | 40.5% | 31.0%-50.0% | 7 |
| base_vs_grpo | 100 | 45.0% | 36.0%-54.0% | 10 |
| sft_vs_dpo | 100 | 58.5% | 50.5%-66.5% | 31 |
| sft_vs_ppo | 100 | 51.5% | 46.0%-57.0% | 67 |
| sft_vs_grpo | 100 | 49.0% | 42.5%-56.0% | 54 |
| dpo_vs_ppo | 100 | 48.0% | 40.5%-55.5% | 44 |
| dpo_vs_grpo | 100 | 44.5% | 37.0%-52.0% | 39 |
| ppo_vs_grpo | 100 | 51.5% | 48.5%-54.5% | 91 |

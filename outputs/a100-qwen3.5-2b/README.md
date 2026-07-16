# A100 Qwen3.5-2B Results

This directory contains the public, reproducible subset of the formal A100
experiment.

Tracked files include aggregate evaluation metrics, training diagnostics,
plots, reports, and derived analysis tables. Full model predictions, sampled
Amazon review text, Reward Model training pairs, prompts, checkpoints, and
adapter weights remain local and ignored by Git.

Run the analysis again with:

```powershell
python legacy/tools/analyze_results.py --root outputs/a100-qwen3.5-2b
```

The `evaluation/` directory also includes the complete AI blind-judge decision
log and its aggregate pairwise summary. The archived run evaluated seven
model pairs with 50 held-out examples per pair.

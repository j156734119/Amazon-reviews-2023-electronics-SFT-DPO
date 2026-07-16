# Legacy Files

This directory contains archived scripts and configs that are no longer part of
the maintained A100 Colab workflow.

Archived files:

- `configs/smoke.yaml`: old standalone Qwen3-0.6B smoke profile.
- `configs/rlhf_smoke.yaml`: old T4-style tiny RLHF smoke profile.
- `configs/rlhf_t4.yaml`: old Qwen3-0.6B T4 compatibility profile.
- `tools/build_pipeline_notebooks.py`: old notebook generator that referenced
  removed notebook variants.
- `tools/analyze_results.py`: historical analysis script for archived output
  artifacts.

The maintained workflow is documented in the root `README.md` and runs through
`Amazon_Review_Alignment_A100.ipynb`.

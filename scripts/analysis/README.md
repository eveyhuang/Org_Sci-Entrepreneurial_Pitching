# Analysis commands

Run from the repository root with `.venv/bin/python`:

- `execute_notebook.py notebooks/archive/stage_b_run_explorer.ipynb`: execute and save outputs
  using the same interpreter, without installing a global kernelspec.
- `analyze_panel.py --run runs/<run> --analysis-id ID`: panel (v3) analysis exports under
  `<run>/analysis/<ID>/`; offline, refuses to overwrite without `--force`. See `analysis/README.md`.
- `analyze_hu_ma_stage_a.py --stage-a-run runs/<stage_a run> --rank-run runs/<ranking run>`: Stage A
  ratings and forced ranking on the Hu & Ma videos against `data/hu-ma/funding_outcomes.csv`
  (Pitch Factors, Table III and IV analogues, repeatability, method convergence, and the agreement with
  Hu & Ma's own Pitch Factor from `--hu-ma-pitch-factor`, default `data/hu-ma/pitch_factor.csv`, `none` to
  skip) under `<stage_a run>/analysis/hu_ma_funding/`. Needs `tables/annotations.parquet` (`scripts/aggregate.py`).
- `compare_v1_v2.py`: historical Stage A comparison on its fixed two runs.
- `pilot5_within_video.py runs/<run>`: historical five-video within-pitch analysis.
- `pilot5_reciprocity.py runs/<run>`: historical five-video reciprocity analysis.

The pilot scripts retain their original methodological choices and fixed-sample assumptions.
They are not the entry point for the 50-video explorer (exact permutation enumeration may be
impractical at that scale). Use the notebook for the current larger run.

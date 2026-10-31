# Stage A notebooks

## `stage_a_hu_ma_analysis.ipynb`

The Hu & Ma accelerator videos under the current Stage A instrument, **without reading or writing Python**: how
far the replicate ratings agree (ICC per code), the Pitch Factor built from the ten ratings and the one built
from the forced-ranking scores (loadings, uniqueness, variance share), how far the two methods agree (Spearman,
overall and per code, with a figure), how far each agrees with Hu & Ma's own Pitch Factor computed with their code
on the same videos (convergent validity: overall, per dimension, and by loadings, with two figures), and the
relation of each to the accelerator's funding decision (Hu & Ma
Table III and Table IV analogues: maximum-likelihood logit marginal effects, Firth penalized logit, exact
permutation test of the AUC). It calls the same functions as `scripts/analysis/analyze_hu_ma_stage_a.py`
(`analysis/hu_ma_funding.py`) and is committed with its outputs from a real execution on the October 2026 runs,
so it can be read without running anything. The written companion is
[`docs/results/stage_A/hu_ma_reannotation_oct7.md`](../../docs/results/stage_A/hu_ma_reannotation_oct7.md).

### Running it

1. Open it in Jupyter or VS Code and pick the project's `.venv` Python as the kernel.
2. Edit the **configuration cell** (the first code cell; it is the only cell you need to edit):

| variable | default | meaning |
|---|---|---|
| `STAGE_A_RUN_NAME` | `"stage_a_pitch_factor_20261008_004807"` | the rating run: a folder under `runs/` (or a full path) with `tables/annotations.parquet` (`scripts/aggregate.py` writes it) |
| `RANK_RUN_NAME` | `"stage_a_hu_ma_ranking_v3dev8"` | the ranking run, a folder with `ranking/scores.csv`; `None` = ratings only |
| `OUTCOMES_CSV` | `"data/hu-ma/funding_outcomes.csv"` | the funding labels (`video_id`, `Applied to`, `Accepted?` as 0, 1 or NA) |
| `HU_MA_PITCH_FACTOR_CSV` | `"data/hu-ma/pitch_factor.csv"` | Hu & Ma's Pitch Factor and ten features from their code on the same videos (`video_id`, `pitch_factor`, `Visual-Positive`, ...); `None` skips section 5 |
| `ANALYSIS_ID` | `"notebook"` | folder name under `runs/<rating run>/analysis/` used when `EXPORT` is `True` |
| `EXPORT` | `False` | `True` writes the tables and a Markdown summary there; `False` writes nothing |

3. *Kernel → Restart & Run All*. A few seconds; **no model (API) calls**; it never changes annotation files.

To re-execute from the repository root and keep the outputs in the file (what was done before committing):

```sh
.venv/bin/python scripts/analysis/execute_notebook.py notebooks/stage_A/stage_a_hu_ma_analysis.ipynb
```

### Sections

1. **Provenance**: model, prompt hash, finished and failed units, ranking design and completion, calls and tokens;
   videos and acceptances per accelerator family.
2. **Ratings, ICC**: per code, mean, spread between and within videos, identical-replicate share, ICC(1), distinct values.
3. **Pitch Factors**: loadings and uniqueness from the ratings and from the rank scores, with the variance share.
4. **Agreement between the methods**: Spearman between the two factors and per code; one figure, one dot per video,
   accepted videos marked and labelled.
5. **Against Hu & Ma's own Pitch Factor**: per Stage A method, Pearson (with a Fisher interval), Spearman, Kendall
   and same-tercile share against their factor; per dimension, Spearman against their feature of the same name; the
   loadings of the three factors and the correlations between loading vectors; two figures.
6. **Funding**: the Table III analogue for each Pitch Factor (Hu & Ma's included) and the Table IV analogue per code
   and per Hu & Ma feature, with the headline numbers in words and the count of feature-level tests below 0.05.
7. **Export**.

`tests/test_stage_a_notebook.py` checks the committed file and executes the notebook on synthetic runs.

## Historical

`notebooks/archive/stage_a_run_explorer.ipynb` inspects one Stage A run end to end (provenance, codebook, prompt,
one video's output, corpus statistics) and reproduces the analyses in
[`docs/results/stage_A/Prog_sep4.md`](../../docs/results/stage_A/Prog_sep4.md); set
`RUN_ID = "stage_a_v1_20260903_222907"` for that document's numbers. The former `compare_v1_v2.py` lives at
`scripts/analysis/compare_v1_v2.py`.

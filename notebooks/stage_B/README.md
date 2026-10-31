# Stage B notebooks

Two notebooks are the main way to see, check and re-run everything in the panel pipeline **without reading or
writing Python**. Both are committed with their outputs from a real execution on the 100 annotated videos of
`runs/full-corpus-100` (release `v3-dev9`), so you can read them without running anything.

**Start with the root [Demo.ipynb](../../Demo.ipynb)** for the guided walkthrough of annotation and the
concept bottleneck model. These two notebooks provide deeper inspection and full analysis.

**The reported run is in the repository.** `runs/full-corpus-100` includes the 100-video research
records, two model passes per video, instrument snapshots, ranking artifacts, and analysis exports.
Local video binaries and runtime provider caches are optional for offline reproduction. See the
[demo guide](../../docs/demo.md) and its release inventory for the exact offline dependencies.

**Which videos.** The 100 videos are listed in `runs/full-corpus-100/selection_all100.json`: the same 100
pitches the earlier `v3-dev8` run annotated, re-annotated with `v3-dev9`. Five of them are the development (pilot)
videos on which the instrument was first written (`st_38f8bf18e0`, `st_8642720802`, `st_aebb7e3764`,
`st_d218d68cff`, `st_ea076612ce`); the selection file names them under `pilot_development_videos`, and the
analysis's sensitivity without them reads them from there. Both notebooks say which videos they read in
`VIDEO_IDS` in the configuration cell, in one of three ways:

- a selection file, inside the run folder or relative to it: `VIDEO_IDS = "selection_all100.json"`
  (the committed setting: all 100);
- a list of video ids: `VIDEO_IDS = ["st_8a6fd25840", "st_ec62d6e492"]`;
- `VIDEO_IDS = None`: every video of the run that has a result.

No other folder of the run is looked at, except where a notebook says so: the Stage A ranking was run once over
all 100 videos (three batches per video), so the batch orders and the ranking-stability table cover all ranked
videos.

Everything in both notebooks is **model-coded** (no human review has been imported).

| Notebook | What it is for |
|---|---|
| [`panel_run_explorer.ipynb`](panel_run_explorer.ipynb) | **See the instrument and the annotations.** The codebooks and the exact prompt as the model received them, one video's full record (people, the prepared presentation's content ratings, the ask, exchanges, every turn, decisions, a timeline, each answer with what the investors did next), how far the model's two passes agree and where exactly they differ, where human review stands, the Stage A ranking, and what the run cost. |
| [`project_v3_panel_analysis.ipynb`](project_v3_panel_analysis.ipynb) | **The pre-registered pitch-level test.** Computes, with the same function as `scripts/analysis/analyze_panel.py`, the outcome of each pitch, the cut, the features, the descriptives and H4, self-consistency, the ladder C0 to C5 with C2′ and C3q, the one-family models C3a, C4r and C5r against the baseline, the verdicts of H1′ and H8, the five sensitivities, each feature on its own, H9 (an exploratory pattern), one prediction taken apart, the limits and the figures for the paper. |

## How to run either notebook

1. Open it in Jupyter or VS Code and pick the project's `.venv` Python as the kernel.
2. Edit the **configuration cell** (the first code cell; it is marked as the only cell you need to edit).
3. *Kernel → Restart & Run All*. On the 100 videos the explorer takes well under a minute and the analysis
   notebook about ten minutes with the reported settings (`BOOTSTRAP_REPS = 1999`, `PERMUTATIONS = 500`, H9 on
   every core but one); set `BOOTSTRAP_REPS = 199`, `PERMUTATIONS = 0` and `RUN_H9 = False` for a quick look.

They make **no model (API) calls** and cost nothing to run. They **never change annotation files**: they only read
a run. With `EXPORT = False` (and, in the analysis notebook, `SAVE_FIGURES = False`), the committed defaults, they
write nothing at all.

To re-execute from the repository root and keep the outputs in the file (what was done before committing):

```sh
.venv/bin/python scripts/analysis/execute_notebook.py notebooks/stage_B/panel_run_explorer.ipynb
.venv/bin/python scripts/analysis/execute_notebook.py notebooks/stage_B/project_v3_panel_analysis.ipynb --timeout 7200
```

The analysis notebook's numbers are those of the export `runs/full-corpus-100/analysis/all100/`, written by

```sh
.venv/bin/python scripts/analysis/analyze_panel.py --run runs/full-corpus-100 --analysis-id all100 \
    --selection selection_all100.json --with-shark-level --h9-exchange-outcomes --force
```

with the same settings; `tests/test_panel_notebooks.py` checks that they agree, figure captions included.

## `panel_run_explorer.ipynb`

Configuration: `RUN_NAME` (a folder under `runs/`, or a full path), `VIDEO_IDS` (the videos every section is
limited to), `VIDEO_ID` (the video sections B to D show; `None` = the first of them), `PASS` (`"primary"`,
`"replicate"`, or `"reviewed"` when a human-corrected record exists; the analysis reads the primary pass, except
where it fails validation and the replicate fails less, and section A says which per video), `SHOW_FULL_PROMPT`,
the two token prices behind the cost column, and `EXPORT`.

Sections: **A** run overview (videos, release, passes, which pass the analysis reads, validation errors and flags,
review, calls, tokens, estimated cost; how often each value of each code is used) · **B** the instrument as the
model saw it (three codebooks as tables, the prompt with its hash, and whether the run's copy still equals the live
release) · **C** one video's record (actors, prepared window, Stage A ratings of both passes, venture context with
team gender, the prepared presentation's four content ratings (C3b) and the ask (C3c) per pass, exchanges, every
turn grouped by exchange, decisions, concepts coded `other`, validation messages, timeline, and in C10 each answer
with its question, its substance and checkable support, and the asker's and the panel's next move) · **D**
agreement between the two passes (per field for the video and pooled; per turn function; concept confusion tables;
each disagreement with clock times) · **E** human review status, the commands that generate and import a review
sheet (shown, not run) and the validation table once it exists · **F** Stage A ranking (batch orders, pooled
scores, and how stable the ranking is) · **G** comparison with the archived structured pilot · **H** optional
export. Since `v3-dev9` no record holds a closing state, so the explorer no longer shows one.

The codebooks and prompt in section B are read from the selected video's own `bundle/` and `prompts/` folders,
never from the current codebooks. A section whose input is missing prints one line saying so. `EXPORT = True`
writes CSV files to `runs/<run>/analysis/explorer/` and nowhere else.

## `project_v3_panel_analysis.ipynb`

Configuration: `RUN_NAME`, `VIDEO_IDS`, `ANALYSIS_ID` (`"all100"`), `PENALTY` (`"inner_cv"`, the pre-registered
setting, or a number), `BOOTSTRAP_REPS` (1999), `PERMUTATIONS` (500), `SINGLE_FEATURE_DRAWS` (9999),
`RANDOM_SEED` (42), `RUN_H9`, `H9_EXCHANGE_OUTCOMES` and `H9_SENSITIVITY_DRAWS` (the exchange-based H9 outcomes,
computed as an unreported sensitivity at 499 draws), `WITH_SHARK_LEVEL` (the archived shark-level run behind the
F1 paragraph and figure S1), `WORKERS` (processes for H9's draws; no number depends on it), `EXPORT`,
`SAVE_FIGURES`, `FIGURE_WIDTH`, `SINGLE_FEATURE_TOP_N` and `SINGLE_FEATURE_SHOW_ALL`. The ladder, the rules and
the sensitivities are fixed by the pre-registration and are not settings.

Sections: **1** provenance (who coded each video, the settings, the texts quoted) · **2** sample and outcomes (one
row per pitch, the rule, the cut) · **3** the cut, made visible (`INSPECT_VIDEO`) · **4** features (the dictionary
with each column's status, one pitch read downwards, the whole table) · **5** descriptives (questions, frames and
frame matching, answers, exchanges, the prepared presentation and the ask, Stage A ratings, ranks and ranking
stability, teams and ventures) · **H4** what the panel does after an answer, with the shares behind figure 7 and
the turns behind any cell · **6** self-consistency and human agreement · **7** the ladder (what each model fits,
metrics with intervals and permutation p, paired differences, penalties) · **8** verdicts (the rules, H1′ and H8
with their numbers and power sentence) · **9** sensitivities · **10** each concept on its own (FDR-adjusted),
the coefficients of C5, answer quality by outcome · **11** H9, reported as an exploratory pattern (the
exchange-based sensitivity is only pointed at) · **12** inspect one prediction (`INSPECT_MODEL`) · **13** limits
(plan v4 section 11 with the run's own count, and the method's) · **14** exports · **15** the ten figures, two supplementary, and
S1, each with the caption the export wrote.

Every result carries its label: under 20 pitches **demonstration only**, otherwise *pitch-held-out estimate*
(model-coded, on development videos). Inspecting a prediction in section 12 shows model sensitivity, not a causal
effect.

## Where the logic lives

Notebook cells only call a function and display the result. The functions are in tested modules:
`analysis/panel_explorer.py` (views of a run), `analysis/panel_report.py` (the analysis notebook's tables, captions
and plain-language readings), `scripts/analysis/analyze_panel.py` (the computation and the export),
`analysis/pitch_report.py` (report.md and the p-value helpers), `analysis/pitch_figures.py` (the figures),
`analysis/panel_bottleneck.py` (models, bootstrap p-values, the permutation test, FDR, coefficients),
`analysis/verdicts.py` and `analysis/answer_level.py` (H9). Tests: `tests/test_panel_explorer.py`,
`tests/test_panel_report.py`, `tests/test_panel_significance.py`, `tests/test_pitch_figures.py`,
`tests/test_analyze_panel_cli.py`, `tests/test_panel_notebooks.py` (both notebooks are executed against a
synthetic run and must write nothing with `EXPORT = False`; the numbers the committed notebooks show are checked
against the selection file and the exported `analysis/all100/` files).

---

# Archived notebooks (earlier pipelines)

The sections below describe notebooks of the earlier exchange-format pipeline. They now live in
`notebooks/archive/` and are kept for reference; they do not read panel runs.

## Run explorer (archived: `notebooks/archive/stage_b_run_explorer.ipynb`)

Open **`notebooks/archive/stage_b_run_explorer.ipynb`** and select the project `.venv/bin/python` kernel.
The committed/saved outputs come from a real execution; the first cell reports its UTC
snapshot time. Restart & Run All to refresh while the batch is running.

Configure `RUN_NAME`, `FRONTEND_RUN_NAME`, `VIDEO_ID`, token prices and `EXPORT` in the first
code cell. Defaults point to `stage_b_sample50` and `stage_b_sample50_frontend`.
The folder name does not determine the sample size: the notebook reads the selection
manifest, shows unfinished and extra folders, and states every analysis denominator.

The notebook contains:

- Archived prompt/schema/codebook/raw-output browsers, with frontend/backend selection.
- Accepted issue/exchange annotations, roster, source windows, turn references and evidence.
- Coverage, active flags, historical failures, codebook/prompt hash checks.
- Code distributions, missingness reasons, descriptive correlations and uptake summaries.
- Shark feature rows; fixed leave-one-video-out ridge logistic AUC; within-video rankings;
  a sensitivity analysis excluding videos with any active flag.
- Token usage, frontend/backend/phase/video cost and successful-call latency.
- Optional derived CSV/JSON/Markdown exports in `runs/<run>/analysis/explorer/`.

It makes **no model API calls** and does not alter annotation artifacts or the registry.
It reads the experimental JSON layout through `analysis/stage_b_explorer.py`; it does not
require the older registry/parquet pipeline. Codebook files are read from run snapshots,
never regenerated from current drafts. AUC is exploratory and not annotation validation.

To execute from the repository root, preserving cell outputs:

```sh
.venv/bin/python scripts/analysis/execute_notebook.py notebooks/archive/stage_b_run_explorer.ipynb
```

Install dependencies with `.venv/bin/python -m pip install -r requirements.txt` if needed.
Widgets need a live Jupyter-compatible kernel; static previews still show executed tables,
figures and an example prompt/annotation. `EXPORT=True` writes derived analysis only.

## Where the former Python files went

- `within_pitch.py` → `analysis/within_pitch.py`: reusable estimators belong in a module.
- `pilot5_within_video.py`, `pilot5_reciprocity.py` → `scripts/analysis/`: standalone historical
  command-line analyses, not notebooks. Their calculation behavior is preserved.
- `qc_stage_b.py` → `qc_stage_b.ipynb`: this was already a Jupytext percent-format notebook.
  Retained for legacy registry/parquet runs; not appropriate for this experiment's JSON layout.

The new explorer incorporates the relevant inspection workflow without copying the pilot
scripts wholesale. In particular, it does not apply exhaustive five-video permutations to
this larger run or treat an unanswered probe as a poor-quality response.

## Project v2 analyses (archived)

`notebooks/archive/project_v2_analysis.ipynb` implements `docs/plans/project_plan_v2.md`: coverage,
construct/missingness summaries, optional standalone Stage A repeatability,
exposure-matched offer prediction, pitch-fixed-effect uptake/framing tests with
wild-cluster bootstrap inference, clean-video sensitivities, rare behaviors,
recorded costs, and prompt/annotation inspection.

Change `RUN_NAME` in the first code cell; `FRONTEND_RUN_NAME=None` infers the paired
frontend from experiment metadata. Change or disable `STAGE_A_RUN_NAME` separately
(it is a different corpus). Restart and Run All. Exchange-format experimental
runs and a selection manifest are required. No paid annotation calls are made.
Outputs go to `runs/<RUN_NAME>/analysis/project_v2/`. Bootstrap repetitions, seed,
and assumed token prices are configurable. Current results are exploratory and
model-coded; the notebook explicitly reports unavailable validation tasks.

Execute from the repository with:

```bash
.venv/bin/python scripts/analysis/execute_notebook.py notebooks/archive/project_v2_analysis.ipynb --timeout 600
```

## Concept bottleneck demonstration

The v2 notebook also contains an **exploratory frozen-extractor concept bottleneck**.
Existing video annotations supply prepared positivity and investor-specific mean
answer directness, checkable support, and substance. A small ridge logistic head
maps those named concepts and availability flags to on-air offer probabilities.
Whole pitches are held out; exposure-matched and probe-adjusted comparisons reuse
the H1 specifications. The strict concept head has no raw-video or identity bypass.

The section includes held-out metrics, unflagged sensitivity, a local prediction
breakdown, hypothetical concept edits, and a live Jupyter widget. Edits show model
sensitivity, not causal effects or actual human corrections. It does not measure
rapport development. A separate table proposes codebook/schema changes for
investor-specific affiliation, repair, and prefix-safe temporal analysis.

Implementation: `analysis/concept_bottleneck.py`. Saved tables and model metadata:
`runs/<RUN_NAME>/analysis/project_v2/concept_bottleneck/`. No new model API calls or
human labels are required to reproduce the current demonstration. Independent
human concept validation and correction experiments remain future work.

The concept section now also ranks the four concept values by held-out log-loss
improvement when each is retained versus removed, and when each is added alone to
availability flags. It reports pointwise conditional pitch-bootstrap intervals and
an unflagged sensitivity. Availability flags are kept fixed across comparisons.
The ranking is exploratory, not a causal importance ranking or an independent
feature-selection test. Implementation: `analysis/concept_ranking.py`; standalone
entry point: `scripts/analysis/rank_bottleneck_concepts.py`; exports are under
`runs/<RUN_NAME>/analysis/project_v2/concept_ranking/`.

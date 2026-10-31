# Analysis helpers

Read-only analysis of saved annotation artifacts. No API calls or registry writes.

- `stage_b_explorer.py`: experimental Stage B JSON → auditable tables and analysis cohorts.
  It imports read-only pipeline gates to check frozen maps and accepted reference windows.
- `auc.py`: shared fixed-ridge logistic and whole-video holdout functions, extracted unchanged
  from the pilot analysis so notebook and CLI results use the same estimator.
- `within_pitch.py`: historical conditional-logit / within-video estimators, moved from notebooks.

Interactive documents live in `notebooks/`; command-line entry points live in
`scripts/analysis/`. Experimental runs emit JSON rather than the production parquet layout;
the explorer adapter is the explicit artifact-reading boundary for those runs. It never
loads an API provider or modifies codebook snapshots.
- `project_v2.py`: exposure-matched offer prediction; conditional pitch-bootstrap
  uncertainty; within-pitch linear probability associations with restricted
  wild-cluster bootstrap-t inference; standalone Stage A repeatability. The
  specifications and limitations are documented in `docs/plans/project_plan_v2.md`.

- `hu_ma_funding.py`: Stage A on the Hu & Ma accelerator videos against the accelerator's decision,
  after Hu & Ma (2025) Tables II to IV: per-video means over replicates and rank scores, a Pitch
  Factor per method (first principal component, signed so positive dimensions load positively),
  ML logit with average marginal effects, Firth penalized logit with and without accelerator
  indicators, and an exact permutation test of the AUC; given Hu & Ma's own Pitch Factor and features
  from their code on the same videos (`data/hu-ma/pitch_factor.csv`), the agreement of each Stage A
  factor with theirs (Pearson with a Fisher interval, Spearman, Kendall, same-tercile share), per
  dimension, and by loadings. Entry point `scripts/analysis/analyze_hu_ma_stage_a.py`.

## Panel analysis (v3)

The export script `scripts/analysis/analyze_panel.py` (`compute` reads and computes, `write_analysis` writes the
files) and two notebooks in `notebooks/stage_B/` that call the same functions and keep no logic of their own:
`project_v3_panel_analysis.ipynb` (the pre-registered pitch-level test: outcomes, the cut, features, descriptives,
the ladder, verdicts, sensitivities, H9, figures) through `panel_report.py`, and `panel_run_explorer.ipynb` (the
instrument, one video's record with its prepared-presentation ratings and ask, pass-to-pass agreement, review status,
ranking, cost) through `panel_explorer.py`. `panel_bottleneck.prediction_breakdown` takes a held-out prediction
apart (model sensitivity, not a causal effect); `panel_features.history_table` lists, for one pitch, the turns its
features read and why the others are excluded.

### Pitch-level analysis (the pre-registered test)

One row per pitch, outcome `any_offer` (did any investor offer), evaluated leave-one-pitch-out. The design and its
decision rules are in `docs/plans/design docs/2026-10-19-pitch-level-refocus.md` (sections 4 and 6).

- `panel_dataset.py`: a panel run as tidy tables (the reviewed record of a video when there is one, else its
  primary pass), `pitch_outcome` (any offer; a human `pitch_override` wins over the record), `pitch_cut` and
  `eligible_turns` (the history cut, below), `answer_followups` (what the panel does after an answer), and
  `load_run(..., passes="both")`, which also reads the second annotation pass for the sensitivity that averages both.
  Since 2026-10-21 the next move and the reactions are read on turn order, not inside the model's exchange: the
  next move is the first investor turn after the answer (`no_reaction` when that turn is decision talk or there is
  none, `after_cut` when it lies at or after the cut), and `reaction_events.reaction_window` runs from the answer to
  the next founder answer or decision talk. A reaction that returns to an earlier answer is credited to the nearest
  preceding one. `answer_followups` adds `next_question_same_frame`, `further_questions_same_frame` and
  `next_question_same_concept`; its two exchange-based columns are kept for an unreported sensitivity only.
- `panel_features.py`: `pitch_features`, the named concept features of a pitch in the families of the ladder
  (control, exposure, stage_a, prepared, questions, answers, reactions, rapport_repair), each mean or share with its
  denominator `<name>_n` and a `<name>_missing` indicator; `feature_dictionary()` documents every column. The
  prepared presentation's ratings are the `pres_*` columns. The questions family has four sequence features over
  the investor questions before the cut in turn order: `q_sev_trend` (later half's mean severity minus the earlier
  half's), `q_pursuit_share` (consecutive pairs by the same asker with rising severity), `q_frame_switch_share`
  (consecutive promotion/prevention pairs whose frames differ) and `q_concept_entropy` (entropy of the concepts, in
  bits). No feature reads the exchange grouping: `demo_seconds` sums the `demonstrates` turns, and
  `n_social_exchanges` is gone (the archive computes its own). `exposure="rates"` gives the counts per minute of history.
- `panel_bottleneck.py`: the ridge-logistic heads over those named features only, and the machinery around them.
  `LADDER` is C0 (base rate), C1 (venture, ask, exposure), C2 (the primary Stage A rank score), C2p (the prepared
  presentation), C3q (questions alone), C3 (answers), C4 (reactions), C5 (rapport and repair); `LADDER_COMPARISONS`
  are the paired differences the spec names and `SENSITIVITY_LADDERS` the variants run with the same ladder.
  Everything learned (imputation, scaling, the ridge penalty chosen by cross-validation inside each training fold,
  the coefficients) is learned from the training pitches of the fold; intervals are a paired pitch-bootstrap of the
  fixed out-of-fold predictions (conditional: heads are not refit); the permutation test is at pitch level. Under 20
  pitches every result is labelled `demonstration only`. The pooled AUC (metrics, intervals, AUC deltas,
  permutation p) ranks the base-rate-adjusted held-out score, the held-out logit minus the logit of the training
  fold's base rate, which removes the shift of the base rate and reduces (does not remove) the leave-one-out bias
  against weak models; it is therefore not comparable with the
  AUCs of the v3-dev8 runs. Log loss and skill over the base rate are unchanged.
- `verdicts.py`: the pre-registered decision rules. H1' (C3 minus C2p: do the answers add to the prepared
  presentation) is supported by an AUC gain of at least +0.05 with an interval excluding 0 or a skill gain whose
  interval excludes 0; H8 (C3 minus C3q: do the answers add to the questions) by an AUC gain of at least +0.03 with
  an interval excluding 0. Otherwise "not distinguishable from zero". The AUC route is withheld only when either
  model's AUC interval lies wholly below 0.5; `auc_meaningful` and `pred_spread` (a near-constant model) are
  diagnostic columns and withhold nothing (ruling I3 of plan B's final review). Every verdict carries a power
  sentence.
- `answer_level.py`: H9, the answer-level test (does a more prepared answer change what the panel does next?). One
  row per answer before the cut; four nested models, each on the answers rated on all of its own predictors:
  `support` (checkable support, admitted uncertainty), `substance` (+ substance), `full` (+ coherence, context) and
  `responsiveness` (admitted uncertainty, a founder repair turn before the next panel turn). Outcomes, on turn
  order: `BINARY_OUTCOMES` (next accept, next challenge, next further question, next question in the same frame) by
  conditional logit stratified by pitch, `CONTINUOUS_OUTCOMES` (next question's severity, further questions in the
  same frame) by a linear model with a dummy per pitch; the two same-frame outcomes also enter the answered
  question's frame `q_promotion` (`FRAME_COVARIATE`) as a covariate with no direction. Pitch-bootstrap intervals,
  for all questions and for hard ones. `h9_sensitivity` runs the exchange-based `SENSITIVITY_OUTCOMES` with the same
  models; it is computed only on request and never reported. No decision rule: `h9_summary` claims a direction only where both
  intervals exclude zero on the same side and the predictor does not separate the outcome.
- `pitch_figures.py`: the nine figures and the supplementary S1 of the paper (`FIGURES`, `figure`, `caption`,
  `write_figures`), each drawn from a table the analysis already computed, as PNG and PDF with a caption file.
- `scripts/analysis/analyze_panel.py`: the entry point; writes `<run>/analysis/<analysis_id>/` with tables, figures,
  `manifest.json` and `report.md` (`pitch_report.report_markdown`: sections 0 to 9, from the pre-registered test and
  the sample through the ladder, verdicts, sensitivities, H9, H4 and H5, self-consistency and limits to provenance).
- `pitch_report.py`: `report.md`, and the helpers the figures and the notebook share (p-values as text, stars, which
  p a figure shows, the ROC models, calibration points). It imports nothing from the archive, so importing
  `analyze_panel` or `pitch_figures` loads no archive module.

      python scripts/analysis/analyze_panel.py --run runs/<run> --analysis-id pitch_level \
          [--video-ids ... | --selection selection_all100.json] [--penalty X] [--n-boot 1999] [--n-perm 500] \
          [--single-draws N] [--workers N] [--skip-h9] [--h9-exchange-outcomes [--h9-sensitivity-draws 499]] \
          [--with-shark-level] [--force] [--figure-width single|double]

  `--video-ids` and `--selection` are alternatives; `--penalty` fixes the ridge penalty of every head instead of
  choosing it inside each training fold; `--n-perm 0` skips the permutation test; `--single-draws` sets the
  bootstrap draws behind the single-feature p-values; `--workers` splits H9's bootstrap draws over processes
  (default: every core but one) and changes no number; `--skip-h9` is a quick run; `--h9-exchange-outcomes` also
  computes H9's exchange-based outcomes as a sensitivity at `--h9-sensitivity-draws` draws (default 499), written to
  `h9_exchange_sensitivity.csv` and named in `report.md` by one sentence with no result (it cannot be combined with
  `--skip-h9`); `--with-shark-level` reruns the
  archived design (below); `--force` replaces an existing analysis of that id.
- `panel_explorer.human_validation(run)`: the paper's validation table (`<run>/review/paper_table.json`, written by
  `review_panel.py summary --paper-table`) as a DataFrame, or `None` before it exists.
- `panel_report.py`: the analysis notebook's tables, captions and plain readings of a pitch-level `Analysis`
  (`compute` and `export` call `analyze_panel`); it reads nothing from the archive.
- `panel_figures.py`: the figure style under its older name, re-exported from `pitch_figures` for
  `human_agreement.py` and the experiment scripts; the investor-row figures were removed in plan B task 9.

**Leakage rule.** The model saw each whole video, decisions included. Leakage is prevented here: a pitch row's
features use only turns strictly before the pitch's cut, which is the first decision of any kind by any investor
(never "the first offer", which would let the length of the history tell the outcome). The turns are read through
`eligible_turns`, which also drops every `decides` and `negotiates` turn and every turn of a negotiation exchange.
When both passes are averaged (the sensitivity of `load_run(..., passes="both")`), each pass is cut at its own first
decision. The descriptive counts (descriptives.json, H4, H5) cover the whole encounter and are not predictors.
`tests/test_panel_features.py` holds the leakage test: truncating a record at a pitch's cut and deleting all
decisions leaves that row's features unchanged.

### Archive (`analysis/archive/`)

The retired shark-level design (one row per pitch and investor, the O2 outcome) and its experiments. Kept importable
and runnable with the numbers they gave. Nothing in the pitch-level path imports them (importing `analyze_panel`,
`pitch_figures` or `panel_report` loads no archive module). They run only behind `analyze_panel.py
--with-shark-level` (and the notebook's `WITH_SHARK_LEVEL`) and from the two experiment scripts.

- `shark_level_dataset.py`: each investor's outcome (`ever_offer`), cut (that investor's own first offer, joined
  offer or exit) and history.
- `shark_level_features.py`: the shark-level feature rows, each stopping at the investor's own cut, with the
  `decision_history`, own-versus-others and `rx_own_*`/`rx_pan_*` families and the `guard_turns` sensitivity.
- `shark_level_models.py`: the B-family B0-B5d, the draft primary set P0-P5d, the reaction-trajectory ladders, the
  removal ranking and the pooled-versus-attributed comparison, run on the pitch-level machinery of `panel_bottleneck`.
- `within_pitch_eval.py`: within-pitch and between-pitch AUC with pitch-bootstrap intervals and paired deltas, and
  the conditional-logit check; they say whether a model ranks the sharks of one pitch, not only the pitches.
- `shark_level.py`: `run(data, run_root, ...)`, the `--with-shark-level` path: the B ladder on shark rows with its
  within- and between-pitch AUC, the identity baseline (`common_cut_rows`, the I and N ladders; the investor prior
  reads other pitches' outcomes and is a sensitivity only) and the F1 numbers.
- `legacy_panel_analysis.py`, the investor-row analysis the notebook computed before the refocus, was deleted in
  plan B task 9, when the notebook moved to the pitch-level analysis.
- `scripts/experiments/reaction_trajectory_test.py`, `scripts/experiments/identity_baseline_test.py`: the two shark-
  level experiments (CLI unchanged; they import from the archive). Their tests are in `tests/archive/`.

At shark level the leak is a shark's own `accepts` before its decision: under v3-dev9 those turns are history for the
shark's own row (the former `expresses_interest` exclusion is gone), which is a decision tell, and the pitch-level
cut has no such exclusion because it ends at the first decision of anyone. The shark-level rule is pinned by
`test_focal_accepts_before_own_decision_is_history_until_plan_b` in `tests/archive/test_shark_level_features.py`.

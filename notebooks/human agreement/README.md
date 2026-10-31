# Human agreement notebooks

| Notebook | What it is for |
|---|---|
| [`training_agreement.ipynb`](training_agreement.ipynb) | **How far the human coders agree on the training videos**, with each other (the ceiling), with the model, and the model with itself, per field class and per code with the validation table's figures for each kind of code (raw agreement, alpha, AC1/AC2, bands, flags), read against a bar of alpha 0.61 (a guide, not a gate); the three Stage A ratings item by item; who chose what on the categorical codes; the list of cards to talk through in calibration; and the clip session's pair judgements against the model's rank scores (section H). **Pending execution** (see below). |

It reads the coders' filled training sheets from `runs/<run>/review/training/training_results/` (set `RESULTS_DIR`
to use another folder): the full-video sheets (`<video_id>__<coder>.csv`) and, when the training had a clip session
(`review_panel.py training --clips`), each coder's clip-session sheet (`clips__<coder>.csv`). It imports them into a
throwaway copy of the run, exactly as `scripts/review_panel.py calibrate` does, so the run itself is never changed.
The helpers are in `analysis/human_agreement.py`. Design: `docs/plans/human_validation/2026-10-08-validation-specification.md`
and `docs/plans/design docs/2026-10-19-pitch-level-refocus.md` §5.

With two training videos each code has only a handful of rows (a Stage A item has two) and the pairs are one pair
shown three times: alphas swing widely and carry no interval. Read the class rows, the plain agreement and the card
list first.

**Pending execution.** The notebook points at `full-corpus-100` and is committed without outputs: it is run
once the coders have filled the `v3-dev9` training sheets. The October 19 training round (AV, MX) coded `v3-dev8`
records, which the `v3-dev9` tooling refuses; its executed notebook is kept in the history (commit `09927ec`) as the
`v3-dev8` calibration record, with the sheets under `runs/stage_b_panel_v3dev8/review/training/`.

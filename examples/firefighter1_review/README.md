# Firefighter1: a human annotation example

Open [human1's completed 32-card sheet](pages/human1__completed.html) to inspect the coding interface and their answers. The [blank sheet](pages/human1__tasks.html) remains available. Watch the [video](https://www.youtube.com/watch?v=dxsV1p3Hue0), or choose a local copy in the sheet to use timestamp playback.

The original submitted files are preserved without edits: [Q&A and decision clock](answers/st_8a6fd25840__human1.csv) and [presentation and investment ask](answers/clips__human1.csv). The completed page embeds those answers and disables editing; it needs no browser draft or CSV import.

The fixed sample is recorded in [manifest.json](manifest.json). The exports use the project's existing review formats. Model values are held in the research manifests for comparison, but are not embedded in the blank HTML.

[Demo.ipynb](../../Demo.ipynb) imports the answers into a temporary run and recalculates agreement through `review_apply.pooled_summary(reporting=...)` and `pipeline/reliability/agreement.py`. The [compact table](derived/agreement_compact.csv), [all item comparisons](derived/item_comparison.csv), and [full per-kind statistics](derived/reporting.json) are also saved. Original research records and prediction inputs are unchanged. This is one author's worked example, with four comparisons per Q&A code, not a corpus-level reliability estimate. See [the guide](../../docs/demo.md) for commands and interpretation.

The [first-decision comparison](derived/decision_turn.json) uses same-turn agreement: human1’s 5:52 and the model’s 5:40 both identify turn 31 (5:40–5:55), so they agree. Their 12-second difference is retained as a timing diagnostic. This revised rule and its provenance are recorded alongside the result.

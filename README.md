# Measuring Catch and Toss and Its Impact on Entrepreneurial Pitching

**How do the exchanges between entrepreneurs and investors reveal preparedness and inform investment judgments beyond the prepared pitch?**

This repository accompanies our submission to *Organization Science* special issue: AI-Enabled Frontiers in Organizational Science. We propose an AI-enabled method for empirically addressing the long-standing question of what makes entrepreneurial pitches effective. Specifically, we seek to address the gap identified by Kalvalpalle et al 2024’s meta-analysis of the entrepreneurial pitch literature that past research on entrepreneurial pitching has almost always focused on either the entrepreneur or the investor but rarely the interaction between the two. Here, we examine entrepreneurial pitch videos where entrepreneurs and investors interact to study how their back-and-forth exchanges convince investors of the merits of the entrepreneurs’ ideas.

## How to: start with the demo

Open [Demo.ipynb](Demo.ipynb) in a notebook viewer to inspect the saved tables and figures. To rerun it locally, follow [Run the demo](#run-the-demo) below.

The notebook follows **Firefighter1** through four sections:

1. **Annotating a single episode:** the included video, codebook, prompt, saved JSON annotation record, and timeline.
2. **Validate with a human coder's annotations:** the completed Firefighter1 coding sheet and agreement between the human and Gemini annotations.
3. **From annotations to concepts and an offer prediction:** a worked answer-family example, all **46 model concepts** with their source annotations and calculations, and a prediction from a model trained on the **other 99 episodes**, compared with Firefighter1's outcome.
4. **Across the full 100 episodes:** summary statistics, concept distributions, validation by **two human coders on 10 sampled episodes**, and predictive comparisons across concept families.

The demo uses saved records and needs **no API key**. It recalculates the Firefighter1 human agreement, rebuilds the concepts, and fits the example's held-out prediction. The corpus validation and predictive comparisons use saved research results. Panel B compares pair-ranking performance with **50% chance**, rather than reporting classification accuracy.

The Firefighter1 MP4 is included and embedded in the notebook. If playback fails, use the notebook's [YouTube link](https://www.youtube.com/watch?v=dxsV1p3Hue0). Download the [completed coding sheet (HTML)](examples/firefighter1_review/pages/human1__completed.html) and open it in a browser to see the full interface and submitted answers. The [blank sheet](examples/firefighter1_review/pages/human1__tasks.html) is also available.

## How to run the demo

Download or clone this repository, then open a terminal in its root folder. Use **Python 3.11 or newer**.

On macOS or Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows (PowerShell):

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Open the repository folder in VS Code with notebook support, open `Demo.ipynb`, select the `.venv` Python environment as the kernel, and choose **Restart Kernel and Run All**. Run from the repository root so the relative paths resolve correctly. Installing dependencies and watching YouTube require internet; the saved-record calculations run offline.

The setup cell's `VIDEO_PATH` points to the included MP4. If your copy lacks that file, set `VIDEO_PATH = ""` to display the YouTube link instead. The human-validation cell opens the completed coding sheet in your browser by default; set `OPEN_REVIEW_SHEET = None` to keep only the notebook preview, or `"blank"` to open the blank sheet.

You can also execute the notebook and save its outputs from a terminal:

```sh
.venv/bin/python scripts/analysis/execute_notebook.py Demo.ipynb
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`.

## Inspect the evidence and code

The reported run, `full-corpus-100`, contains the saved annotation records and analysis outputs for the 100-episode corpus. The included Firefighter1 video provides the worked example.

| To inspect | Open |
|---|---|
| Code definitions | [Interaction codebook](pipeline/codebooks/stage_b_panel.md), [pitch delivery](pipeline/codebooks/stage_a_perceived_delivery.md), [venture context](pipeline/codebooks/venture_context.md) |
| Firefighter1's archived instrument and records | [Example run](runs/full-corpus-100/st_8a6fd25840), including `bundle/`, `prompts/`, and `results/` |
| Agreement between Gemini's two annotation passes | [Code and outcome comparisons](runs/full-corpus-100/analysis/all100/self_consistency.csv), [turn-function comparisons](runs/full-corpus-100/analysis/all100/self_consistency_turn_functions.csv) |
| Firefighter1 human annotations and comparisons | [Completed coding sheet](examples/firefighter1_review/pages/human1__completed.html), [submitted answers](examples/firefighter1_review/answers), [calculated agreement](examples/firefighter1_review/derived) |
| Two-coder validation on 10 episodes | [Validation table](runs/full-corpus-100/review/agreement/validation_table.md), [machine-readable results](runs/full-corpus-100/review/agreement/validation_table.json), [agreement figure](runs/full-corpus-100/review/agreement/agreement_by_code.png) |
| Concepts and held-out predictions | [100-episode analysis](runs/full-corpus-100/analysis/all100), including `features.csv`, `predictions.csv`, `metrics.csv`, and `manifest.json` |
| Concept calculations and prediction | [Feature construction](analysis/panel_features.py), [model fitting](analysis/panel_bottleneck.py), [demo assembly](analysis/demo.py) |
| Further inspection and analysis | [Record explorer](notebooks/stage_B/panel_run_explorer.ipynb), [full analysis notebook](notebooks/stage_B/project_v3_panel_analysis.ipynb) |

The demo checks saved inputs against the [release manifest](examples/demo_release.json) and the analysis manifest, then compares regenerated concepts with the saved export. The Firefighter1 coding sheet is a worked example; the separate 10-episode sample supports the corpus human-validation results shown later in the notebook.

The prediction concerns whether any investor **makes an offer**. Acceptance and closing a deal are different outcomes. The findings concern edited television pitches, and agreement between repeated model calls measures consistency rather than establishing annotation validity.

## Extend the workflow

To use this pipeline for new videos or codebooks, start with the [annotation runner](scripts/experiments/run_panel_pilot.py), its [configuration](configs/stage_b_panel.yaml), and the [demo helpers](analysis/demo.py). The [custom codebook example](examples/demo_custom_instrument/codebooks/stage_b_panel.md) illustrates a definition change that retains the existing code IDs and scales. New Gemini annotations require configured API access and should be saved in a separate run.

Changing code IDs, scales, or outcomes also requires corresponding changes to schemas, concept construction, and validation. See the [analysis documentation](analysis/README.md) and [analysis commands](scripts/analysis/README.md) for the implementation and entry points.

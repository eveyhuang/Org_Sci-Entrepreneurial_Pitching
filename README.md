# Measuring Catch and Toss and Its Impact on Entrepreneurial Pitching

**How do the exchanges between entrepreneurs and investors reveal preparedness and inform investment judgments beyond the prepared pitch?**

This repository accompanies our submission to *Organization Science* special issue: AI-Enabled Frontiers in Organizational Science. We propose an AI-enabled method for empirically addressing the long-standing question of what makes entrepreneurial pitches effective. Specifically, we seek to address the gap identified by Kalvalpalle et al 2024’s meta-analysis of the entrepreneurial pitch literature that past research on entrepreneurial pitching has almost always focused on either the entrepreneur or the investor but rarely the interaction between the two. Here, we examine entrepreneurial pitch videos where entrepreneurs and investors interact to study how their back-and-forth exchanges convince investors of the merits of the entrepreneurs’ ideas.

<a id="run-the-demo"></a>

## Step 1: Set up your environment

Download or clone this repository, then open a terminal in its root folder. Use **Python 3.11 or newer**; the commands below explicitly select Python 3.12. If you use another supported version, substitute its version number. The command `python3` may select an older installation, even when a newer Python is installed.

On macOS or Linux:

```sh
python3.12 --version
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import sys, ipykernel; print(sys.version)"
```

On Windows (PowerShell):

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -c "import sys, ipykernel; print(sys.version)"
```

## Step 2: Run the Demo.ipynb

Open the downloaded repository folder in VS Code, then click **[Open Demo.ipynb](Demo.ipynb)**. Select the repository's `.venv` kernel showing **Python 3.11 or newer**, and choose **Restart Kernel and Run All**. Run from the repository root so the relative paths resolve correctly. The saved-record calculations run offline; watching YouTube requires internet.

### Troubleshooting: If `.venv` is missing from the kernel list

Installing the Python packages does not install VS Code's Python extension. If `.venv` is not listed when you try to choose the venv kernel in the Demo.ipynb:

1. Click **Install Python Extension** in the kernel picker and install **Python by Microsoft**. If that option is absent, open VS Code's Extensions view and find it there. Make sure **Jupyter by Microsoft** is also installed and enabled.
2. Press **⌘⇧P** on macOS or **Ctrl+Shift+P** on Windows/Linux, then run **Developer: Reload Window**.
3. Open [Demo.ipynb](Demo.ipynb), click **Select Kernel → Select Another Kernel… → Python Environments…**. If the kernel-source menu is already open, choose **Python Environments…** directly.
4. Select this repository's **`.venv`**, showing **Python 3.11 or newer**.

If it is still missing, run **Python: Select Interpreter** from the Command Palette, choose **Enter interpreter path → Find**, and select `.venv/bin/python` inside this repository (Windows: `.venv\Scripts\python.exe`). On macOS, press **⌘⇧.** in the file picker to reveal the hidden `.venv` folder. Then return to **Select Kernel → Python Environments…** and select that environment.

You can also execute the notebook and save its outputs from a terminal:

```sh
.venv/bin/python scripts/analysis/execute_notebook.py Demo.ipynb
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`.

### The demo includes four sections that walks you through the whole pipeline and our results:

1. **Annotating a single episode:** you can watch an specific episode from Sharktank, inspect the codebook and prompt, the annotation record returned by Gemini on this video, and the resulted timeline (as shown in Figure 1).

2. **Validate with a human coder's annotations:** The demo includes the original answer and interface from a human coder’s annotation on this episode, and calculate the agreement between Gemini and the human coder.

3. **From annotations to concepts and an offer prediction:** Using the annotations from the example episode, the demo walks through a concrete example on how these annotations become a concept, “founder answer”, used in the concept bottleneck model, and shows the full 46 model concepts. Then the demo shows the prediction of funding outcome from our model trained on the other 99 episodes and compare with the actual outcome.

4. **Across the full 100 episodes:** Using the full corpus of 100 episodes, the demo then shows the summary statistics and distributions of annotations and concepts, the full results from human validation (by two human coders) on sampled 10 episodes to reproduce Figure 3 in proposal. Lastly, it compares predictive performance across concept families and reproduces Figure 2.


## Inspect the evidence and code

The reported run, `full-corpus-100`, contains the saved annotation records and analysis outputs for the 100-episode corpus. The included Firefighter1 video provides the worked example. The saved analysis does not need the other source videos. To rerun annotation or export clips for the full corpus, supply those videos under `data/shark tank/downloads/`, using the filenames in `selection_all100.json`, and run the commands from the repository root. Archived source-video paths are relative to that root.

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





"""The clip session of the human validation: each coder watches the ten sampled videos' prepared presentations, one
at a time, codes each one's seven cards (the Stage A delivery items, the prepared presentation's codes but PRES-CTX,
the ask), and then judges 25 pairs of them on the Stage A items: which shows more? Outcome-blind throughout.

Since 2026-10-21 no clip is cut (the researcher's decisions): the page plays each pitch's full video, found in the
coder's folder by its `experiment.json` source base name, with its prepared window (every span) on a ▶ button that
seeks to the window's start, and the coders are trusted to watch only that part. "Wider context" (PRES-CTX) cannot be
judged by coders, so it is left off their cards (the model keeps it), and the count of checkable claims (PRES-EVID) is
answered in bands (review_items.EVID_BANDS), compared with the model's count banded the same way. The session keeps
its name (the code, the `clips` page mode, `clips-export`, `clips-import`, clips.csv), so earlier exports still read;
a manifest written before then names clip files (`clip_files`), which a page built from it still loads.

The pairs are a balanced design (`pair_design`: every video compared with five others, seeded, never chosen by the
model's scores), and they test the model's Plackett-Luce rank scores (`<run>/ranking/scores.json`) with no further
model call: the scores imply an answer to every pair. A coder's own pair choices give that coder's Bradley-Terry
scores (`bradley_terry`, the pair form of the same likelihood) and `pair_agreement` sets them beside the model's.

Layout, under <run>/review/clips/: export_manifest.json (the design, the rows and the model's answers), clips.csv
(the session's blank sheet), pages/<coder>.html, pairs_log.jsonl (every imported pair judgement) and <coder>_scores.json (a coder's Bradley-Terry scores). The per-video cards are imported into each
video's own review log, as a full-video sheet's are. No I/O beyond these files; no provider call anywhere."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
from pathlib import Path

import numpy as np

from pipeline.reliability.agreement import MIN_DEFINED_SHARE
from pipeline.schemas.stage_b_panel import PanelRecord
from pipeline.stage_a.rank_aggregate import plackett_luce
from pipeline.stage_a.rank_design import Batch, design_summary, make_design
from pipeline.stage_b.panel_books import PanelBooks
from pipeline.ingest import sha256_file
from pipeline.stage_b.panel_pipeline import _read, model_pass, prepared_window
from pipeline.stage_b.review_apply import MIN_CLUSTER_VIDEOS
from pipeline.stage_b.review_items import ASK, PAIR, SHEET_COLUMNS, allowed_values, canonical_full, read_value
from pipeline.stage_b.review_labels import PAIR_VALUES

CLIPS_DIR = "clips"                         # under <run>/review/
MANIFEST = "export_manifest.json"
SHEET = "clips.csv"
PAGES = "pages"
PAIRS_LOG = "pairs_log.jsonl"
SCORES_SOURCE = "ranking/scores.json"       # the model's rank scores, relative to the run
PER_VIDEO = 5                               # comparisons per video: 10 videos × 5 / 2 = 25 pairs
TRAINING_PER_VIDEO = 3                      # the training session: two clips, so their one pair three times
PAIRS_ID = "pairs"                          # the page's entry that holds the pairs (a video id the session never uses)
CLOSE = "[close]"                           # how a page writes the "close" tick into a pair row's note
DESIGN = re.compile(r"\bdesign ([0-9a-f]{12})\b")     # how a sheet's context cells name the session's design
TEN = 10                                    # the session the labels' pairs intro is worded for ("all ten ...")
# codes the model keeps that coders do not judge: "wider context" cannot be judged by a coder (2026-10-21)
MODEL_ONLY_CODES = ("PRES-CTX",)
SAMPLED = "sampled"
SHEET_ENCODING = "utf-8-sig"
N_BOOT = 1999
MODEL_COLUMNS = ("priority", "primary", "replicate", "reasons", "selected_as")      # blank on the coders' sheet


class ClipsError(ValueError):
    pass


def clip_file(video_id: str) -> str:
    """The name a video's prepared clip had in the coder's folder before 2026-10-21; kept to read old manifests."""
    return f"{video_id}__prepared.mp4"


def video_file(vdir: Path) -> str:
    """The name of a video's full file in the coder's folder: its `experiment.json` source base name, as the
    full-video page and `copy-videos` name it (`<video_id>.mp4` when the experiment names no source)."""
    vdir = Path(vdir)
    experiment = _read(vdir / "experiment.json") if (vdir / "experiment.json").exists() else {}
    return os.path.basename(experiment.get("source") or f"{vdir.name}.mp4")


def window_text(record: PanelRecord) -> str:
    """A record's prepared window as the page shows it: every span, "1:05-2:40, 3:10-3:30" ("" with none)."""
    return ", ".join(prepared_window(record))


def coder_codes(books: PanelBooks) -> list[str]:
    """The prepared presentation's codes a coder judges: the books' codes but MODEL_ONLY_CODES."""
    return [e.id for e in books.prepared_codes() if e.id not in MODEL_ONLY_CODES]


def session_paths(books: PanelBooks) -> list[str]:
    """The paths of a video's seven cards in the session, in the order the page asks them: the Stage A items, the
    prepared codes a coder judges, the ask. A video's full-video sheet must not carry any of them."""
    return ([f"stage_a:{c}" for c in books.stage_a_ids()] + [f"prep_pres:{code}" for code in coder_codes(books)]
            + [f"venture:{ASK}"])


def ranking_command(run: Path) -> str:
    return f"scripts/experiments/run_stage_a_ranking.py --run {run} --appearances 3 --execute"


def pair_design(video_ids: list[str], seed: int, per_video: int = PER_VIDEO,
                allow_repeats: bool = False) -> list[Batch]:
    """The balanced pair design: batches of two (b01, b02, ...) in which every video appears `per_video` times and no
    pair repeats. Left is `batch.video_ids[0]`, right `[1]`; the seeded shuffle places them. `allow_repeats` is for
    the training session, whose two clips make one pair that can only be shown again (sides seeded each time)."""
    batches = make_design(sorted(video_ids), batch_size=2, appearances=per_video, seed=seed)
    summary = design_summary(batches)
    if (summary["repeated_pairs"] and not allow_repeats) or set(summary["batch_sizes"]) != {2}:
        raise ClipsError(f"seed {seed} gives no balanced pair design of {len(video_ids)} videos at {per_video} each "
                         f"({summary['repeated_pairs']} repeated pairs); choose another seed")
    return batches


def pairs_intro(text: str, n_clips: int) -> str:
    """The pairs' intro for a session of `n_clips`: the labels word it for ten ("all ten prepared presentations",
    earlier "all ten clips"); another count is written in ({clips} in the text, else the number in place of the
    word after "all")."""
    if "{clips}" in text:
        return text.replace("{clips}", str(n_clips))
    return text if n_clips == TEN else re.sub(r"\ball \w+ (?=prepared presentations\b|clips\b)", f"all {n_clips} ",
                                               text, count=1)


def model_scores(run: Path, video_ids: list[str], codes: list[str]) -> dict[str, dict[str, float]]:
    """{code: {video_id: pl_score}} from the run's ranking. Refused, naming the command that makes it, when the file,
    an item or a session video is missing: a pair without the model's side is never exported."""
    path = Path(run) / SCORES_SOURCE
    command = ranking_command(Path(run))
    if not path.exists():
        raise ClipsError(f"{path} not found: the model's side of a pair comes from the Stage A ranking; run {command}")
    scores = _read(path)
    missing = [code for code in codes if not isinstance((scores.get(code) or {}).get("pl_score"), dict)]
    if missing:
        raise ClipsError(f"{path} has no rank scores for {', '.join(missing)} (the instrument's Stage A items); "
                         f"run {command}")
    out = {}
    for code in codes:
        pl = scores[code]["pl_score"]
        absent = [v for v in video_ids if v not in pl]
        if absent:
            raise ClipsError(f"{path} has no {code} score for {', '.join(absent)}; run {command}")
        out[code] = {v: float(pl[v]) for v in video_ids}
    return out


def model_side(scores: dict[str, float], left: str, right: str) -> tuple[str | None, float]:
    """The model's answer to a pair (the clip with the higher rank score) and the score gap. Equal scores are no
    answer (None): the model prefers neither clip, so the pair is left out of coder–model agreement."""
    if scores[left] == scores[right]:
        return None, 0.0
    return ("left" if scores[left] > scores[right] else "right"), abs(scores[left] - scores[right])


def design_id(seed: int, video_ids: list[str], batches: list[Batch], codes: list[str],
              cards: list[str] | None = None) -> str:
    """The session's identity: a digest of the seed, the videos, every pair with its sides and items, and the paths
    of each video's cards (`cards`, since Task 7c: a session whose cards changed, PRES-CTX dropped, is another
    session). A sheet, a page's stored answers and a logged pair answer carry it, so an answer is only ever read
    against its own session."""
    design = {"seed": seed, "videos": sorted(video_ids), "codes": list(codes),
              "pairs": [[b.batch_id, *b.video_ids] for b in batches]}
    if cards is not None:
        design["cards"] = list(cards)
    text = json.dumps(design, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def session_order(video_ids: list[str], coder: str, seed: int) -> list[str]:
    """A coder's own seeded order of the clips (as `review_panel.py page` orders a coder's videos)."""
    order = sorted(video_ids)
    random.Random(f"{seed}:{coder}").shuffle(order)
    return order


def drawn_pass(run: Path, video_id: str, manifest: dict | None = None) -> str:
    """The pass a video's cards are drawn from: the manifest's record of it (`passes`) when given, an older manifest's
    primary, else the pass the analysis reads (panel_pipeline.model_pass; final review I1)."""
    if manifest is not None:
        return (manifest.get("passes") or {}).get(video_id) or "primary"
    return model_pass(Path(run) / video_id)


def record_of(run: Path, video_id: str, books: PanelBooks, manifest: dict | None = None) -> PanelRecord:
    """A video's canonical record of the pass its cards are drawn from (`drawn_pass`)."""
    path = Path(run) / video_id / "results" / f"{drawn_pass(run, video_id, manifest)}.json"
    return canonical_full(PanelRecord.model_validate(_read(path)["record"]), books)



def _video_rows(run: Path, video_id: str, books: PanelBooks) -> list[dict]:
    """A video's seven cards, in the order the page asks them: the Stage A items, the prepared codes a coder judges
    (`coder_codes`: not PRES-CTX), the ask. Their values are a coder sheet's (PRES-EVID in bands); `primary` and
    `replicate` hold what the passes recorded (PRES-EVID's count). The context names the full video and its window."""
    vdir = Path(run) / video_id
    record = record_of(run, video_id, books)
    other = "replicate" if drawn_pass(run, video_id) == "primary" else "primary"
    replicate_path = vdir / "results" / f"{other}.json"
    replicate = (canonical_full(PanelRecord.model_validate(_read(replicate_path)["record"]), books)
                 if replicate_path.exists() else None)
    d = record.model_dump(mode="json")
    r = replicate.model_dump(mode="json") if replicate is not None else None
    paths = session_paths(books)
    where = f"video {video_id}: {video_file(vdir)}, prepared window {window_text(record) or 'none'}"
    rows = []
    for path in paths:
        allowed = allowed_values(path, record, books, coder=True)
        try:
            second = read_value(r, path) if r is not None else None
        except ValueError:
            second = None
        rows.append({"video_id": video_id, "path": path, "field_class": path.split(":", 1)[0],
                     "context": where, "primary": read_value(d, path),
                     "replicate": second, "allowed_values": list(allowed) if allowed else None,
                     "selected_as": SAMPLED})
    return rows


def _pair_rows(batches: list[Batch], scores: dict[str, dict[str, float]], codes: list[str], files: dict[str, str],
               windows: dict[str, str]) -> list[dict]:
    rows = []
    for batch in batches:
        left, right = batch.video_ids
        shown = {v: f"{files[v]} ({windows[v] or 'no window'})" for v in (left, right)}
        for code in codes:
            answer, gap = model_side(scores[code], left, right)
            rows.append({"video_id": None, "path": f"{PAIR}:{batch.batch_id}/{code}", "field_class": PAIR,
                         "context": f"pair {batch.batch_id}: left {shown[left]}, right {shown[right]}",
                         "primary": answer, "replicate": None, "allowed_values": list(PAIR_VALUES),
                         "selected_as": SAMPLED, "batch_id": batch.batch_id, "code": code, "left": left,
                         "right": right, "gap": gap, "tie": answer is None})
    return rows


def write_sheet(path: Path, rows: list[dict]) -> None:
    """The session's blank sheet: the 15 columns, one row per card, no model value (MODEL_COLUMNS blank)."""
    with open(path, "w", newline="", encoding=SHEET_ENCODING) as fh:
        writer = csv.writer(fh)
        writer.writerow(SHEET_COLUMNS)
        for row in rows:
            cells = {**dict.fromkeys(SHEET_COLUMNS, ""), "row": row["row"], "path": row["path"],
                     "field_class": row["field_class"], "context": row["context"],
                     "allowed_values": " | ".join(row["allowed_values"] or ())}
            writer.writerow([" ".join(str(cells[c]).split()) for c in SHEET_COLUMNS])


def session_per_video(n_videos: int) -> int:
    """Comparisons per video in a session of `n_videos`: PER_VIDEO (five), or every other video when there are fewer
    than six (`review_panel.py session-export` on two videos makes their one pair). The design pairs every video
    once per round, so the number of videos must be even."""
    if n_videos < 2 or n_videos % 2:
        raise ClipsError(f"a session's pair design takes an even number of videos, at least two (each round pairs "
                         f"every video once); got {n_videos}")
    return min(PER_VIDEO, n_videos - 1)


def build_manifest(run: Path, video_ids: list[str], coders: list[str], seed: int, books: PanelBooks,
                   release_id: str, primary_sha256: dict[str, str], training: bool = False,
                   per_video: int | None = None, include_pairs: bool = True) -> dict:
    """The session's manifest (no file written): the design, the video order per coder, every card's row with the
    model's side (a pair's from the rank scores, with the gap), and each video's full file (`video_files`) and
    prepared window (`windows`). A `training` session (`review_panel.py training --clips`) has each video in
    TRAINING_PER_VIDEO comparisons, repeats allowed; `per_video` sets another count for a session that is not
    training (default PER_VIDEO). With include_pairs=False, only presentation cards are built and
    ranking scores are not required; existing session defaults remain unchanged."""
    video_ids = sorted(video_ids)
    if PAIRS_ID in video_ids:
        raise ClipsError(f"a video may not be called {PAIRS_ID!r}: the page keeps the pairs under that name")
    codes = list(books.stage_a_ids())
    # A worked example can use the same presentation cards without a ranking session.
    scores = model_scores(run, video_ids, codes) if include_pairs else {}
    per_video = (TRAINING_PER_VIDEO if training else (per_video or PER_VIDEO)) if include_pairs else 0
    batches = pair_design(video_ids, seed, per_video, allow_repeats=training) if include_pairs else []
    files = {v: video_file(Path(run) / v) for v in video_ids}
    windows = {v: window_text(record_of(run, v, books)) for v in video_ids}
    passes = {v: drawn_pass(run, v) for v in video_ids}
    video_rows = [row for v in video_ids for row in _video_rows(run, v, books)]
    cards = list(dict.fromkeys(row["path"] for row in video_rows))
    design = design_id(seed, video_ids, batches, codes, cards)
    rows = video_rows + _pair_rows(batches, scores, codes, files, windows)
    for number, row in enumerate(rows, 1):
        row["row"] = number
        row["context"] = f"design {design}: {row['context']}"       # the sheet names the session it belongs to
    return {"seed": seed, "design_id": design, "videos": video_ids, "coders": list(coders),
            "order": {coder: session_order(video_ids, coder, seed) for coder in coders},
            "rows": rows,
            "pairs": [{"batch_id": b.batch_id, "left": b.video_ids[0], "right": b.video_ids[1], "codes": codes}
                      for b in batches],
            "video_files": files, "windows": windows,
            "scores_source": SCORES_SOURCE if include_pairs else None,
            "release_id": release_id, "primary_sha256": primary_sha256,
            # the pass each video's cards and window were drawn from (the one the analysis reads), and its sha256
            "passes": passes,
            "pass_sha256": {v: sha256_file(Path(run) / v / "results" / f"{passes[v]}.json") for v in video_ids},
            "per_video": per_video, "training": training}


def export_session(run: Path, video_ids, coders, seed: int, books: PanelBooks, labels, out: Path,
                   with_codebook: bool = True, primary_sha256: dict[str, str] | None = None,
                   training: bool = False, pages: bool = True, per_video: int | None = None) -> dict:
    """Write the session: `out`/export_manifest.json, `out`/clips.csv and `out`/pages/<coder>.html (mode clips,
    each coder's own video order, the pairs in b01..b25 order, the codebook laid out unless `with_codebook` is
    False). Returns the manifest. Nothing is written when the rank scores are missing (ClipsError). A `training`
    session says so on its pages (see `build_manifest`). `pages=False` writes no clip-session page: the coders'
    pages are then the task pages of `review_panel.py session-export` (or `training --clips`), which write the same
    sheet."""
    from pipeline.instrument import load_release
    from pipeline.stage_b.review_codebook import codebook_model
    from pipeline.stage_b.review_page import clips_page_model, on_sheet, render_page
    release = load_release()
    manifest = build_manifest(Path(run), list(video_ids), list(coders), seed, books, release.release_id,
                              primary_sha256 or {}, training=training, per_video=per_video)
    written = {}
    for coder in (manifest["coders"] if pages else []):
        model = clips_page_model(Path(run), manifest, coder, books, labels, training=training)
        if with_codebook:
            model["codebook"] = codebook_model(release, books, labels, on_sheet(model))
        written[coder] = render_page(model)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if written:
        (out / PAGES).mkdir(exist_ok=True)
    write_sheet(out / SHEET, manifest["rows"])
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for coder, html in written.items():
        (out / PAGES / f"{coder}.html").write_text(html, encoding="utf-8")
    return manifest


# ---- import ----------------------------------------------------------------------------------------------------

def pair_line(row: dict, value: str | None, note: str, reviewer: str, imported_at: str, design: str,
              withdrawn: bool = False) -> dict:
    """A pairs_log line: the coder's choice beside the model's (None on a tie in the rank scores), with the close
    tick read from the note and the session's design id. A withdrawn line (value None) takes back an earlier answer
    the coder has left blank since."""
    line = {"reviewer": reviewer, "batch_id": row["batch_id"], "code": row["code"], "left": row["left"],
            "right": row["right"], "value": value, "close": CLOSE in (note or ""), "model": row["primary"],
            "gap": float(row["gap"]), "note": " ".join((note or "").replace(CLOSE, " ").split()),
            "design_id": design, "imported_at": imported_at}
    return {**line, "withdrawn": True} if withdrawn else line


def of_design(lines: list[dict], design: str | None) -> list[dict]:
    """The pair lines of one design (every pair line when `design` is None); lines of any other kind are kept."""
    return [line for line in lines if "batch_id" not in line or design is None or line.get("design_id") == design]


def last_pairs(lines: list[dict]) -> dict[tuple[str, str, str], dict]:
    """The standing answer per (reviewer, batch, code): the last line decides, so a later import replaces an
    earlier answer and a withdrawn line takes it back."""
    last: dict[tuple[str, str, str], dict] = {}
    for line in lines:
        if "batch_id" in line:
            last[(line["reviewer"], line["batch_id"], line["code"])] = line
    return {key: line for key, line in last.items() if line.get("value") in PAIR_VALUES}


def _chosen(line: dict) -> str:
    """The clip a pair answer chose."""
    return line["left"] if line["value"] == "left" else line["right"]


def _winner(line: dict) -> list[str]:
    return [line["left"], line["right"]] if line["value"] == "left" else [line["right"], line["left"]]


def bradley_terry(lines: list[dict], reviewer: str, design: str | None = None) -> dict[str, dict[str, float]]:
    """{code: {video_id: score}}: one coder's Bradley-Terry scores per Stage A item, `plackett_luce` over that coder's
    pair choices as orders [winner, loser] (log-worths centred on zero), of one design when `design` is given. An
    unanswered pair carries no evidence."""
    by_code: dict[str, list[list[str]]] = {}
    for (who, _, code), line in last_pairs(of_design(lines, design)).items():
        if who == reviewer:
            by_code.setdefault(code, []).append(_winner(line))
    return {code: plackett_luce(orders) for code, orders in by_code.items()}


def spearman(a: dict[str, float], b: dict[str, float]) -> float | None:
    """Spearman's rank correlation of two scorings over the videos both score (descriptive; None under three
    videos or when either scoring is constant)."""
    from scipy.stats import spearmanr
    common = sorted(set(a) & set(b))
    x, y = [a[v] for v in common], [b[v] for v in common]
    if len(common) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def rating_lines(run: Path, video_ids: list[str]) -> list[dict]:
    """Each coder's last rating of each Stage A item per video, from the videos' review logs, as
    {"reviewer", "video_id", "path", "new"}: what `pair_agreement` checks a coder's pair choices against."""
    out = []
    for video_id in video_ids:
        log = Path(run) / video_id / "review" / "review_log.jsonl"
        if not log.exists():
            continue
        last: dict[tuple[str, str], dict] = {}
        for text in log.read_text(encoding="utf-8").splitlines():
            if not text.strip():
                continue
            line = json.loads(text)
            if (line.get("role") == "coder" and not line.get("implied")
                    and str(line.get("path", "")).startswith("stage_a:")):
                last[(line["reviewer"], line["path"])] = line
        out += [{"reviewer": who, "video_id": video_id, "path": path, "new": line.get("new")}
                for (who, path), line in sorted(last.items())]
    return out


# ---- agreement -------------------------------------------------------------------------------------------------

def _pair_weights(draws, left, right) -> np.ndarray:
    """The weight of each judgement in each resample of videos: m_left × m_right, where m is how often a video was
    drawn. `draws` (n_boot × n_videos) holds the counts; `left` and `right` the videos' indices per judgement."""
    counts = np.asarray(draws, dtype=float)
    return counts[:, np.asarray(left, dtype=int)] * counts[:, np.asarray(right, dtype=int)]


def _share(agree: list[bool]) -> float | None:
    return float(np.mean(agree)) if agree else None


def _rating(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _interval(index: dict[str, int], judged: list[tuple[str, str, bool]], draws: np.ndarray) -> list[float] | None:
    """The 95% interval of a share of pair judgements, resampling videos: a pair counts m_a × m_b times in a draw.
    None without judgements, with fewer than MIN_CLUSTER_VIDEOS videos, or when too few draws are defined."""
    if not judged or len({v for a, b, _ in judged for v in (a, b)}) < MIN_CLUSTER_VIDEOS:
        return None
    weights = _pair_weights(draws, [index[a] for a, _, _ in judged], [index[b] for _, b, _ in judged])
    total = weights.sum(axis=1)
    defined = total > 0
    if defined.mean() < MIN_DEFINED_SHARE:
        return None
    agree = np.array([ok for *_, ok in judged], dtype=float)
    shares = (weights[defined] @ agree) / total[defined]
    low, high = np.quantile(shares, [0.025, 0.975])
    return [float(low), float(high)]


def pair_agreement(lines: list[dict], video_ids: list[str], n_boot: int = N_BOOT, seed: int = 0,
                   codes: list[str] | None = None) -> dict:
    """Pair agreement per Stage A item, from pairs_log lines (with "batch_id") and, for coherence, rating lines
    ({"reviewer", "video_id", "path": "stage_a:<item>", "new"}; see `rating_lines`). Per item:

    - "coder_model": {coder: share of that coder's answered pairs on which the coder chose as the model's scores};
    - "coder_coder": share of the pairs both of exactly two coders answered on which they chose alike (else None);
    - "n_pairs" (pairs anyone answered), "n_judgements", "videos", "coders";
    - "by_gap_tercile": the same shares in thirds of the pairs ordered by the model's score gap (1 the narrowest),
      [{"tercile", "coder_model", "coder_coder", "n", "gap_range", "agree", "n_judged", "coder_coder_agree",
      "n_both"}] ("agree" of "n_judged" the coders' answers pooled against the model, "coder_coder_agree" of "n_both"
      the pairs both coders answered): the scores mean something when agreement rises with the gap;
    - "close_in_lowest_tercile_share": of the "close" ticks, the share on pairs of the narrowest third;
    - "coherence": {coder: of the pairs whose two clips the coder rated differently, the share where the pair
      choice agrees with the ratings}; "ratings_tie": {coder: share of rated pairs whose ratings tie};
    - "ci95": {"coder_model": {coder: [low, high]}, "coder_model_pooled", "coder_coder"}: resampling the session's
      videos with replacement (seeded), a pair counting m_a × m_b times when its videos are drawn m_a and m_b times,
      since pairs that share a video are not independent.

    Shares are None where nothing was answered; chance is 0.5."""
    pairs = last_pairs(lines)
    ratings: dict[tuple[str, str, str], float | None] = {}
    for line in lines:
        if "batch_id" not in line and str(line.get("path", "")).startswith("stage_a:"):
            ratings[(line["reviewer"], line["video_id"], line["path"].split(":", 1)[1])] = _rating(line.get("new"))
    found = list(dict.fromkeys(code for (_, _, code) in pairs))         # in the order the lines name them
    order = [c for c in (codes or []) if c in found] + [c for c in found if c not in (codes or [])]
    index = {v: i for i, v in enumerate(video_ids)}
    rng = np.random.default_rng(seed)
    n = len(video_ids)
    # how often each video is drawn in each resample (n_boot × n)
    draws = (rng.integers(n, size=(n_boot, n))[:, :, None] == np.arange(n)).sum(axis=1) if n else np.zeros((0, 0))
    out = {}
    for code in order:
        mine = {key: line for key, line in pairs.items() if key[2] == code}
        coders = sorted({who for who, _, _ in mine})
        batches = sorted({batch for _, batch, _ in mine})
        meta = {batch: next(line for (_, b, _), line in mine.items() if b == batch) for batch in batches}
        by_coder = {c: {batch: line for (who, batch, _), line in mine.items() if who == c} for c in coders}
        both = ([batch for batch in batches if all(batch in by_coder[c] for c in coders)] if len(coders) == 2 else [])

        def model_agree(coder, chosen):
            # each line against its own model side; a tie in the rank scores is no model answer
            return [(line["left"], line["right"], line["value"] == line["model"])
                    for b in chosen if b in by_coder[coder]
                    for line in [by_coder[coder][b]] if line.get("model") in PAIR_VALUES]

        def coder_agree(chosen):
            # the two coders agree when they chose the same clip, whichever side their pages showed it on
            return [(meta[b]["left"], meta[b]["right"],
                     _chosen(by_coder[coders[0]][b]) == _chosen(by_coder[coders[1]][b])) for b in chosen if b in both]

        ranked = sorted(batches, key=lambda b: (meta[b]["gap"], b))
        thirds = [list(part) for part in np.array_split(np.array(ranked, dtype=object), 3)] if ranked else []
        terciles = [{"tercile": i + 1, "n": len(part),
                     "gap_range": [float(meta[part[0]]["gap"]), float(meta[part[-1]]["gap"])] if part else None,
                     "coder_model": {c: _share([ok for *_, ok in model_agree(c, part)]) for c in coders},
                     "coder_coder": _share([ok for *_, ok in coder_agree(part)]) if both else None,
                     # the counts behind the shares, pooled over the coders (the reporting's k of n per third)
                     "agree": sum(ok for c in coders for *_, ok in model_agree(c, part)),
                     "n_judged": sum(len(model_agree(c, part)) for c in coders),
                     "coder_coder_agree": sum(ok for *_, ok in coder_agree(part)) if both else None,
                     "n_both": len(coder_agree(part)) if both else None}
                    for i, part in enumerate(thirds)]
        lowest = set(thirds[0]) if thirds else set()
        closes = [b for c in coders for b, line in by_coder[c].items() if line.get("close")]
        coherence, ties = {}, {}
        for c in coders:
            agreeing, tied, rated = [], 0, 0
            for b, line in by_coder[c].items():
                left = ratings.get((c, line["left"], code))
                right = ratings.get((c, line["right"], code))
                if left is None or right is None:
                    continue
                rated += 1
                if left == right:
                    tied += 1
                else:
                    agreeing.append((line["value"] == "left") == (left > right))
            coherence[c], ties[c] = _share(agreeing), (tied / rated if rated else None)
        judged = {c: model_agree(c, batches) for c in coders}
        out[code] = {
            "coders": coders, "n_pairs": len(batches), "n_judgements": sum(len(j) for j in judged.values()),
            "n_by_coder": {c: len(judged[c]) for c in coders},
            "n_model_ties": sum(1 for b in batches if meta[b].get("model") not in PAIR_VALUES),
            "videos": len({v for b in batches for v in (meta[b]["left"], meta[b]["right"])}),
            "coder_model": {c: _share([ok for *_, ok in judged[c]]) for c in coders},
            "coder_coder": _share([ok for *_, ok in coder_agree(batches)]) if both else None,
            "n_both": len(both),
            "by_gap_tercile": terciles,
            "close_in_lowest_tercile_share": (sum(b in lowest for b in closes) / len(closes)) if closes else None,
            "n_close": len(closes),
            "coherence": coherence, "ratings_tie": ties,
            "ci95": {"coder_model": {c: _interval(index, judged[c], draws) for c in coders},
                     "coder_model_pooled": _interval(index, [x for c in coders for x in judged[c]], draws),
                     "coder_coder": _interval(index, coder_agree(batches), draws) if both else None},
            "n_boot": n_boot, "seed": seed}
    return out


def pooled_pairs(agreement: dict) -> dict:
    """pooled_summary["pairs"]: per item the figures the paper table judges, "coder_model" pooled over the coders'
    judgements (the share of all coder-pair answers that agree with the model) and "coder_coder", with "n" the
    judgements and "videos"; the per-coder shares and the rest of `pair_agreement`'s figures (the "close" ticks
    "n_close", the pairs both coders answered "n_both") are kept beside them."""
    out = {}
    for code, f in agreement.items():
        n_by = f["n_by_coder"]
        judged = sum(n_by.values())
        # each coder's share times that coder's answered pairs, summed: the share of all coder-pair answers
        pooled = (sum(f["coder_model"][c] * n for c, n in n_by.items() if n) / judged) if judged else None
        out[code] = {"coder_model": pooled, "coder_coder": f["coder_coder"], "n": judged, "videos": f["videos"],
                     "n_pairs": f["n_pairs"], "n_model_ties": f["n_model_ties"],
                     "coder_model_by_coder": f["coder_model"],
                     "ci95": {"coder_model": f["ci95"]["coder_model_pooled"], "coder_coder": f["ci95"]["coder_coder"],
                              "by_coder": f["ci95"]["coder_model"]},
                     **{k: f[k] for k in ("by_gap_tercile", "close_in_lowest_tercile_share", "coherence",
                                          "ratings_tie", "n_close", "n_both")}}
    return out

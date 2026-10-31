#!/usr/bin/env python3
"""Stage A cross-video ranking of prepared presentations.

Two sources of videos: a finished panel run (`--run runs/<panel_run>`), where each video's
prepared window is the one its primary pass recorded and is cut to a clip; or a directory of
standalone pitch videos (`--run <out_dir> --video-dir DIR`, the Hu & Ma corpus), where the whole
file is the prepared presentation and is uploaded as it is.

Without --execute: prints the plan (batches with their videos, skipped videos, billable
calls) and builds nothing. With --execute: makes one call per batch and writes
<run>/ranking/. Ranks are relative to the videos ranked together in this run.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.errors import ConfigError, RunFatalError  # noqa: E402
from pipeline.execute import TokenBucket  # noqa: E402
from pipeline.instrument import check_release, load_release  # noqa: E402
from pipeline.ingest import is_video_file  # noqa: E402
from pipeline.media import clip  # noqa: E402
from pipeline.stage_a.rank_pipeline import (  # noqa: E402
    RANKING_DIR, full_video_windows, plan_ranking, plan_ranking_from_windows, run_ranking, whole_source,
)
from pipeline.stage_b.panel_books import load_panel_books  # noqa: E402
from pipeline.stage_b.panel_pipeline import _read, load_panel_config  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True,
                   help="panel run directory (runs/<run_name>); with --video-dir, the output directory")
    p.add_argument("--video-dir", help="rank whole videos from this directory instead of a panel run")
    p.add_argument("--config", default=str(ROOT / "configs" / "stage_b_panel.yaml"))
    p.add_argument("--video-ids", nargs="+",
                   help="default: every video folder in the run, or every video file in --video-dir")
    p.add_argument("--video-ids-from", help="a CSV with a video_id column listing the videos to rank")
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--appearances", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--provider", choices=["gemini", "fake"], default="gemini")
    p.add_argument("--fixtures")
    return p.parse_args(argv)


def video_ids_in(run_root: Path) -> list[str]:
    """Every video folder of a panel run: a folder with an experiment.json."""
    return sorted(d.name for d in run_root.iterdir()
                  if d.is_dir() and d.name != RANKING_DIR and (d / "experiment.json").exists())


def video_ids_from_csv(path: Path) -> list[str]:
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        if "video_id" not in (reader.fieldnames or []):
            raise ConfigError(f"{path} has no video_id column")
        return sorted({row["video_id"].strip() for row in reader if row.get("video_id", "").strip()})


def wanted_video_ids(args, run_root: Path) -> list[str]:
    if args.video_ids and args.video_ids_from:
        raise ConfigError("--video-ids and --video-ids-from are mutually exclusive")
    if args.video_ids:
        return list(args.video_ids)
    if args.video_ids_from:
        return video_ids_from_csv(Path(args.video_ids_from))
    if args.video_dir:
        return sorted({p.stem for p in Path(args.video_dir).iterdir() if is_video_file(p)})
    return video_ids_in(run_root)


def build_provider(args, run_root: Path):
    if args.provider == "fake":
        from pipeline.providers.fake import FakeProvider
        return FakeProvider(args.fixtures)
    from pipeline.providers.gemini import build_gemini_provider
    from pipeline.registry import Registry
    # the upload cache lives in the ranking folder: nothing else in the panel run is written
    rdir = run_root / RANKING_DIR
    rdir.mkdir(parents=True, exist_ok=True)
    return build_gemini_provider(Registry(rdir / "registry.sqlite"))


def print_plan(release, cfg, run_root: Path, plan: dict) -> None:
    problems = check_release(release)
    print(f"release: {release.release_id}")
    print(f"check_release: {problems if problems else 'ok'}")
    print(f"model: {cfg.model.name} temperature {cfg.model.temperature}")
    print(f"windows: {'whole videos, uploaded as they are' if plan['windows_from'] == 'whole_videos' else 'prepared windows from the primary pass, cut to clips'}")
    summary = plan["summary"]
    print(f"videos: {summary['n_videos']} batches: {summary['n_batches']} "
          f"repeated pairs: {summary['repeated_pairs']} settings: {plan['settings']}")
    billable = 0
    for batch in plan["design"]:
        cached = (run_root / RANKING_DIR / "batches" / batch["batch_id"] / "result.json").exists()
        billable += not cached
        clips = " ".join(f"{label}={video}" for label, video in zip(batch["labels"], batch["video_ids"]))
        print(f"{batch['batch_id']} {'cached' if cached else 'billable'}: {clips}")
    for video_id, reason in plan["skipped"].items():
        print(f"skipped {video_id}: {reason}")
    multi = [v for v, w in plan["windows"].items() if w["n_spans"] > 1]
    if multi:
        print(f"clips spanning an interruption (more than one prepared span): {' '.join(multi)}")
    print(f"total billable calls: {billable}")


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        run_root = Path(args.run)
        if args.video_dir:
            video_dir = Path(args.video_dir)
            if not video_dir.is_dir():
                raise ConfigError(f"video directory not found: {video_dir}")
        elif not run_root.is_dir():
            raise ConfigError(f"panel run directory not found: {run_root}")
        if args.execute and args.provider == "fake" and not args.fixtures:
            raise ConfigError("--fixtures is required with --provider fake")
        cfg = load_panel_config(args.config)
        release = load_release()
        video_ids = wanted_video_ids(args, run_root)
        if args.video_dir:
            windows, skipped = full_video_windows(video_dir, video_ids)
            plan = plan_ranking_from_windows(windows, skipped, args.batch_size, args.appearances, args.seed,
                                             windows_from="whole_videos")
            cut = whole_source
        else:
            plan = plan_ranking(run_root, video_ids, args.batch_size, args.appearances, args.seed)
            cut = clip

        if not args.execute:
            print("without --execute: plan only, no calls made, nothing written")
            print_plan(release, cfg, run_root, plan)
            return 0

        run_root.mkdir(parents=True, exist_ok=True)
        provider = build_provider(args, run_root)
        bucket = TokenBucket(cfg.limits.requests_per_minute)
        result = run_ranking(provider, release, cfg, run_root, plan, before_call=bucket.acquire, cut=cut)
        print(f"RANKING {result['status']} ({result['batches_ok']}/{result['batches_total']} batches, "
              f"{result['generation_calls']} calls)")
        scores = _read(run_root / RANKING_DIR / "scores.json")
        for code in load_panel_books(release).stage_a.ids:
            print(f"{code}: {' > '.join(scores[code]['order'])}")
        return 0 if result["status"] == "complete" else 1
    except (ConfigError, RunFatalError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

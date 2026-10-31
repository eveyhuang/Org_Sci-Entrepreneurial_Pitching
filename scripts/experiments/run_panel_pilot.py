#!/usr/bin/env python3
"""Stage B v3 panel pipeline: two identical full-video passes per video.

Phases: preflight (plan, no provider), annotate (needs --execute to call the model),
agreement (offline recomputation of each pass's validation report and of agreement.json;
exits 1 when a selected video lacks a pass).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.errors import ConfigError, RunFatalError  # noqa: E402
from pipeline.execute import TokenBucket  # noqa: E402
from pipeline.instrument import check_release, load_release  # noqa: E402
from pipeline.prompts import prompt_hash  # noqa: E402
from pipeline.stage_b.panel_books import load_panel_books, render_panel_prompt  # noqa: E402
from pipeline.stage_b.panel_pipeline import (  # noqa: E402
    PASSES, load_panel_config, plan_calls, revalidate, run_agreement, run_video,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs" / "stage_b_panel.yaml"))
    p.add_argument("--instrument", type=Path, help="Optional custom instrument manifest")
    p.add_argument("--selection", help="JSON with videos: [{video_id, path, duration_s}]")
    p.add_argument("--video-ids", nargs="+")
    p.add_argument("--run-name")
    p.add_argument("--runs-root", default=str(ROOT / "runs"))
    p.add_argument("--phase", choices=["preflight", "annotate", "agreement"], default="preflight")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--provider", choices=["gemini", "fake"], default="gemini")
    p.add_argument("--fixtures")
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args(argv)


def load_videos(args) -> list[dict]:
    if not args.selection:
        raise ConfigError("--selection is required")
    videos = json.loads(Path(args.selection).read_text())["videos"]
    if args.video_ids:
        wanted = set(args.video_ids)
        videos = [v for v in videos if v["video_id"] in wanted]
        missing = wanted - {v["video_id"] for v in videos}
        if missing:
            raise ConfigError(f"video ids not in selection: {sorted(missing)}")
    return videos


def build_provider(args, run_root: Path):
    if args.provider == "fake":
        from pipeline.providers.fake import FakeProvider
        if not args.fixtures:
            raise ConfigError("--fixtures is required with --provider fake")
        return FakeProvider(args.fixtures)
    from pipeline.providers.gemini import build_gemini_provider
    from pipeline.registry import Registry
    run_root.mkdir(parents=True, exist_ok=True)
    return build_gemini_provider(Registry(run_root / "registry.sqlite"))


def preflight(release, books, videos, run_root: Path) -> None:
    problems = check_release(release)
    if problems:
        raise ConfigError("instrument release is not clean: " + "; ".join(problems))
    prompt = render_panel_prompt(books, release.templates_dir)
    print(f"release: {release.release_id}")
    print(f"check_release: {problems if problems else 'ok'}")
    print(f"prompt_hash: {prompt_hash(prompt)} length: {len(prompt)}")
    rows = plan_calls(videos, run_root)
    for row in rows:
        print(f"{row['video_id']} {row['pass']} {'cached' if row['cached'] else 'billable'}")
    print(f"total billable: {sum(1 for r in rows if not r['cached'])}")


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_panel_config(args.config)
        run_root = Path(args.runs_root) / (args.run_name or cfg.run_name)
        release = load_release(args.instrument)
        books = load_panel_books(release)
        videos = load_videos(args)

        if args.phase == "preflight" or (args.phase == "annotate" and not args.execute):
            if args.phase == "annotate":
                print("annotate without --execute: preflight only, no calls made")
            preflight(release, books, videos, run_root)
            return 0

        if args.phase == "agreement":
            lacking = False
            for v in videos:
                present = revalidate(books, v["video_id"], float(v["duration_s"]), run_root)
                print(f"VALIDATION {v['video_id']} rewritten for: {', '.join(present) or 'no pass'}")
                if len(present) == len(PASSES):
                    run_agreement(release, books, cfg, v["video_id"], run_root)
                    print(f"AGREEMENT {v['video_id']} written")
                else:
                    lacking = True
                    print(f"AGREEMENT {v['video_id']} skipped (needs both passes)")
            return 1 if lacking else 0

        preflight(release, books, videos, run_root)
        provider = build_provider(args, run_root)
        bucket = TokenBucket(cfg.limits.requests_per_minute)
        statuses = {}
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers))
        try:
            futures = {
                pool.submit(run_video, provider, release, books, cfg, v, run_root,
                            before_call=bucket.acquire): v["video_id"]
                for v in videos
            }
            for fut in concurrent.futures.as_completed(futures):
                vid = futures[fut]
                try:
                    statuses[vid] = fut.result()["status"]
                except RunFatalError:
                    for pending in futures:
                        pending.cancel()
                    raise
                except Exception as exc:  # noqa: BLE001 - one video must not hide the others
                    statuses[vid] = "error"
                    print(f"VIDEO {vid} error: {exc}", flush=True)
                    continue
                print(f"VIDEO {vid} {statuses[vid]}", flush=True)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        return 0 if all(s == "complete" for s in statuses.values()) else 1
    except (ConfigError, RunFatalError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

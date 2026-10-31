#!/usr/bin/env python3
"""Run a config's annotation passes. Idempotent: finished units are skipped."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.annotate import run_all_passes  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.errors import RunFatalError  # noqa: E402
from pipeline.registry import Registry  # noqa: E402
from pipeline.runs import create_or_resume_run, update_manifest  # noqa: E402


def build_provider(args, registry):
    if args.provider == "fake":
        from pipeline.providers.fake import FakeProvider

        if not args.fixtures:
            raise SystemExit("--fixtures is required with --provider fake")
        return FakeProvider(args.fixtures)
    from pipeline.providers.gemini import build_gemini_provider

    return build_gemini_provider(registry)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--registry", default="registry.sqlite")
    parser.add_argument("--pass", dest="pass_name", default=None, help="run a single pass")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--video-ids", nargs="+", default=None)
    parser.add_argument("--dev-only", action="store_true", help="only the videos in corpus.dev_set")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--new-run", action="store_true", help="do not resume an existing run dir")
    parser.add_argument("--provider", choices=["gemini", "fake"], default="gemini")
    parser.add_argument("--fixtures", default=None, help="fixture dir for --provider fake")
    parser.add_argument("--runs-root", default="runs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dev_only:
        if args.video_ids:
            raise SystemExit("--dev-only and --video-ids are mutually exclusive")
        args.video_ids = cfg.dev_video_ids()
    registry = Registry(args.registry)
    exit_code = 0
    try:
        run = create_or_resume_run(cfg, runs_root=args.runs_root, force_new=args.new_run)
        update_manifest(
            run,
            {
                "model_requested": cfg.model.name,
                "temperature": cfg.model.temperature,
                "corpus": cfg.corpus.name,
            },
        )
        print(f"run: {run.run_id}  model: {cfg.model.name}")

        summaries = run_all_passes(
            cfg, registry, run, build_provider(args, registry),
            pass_name=args.pass_name, video_ids=args.video_ids, limit=args.limit,
            retry_failed=args.retry_failed,
        )
        for summary in summaries:
            print(
                f"{summary.pass_name} [{summary.model}]: done={summary.done} failed={summary.failed} "
                f"skipped={summary.skipped} tokens_in={summary.input_tokens} "
                f"tokens_out={summary.output_tokens} tokens_thinking={summary.thinking_tokens}"
            )
            if summary.failures:
                print(f"  failures: {summary.failures}")
        print(f"manifest: {run.root / 'manifest.json'}")
    except RunFatalError as exc:
        print(f"RUN FATAL: {exc}", file=sys.stderr)
        print("Progress is saved in the registry; fix the cause and re-run.", file=sys.stderr)
        exit_code = 2
    finally:
        registry.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

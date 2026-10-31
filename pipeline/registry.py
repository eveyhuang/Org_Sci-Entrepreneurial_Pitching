"""SQLite-backed work registry.

Work units are content-addressed: the unit_id hashes everything that could change
the result (video, span, pass, prompt, model, media mode, replicate index), so
re-running a command skips finished work and editing a prompt invalidates exactly
the affected units. unit_id is global — resume works across runs.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id   TEXT PRIMARY KEY,
    corpus     TEXT NOT NULL,
    path       TEXT NOT NULL,
    sha256     TEXT,
    duration_s REAL,
    width      INTEGER,
    height     INTEGER,
    has_audio  INTEGER,
    probe_ok   INTEGER,
    meta_json  TEXT
);
CREATE TABLE IF NOT EXISTS uploads (
    file_sha256 TEXT PRIMARY KEY,
    gemini_name TEXT NOT NULL,
    uploaded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS work_units (
    unit_id       TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    video_id      TEXT NOT NULL,
    pass_name     TEXT NOT NULL,
    span_start_s  REAL,
    span_end_s    REAL,
    prompt_hash   TEXT NOT NULL,
    model         TEXT NOT NULL,
    media         TEXT NOT NULL,
    k             INTEGER NOT NULL,
    status        TEXT NOT NULL,
    error_class   TEXT,
    error_detail  TEXT,
    raw_path      TEXT,
    parsed_path   TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    finished_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_units_run_status ON work_units (run_id, status);
CREATE INDEX IF NOT EXISTS idx_units_pass ON work_units (pass_name, status);
CREATE TABLE IF NOT EXISTS episodes (
    video_id       TEXT NOT NULL,
    episode_idx    INTEGER NOT NULL,
    start_s        REAL NOT NULL,
    end_s          REAL NOT NULL,
    source_unit_id TEXT,
    frozen         INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (video_id, episode_idx)
);
CREATE TABLE IF NOT EXISTS decisions (
    video_id       TEXT NOT NULL,
    speaker        TEXT NOT NULL,
    decision       TEXT NOT NULL,
    turn           INTEGER,
    terms          TEXT,
    source_unit_id TEXT,
    PRIMARY KEY (video_id, speaker)
);
"""


# Columns added after the first release. registry.sqlite holds paid-for state, so the
# schema only ever grows, by guarded ALTER TABLE, and existing rows are never rewritten.
MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("videos", "skip_reason", "TEXT"),
    ("episodes", "kind", "TEXT NOT NULL DEFAULT 'episode'"),
    ("episodes", "probing_speaker", "TEXT"),
    ("episodes", "start_turn", "INTEGER"),
    ("episodes", "end_turn", "INTEGER"),
    ("episodes", "cut", "INTEGER"),
    ("episodes", "topic", "TEXT"),
    ("episodes", "resolution", "TEXT"),
    ("work_units", "thinking_tokens", "INTEGER"),
    ("videos", "deal", "INTEGER"),
)

# The prepared (pre-Q&A) segment shares the episodes table; (video_id, episode_idx) is the
# primary key and SQLite cannot alter one, so it takes an index no real episode can have.
PREPARED_IDX = -1


@dataclass
class VideoRow:
    video_id: str
    corpus: str
    path: str
    sha256: str | None = None
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    has_audio: bool | None = None
    probe_ok: bool = False
    meta_json: str = "{}"
    skip_reason: str | None = None
    deal: bool | None = None  # study OUTCOME (Revision R2): a deal was agreed on air


@dataclass
class WorkUnit:
    unit_id: str
    run_id: str
    video_id: str
    pass_name: str
    prompt_hash: str
    model: str
    media: str
    k: int
    span_start_s: float | None = None
    span_end_s: float | None = None
    status: str = "pending"
    error_class: str | None = None
    error_detail: str | None = None
    raw_path: str | None = None
    parsed_path: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    # Not a database column: the span already identifies the episode, and this is
    # carried only so the parsed record can name the index. Rows read back from
    # SQLite leave it None.
    episode_idx: int | None = None

    @property
    def span(self) -> tuple[float, float] | None:
        if self.span_start_s is None or self.span_end_s is None:
            return None
        return (self.span_start_s, self.span_end_s)


@dataclass
class EpisodeRow:
    video_id: str
    episode_idx: int
    start_s: float
    end_s: float
    source_unit_id: str | None = None
    frozen: bool = False
    kind: str = "episode"  # 'episode' | 'prepared'
    probing_speaker: str | None = None
    start_turn: int | None = None
    end_turn: int | None = None
    cut: bool | None = None
    topic: str | None = None
    resolution: str | None = None


@dataclass
class DecisionRow:
    """One investor's on-air decision (Revision R2): a study OUTCOME, never a predictor."""

    video_id: str
    speaker: str
    decision: str  # "out" | "offer" | "joins_offer" | "no_decision_shown"
    turn: int | None = None
    terms: str | None = None
    source_unit_id: str | None = None


def make_unit_id(
    *,
    video_id: str,
    pass_name: str,
    prompt_hash: str,
    model: str,
    media: str,
    k: int,
    span: tuple[float, float] | None = None,
    config_fingerprint: str = "",
) -> str:
    """Content-address a work unit.

    `config_fingerprint` carries settings that change the instruction sent to the
    model without appearing in the prompt text (temperature, max_output_tokens, the
    response schema). It affects identity only — the manifest records the values.
    """
    span_part = "whole" if span is None else f"{span[0]:.3f}-{span[1]:.3f}"
    payload = "|".join(
        [video_id, pass_name, prompt_hash, model, media, str(k), span_part, config_fingerprint]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _now_iso(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).isoformat()


class Registry:
    """Thread-safe SQLite wrapper. One instance per process; share across threads."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()
            self._migrate()

    def _migrate(self) -> None:
        for table, column, declaration in MIGRATIONS:
            present = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")}
            if column not in present:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- videos ----------------------------------------------------------
    def upsert_video(self, row: VideoRow) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO videos (video_id, corpus, path, sha256, duration_s, width,
                       height, has_audio, probe_ok, meta_json, skip_reason)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(video_id) DO UPDATE SET
                       corpus=excluded.corpus, path=excluded.path, sha256=excluded.sha256,
                       duration_s=excluded.duration_s, width=excluded.width,
                       height=excluded.height, has_audio=excluded.has_audio,
                       probe_ok=excluded.probe_ok, meta_json=excluded.meta_json,
                       skip_reason=excluded.skip_reason""",
                (row.video_id, row.corpus, row.path, row.sha256, row.duration_s, row.width,
                 row.height, int(bool(row.has_audio)) if row.has_audio is not None else None,
                 int(row.probe_ok), row.meta_json, row.skip_reason),
            )
            self._conn.commit()

    def get_video(self, video_id: str) -> VideoRow | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM videos WHERE video_id = ?", (video_id,)
            ).fetchone()
        return _video_from_row(row) if row else None

    def list_videos(
        self, corpus: str | None = None, probe_ok_only: bool = True, include_skipped: bool = True
    ) -> list[VideoRow]:
        sql = "SELECT * FROM videos WHERE 1=1"
        params: list[object] = []
        if corpus:
            sql += " AND corpus = ?"
            params.append(corpus)
        if probe_ok_only:
            sql += " AND probe_ok = 1"
        if not include_skipped:
            sql += " AND skip_reason IS NULL"
        sql += " ORDER BY video_id"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_video_from_row(r) for r in rows]

    def set_skip_reason(self, video_id: str, reason: str | None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE videos SET skip_reason = ? WHERE video_id = ?", (reason, video_id)
            )
            self._conn.commit()

    # --- upload cache ----------------------------------------------------
    def get_upload(
        self, file_sha256: str, max_age_h: float = 44.0, now: datetime | None = None
    ) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT gemini_name, uploaded_at FROM uploads WHERE file_sha256 = ?",
                (file_sha256,),
            ).fetchone()
        if row is None:
            return None
        uploaded_at = datetime.fromisoformat(row["uploaded_at"])
        current = now or datetime.now(timezone.utc)
        if (current - uploaded_at).total_seconds() > max_age_h * 3600:
            return None
        return row["gemini_name"]

    def put_upload(self, file_sha256: str, gemini_name: str, now: datetime | None = None) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO uploads (file_sha256, gemini_name, uploaded_at) VALUES (?,?,?)
                   ON CONFLICT(file_sha256) DO UPDATE SET
                       gemini_name=excluded.gemini_name, uploaded_at=excluded.uploaded_at""",
                (file_sha256, gemini_name, _now_iso(now)),
            )
            self._conn.commit()

    def delete_upload(self, file_sha256: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM uploads WHERE file_sha256 = ?", (file_sha256,))
            self._conn.commit()

    # --- work units ------------------------------------------------------
    def enqueue(self, unit: WorkUnit) -> bool:
        """Insert if absent, else re-point the existing row at the current run.

        unit_id is global so completed work resumes across runs, but every read
        filters by run_id. Without the re-point, a second run over already-done
        units would see nothing in done_units() and aggregate would silently write
        empty tables. Returns True only when a new pending row was created.
        """
        with self._lock:
            cur = self._conn.execute(
                """INSERT OR IGNORE INTO work_units
                   (unit_id, run_id, video_id, pass_name, span_start_s, span_end_s,
                    prompt_hash, model, media, k, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (unit.unit_id, unit.run_id, unit.video_id, unit.pass_name,
                 unit.span_start_s, unit.span_end_s, unit.prompt_hash, unit.model,
                 unit.media, unit.k, "pending"),
            )
            inserted = cur.rowcount == 1
            if not inserted:
                self._conn.execute(
                    "UPDATE work_units SET run_id = ? WHERE unit_id = ? AND run_id != ?",
                    (unit.run_id, unit.unit_id, unit.run_id),
                )
            self._conn.commit()
            return inserted

    def get_unit(self, unit_id: str) -> WorkUnit | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM work_units WHERE unit_id = ?", (unit_id,)
            ).fetchone()
        return _unit_from_row(row) if row else None

    def pending_units(
        self, run_id: str, pass_name: str | None = None, include_failed: bool = False
    ) -> list[WorkUnit]:
        statuses = ["pending", "failed"] if include_failed else ["pending"]
        sql = (
            "SELECT * FROM work_units WHERE run_id = ? AND status IN "
            f"({','.join('?' * len(statuses))})"
        )
        params: list[object] = [run_id, *statuses]
        if pass_name:
            sql += " AND pass_name = ?"
            params.append(pass_name)
        sql += " ORDER BY video_id, k"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_unit_from_row(r) for r in rows]

    def mark_done(
        self, unit_id: str, raw_path: str, parsed_path: str,
        input_tokens: int | None, output_tokens: int | None,
        thinking_tokens: int | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE work_units SET status='done', raw_path=?, parsed_path=?,
                       input_tokens=?, output_tokens=?, thinking_tokens=?, error_class=NULL,
                       error_detail=NULL, finished_at=?
                   WHERE unit_id = ?""",
                (raw_path, parsed_path, input_tokens, output_tokens, thinking_tokens,
                 _now_iso(), unit_id),
            )
            self._conn.commit()

    def mark_failed(
        self, unit_id: str, error_class: str, error_detail: str, raw_path: str | None = None
    ) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE work_units SET status='failed', error_class=?, error_detail=?,
                       raw_path=COALESCE(?, raw_path), finished_at=?
                   WHERE unit_id = ?""",
                (error_class, error_detail[:2000], raw_path, _now_iso(), unit_id),
            )
            self._conn.commit()

    def status_counts(self, run_id: str) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM work_units WHERE run_id=? GROUP BY status",
                (run_id,),
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def failure_counts(self, run_id: str) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT error_class, COUNT(*) AS n FROM work_units
                   WHERE run_id=? AND status='failed' GROUP BY error_class""",
                (run_id,),
            ).fetchall()
        return {r["error_class"]: r["n"] for r in rows}

    def done_units(self, run_id: str, pass_name: str | None = None) -> list[WorkUnit]:
        sql = "SELECT * FROM work_units WHERE run_id=? AND status='done'"
        params: list[object] = [run_id]
        if pass_name:
            sql += " AND pass_name = ?"
            params.append(pass_name)
        sql += " ORDER BY video_id, span_start_s, k"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_unit_from_row(r) for r in rows]

    def latest_done_unit(self, video_id: str, pass_name: str, k: int = 0) -> WorkUnit | None:
        """Newest finished unit of a pass for a video, across ALL runs.

        Downstream passes find upstream results here rather than in run-directory files:
        unit ids are global, run directories are per-config, so a fresh run directory
        skips an already-done upstream unit and would otherwise have nothing to read.
        """
        with self._lock:
            row = self._conn.execute(
                """SELECT * FROM work_units
                   WHERE video_id = ? AND pass_name = ? AND k = ? AND status = 'done'
                   ORDER BY finished_at DESC LIMIT 1""",
                (video_id, pass_name, k),
            ).fetchone()
        return _unit_from_row(row) if row else None

    # --- episodes --------------------------------------------------------
    def upsert_episode(
        self, video_id: str, episode_idx: int, start_s: float, end_s: float,
        source_unit_id: str | None = None, frozen: bool = False,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO episodes
                   (video_id, episode_idx, start_s, end_s, source_unit_id, frozen)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(video_id, episode_idx) DO UPDATE SET
                       start_s=excluded.start_s, end_s=excluded.end_s,
                       source_unit_id=excluded.source_unit_id, frozen=excluded.frozen""",
                (video_id, episode_idx, start_s, end_s, source_unit_id, int(frozen)),
            )
            self._conn.commit()

    def episodes_for(
        self, video_id: str, frozen_only: bool = True, kind: str | None = "episode"
    ) -> list[EpisodeRow]:
        sql = "SELECT * FROM episodes WHERE video_id = ?"
        params: list[object] = [video_id]
        if frozen_only:
            sql += " AND frozen = 1"
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY episode_idx"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_episode_from_row(r) for r in rows]

    def get_episode(self, video_id: str, episode_idx: int) -> EpisodeRow | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM episodes WHERE video_id = ? AND episode_idx = ?",
                (video_id, episode_idx),
            ).fetchone()
        return _episode_from_row(row) if row else None

    def has_frozen_episodes(self, video_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM episodes WHERE video_id = ? AND frozen = 1", (video_id,)
            ).fetchone()
        return row["n"] > 0

    def set_frozen(self, video_id: str, frozen: bool) -> int:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE episodes SET frozen = ? WHERE video_id = ?", (int(frozen), video_id)
            )
            self._conn.commit()
            return cur.rowcount

    def replace_unfrozen_episodes(self, video_id: str, rows: list[EpisodeRow]) -> None:
        """Replace a video's proposed segmentation. Frozen boundaries are never overwritten."""
        with self._lock:
            frozen = self._conn.execute(
                "SELECT COUNT(*) AS n FROM episodes WHERE video_id = ? AND frozen = 1", (video_id,)
            ).fetchone()["n"]
            if frozen:
                raise ValueError(
                    f"{video_id}: episodes are frozen and cannot be replaced; unfreeze first "
                    f"(scripts/freeze_episodes.py --unfreeze {video_id})"
                )
            self._conn.execute("DELETE FROM episodes WHERE video_id = ?", (video_id,))
            self._conn.executemany(
                """INSERT INTO episodes (video_id, episode_idx, start_s, end_s, source_unit_id,
                       frozen, kind, probing_speaker, start_turn, end_turn, cut, topic, resolution)
                   VALUES (?,?,?,?,?,0,?,?,?,?,?,?,?)""",
                [
                    (video_id, r.episode_idx, r.start_s, r.end_s, r.source_unit_id, r.kind,
                     r.probing_speaker, r.start_turn, r.end_turn,
                     None if r.cut is None else int(r.cut), r.topic, r.resolution)
                    for r in rows
                ],
            )
            self._conn.commit()

    # --- decisions (Revision R2: outcomes, not episodes) ------------------
    def replace_decisions(self, video_id: str, rows: list[DecisionRow]) -> None:
        """Replace a video's investor decisions. Not subject to freeze: decisions are
        outcomes, not boundaries, so there is nothing here to protect from re-segmentation."""
        with self._lock:
            self._conn.execute("DELETE FROM decisions WHERE video_id = ?", (video_id,))
            self._conn.executemany(
                """INSERT INTO decisions
                       (video_id, speaker, decision, turn, terms, source_unit_id)
                   VALUES (?,?,?,?,?,?)""",
                [(video_id, r.speaker, r.decision, r.turn, r.terms, r.source_unit_id)
                 for r in rows],
            )
            self._conn.commit()

    def decisions_for(self, video_id: str) -> list[DecisionRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM decisions WHERE video_id = ? ORDER BY speaker", (video_id,)
            ).fetchall()
        return [_decision_from_row(r) for r in rows]

    def set_deal(self, video_id: str, deal: bool | None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE videos SET deal = ? WHERE video_id = ?",
                (None if deal is None else int(deal), video_id),
            )
            self._conn.commit()


def _video_from_row(row: sqlite3.Row) -> VideoRow:
    return VideoRow(
        video_id=row["video_id"], corpus=row["corpus"], path=row["path"],
        sha256=row["sha256"], duration_s=row["duration_s"], width=row["width"],
        height=row["height"],
        has_audio=None if row["has_audio"] is None else bool(row["has_audio"]),
        probe_ok=bool(row["probe_ok"]), meta_json=row["meta_json"] or "{}",
        skip_reason=row["skip_reason"],
        deal=None if row["deal"] is None else bool(row["deal"]),
    )


def _decision_from_row(row: sqlite3.Row) -> DecisionRow:
    return DecisionRow(
        video_id=row["video_id"], speaker=row["speaker"], decision=row["decision"],
        turn=row["turn"], terms=row["terms"], source_unit_id=row["source_unit_id"],
    )


def _episode_from_row(row: sqlite3.Row) -> EpisodeRow:
    return EpisodeRow(
        video_id=row["video_id"], episode_idx=row["episode_idx"], start_s=row["start_s"],
        end_s=row["end_s"], source_unit_id=row["source_unit_id"], frozen=bool(row["frozen"]),
        kind=row["kind"], probing_speaker=row["probing_speaker"], start_turn=row["start_turn"],
        end_turn=row["end_turn"], cut=None if row["cut"] is None else bool(row["cut"]),
        topic=row["topic"], resolution=row["resolution"],
    )


def _unit_from_row(row: sqlite3.Row) -> WorkUnit:
    return WorkUnit(
        unit_id=row["unit_id"], run_id=row["run_id"], video_id=row["video_id"],
        pass_name=row["pass_name"], span_start_s=row["span_start_s"],
        span_end_s=row["span_end_s"], prompt_hash=row["prompt_hash"], model=row["model"],
        media=row["media"], k=row["k"], status=row["status"], error_class=row["error_class"],
        error_detail=row["error_detail"], raw_path=row["raw_path"],
        parsed_path=row["parsed_path"], input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"], thinking_tokens=row["thinking_tokens"],
    )

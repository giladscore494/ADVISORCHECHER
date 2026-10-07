"""Durable research-run store: incremental checkpoints that survive Streamlit reruns and process restarts.

Backends (chosen by RESEARCH_STORE_URL):
  postgresql://user:pass@host:5432/db   External PostgreSQL (Supabase, Neon, RDS, ...). Use this on
                                        Streamlit Community Cloud or any host whose disk is ephemeral.
  sqlite:////absolute/path/runs.sqlite  Local SQLite file (WAL, synchronous=FULL). Survives process
                                        restarts on the same disk, NOT container replacement.
Default: sqlite at .research_runs/runs.sqlite next to the app (git-ignored; never committed).

Research content is private: it is written only to this store, never to the repository. Known API keys
are redacted from everything that is stored.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import get_setting

DEFAULT_SQLITE_PATH = Path(__file__).resolve().parent / ".research_runs" / "runs.sqlite"
STALE_AFTER_S = 120  # a "running" run without a heartbeat for this long is reported as interrupted
SECRET_SETTINGS = ("KIMI_API_KEY", "GLM_API_KEY", "OPENAI_API_KEY", "SERPER_API_KEY")
STATUSES = ("running", "completed", "failed", "interrupted")

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS research_runs (
        run_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        heartbeat_at TEXT,
        last_checkpoint_at TEXT,
        checkpoint_seq INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL,
        phase TEXT,
        label TEXT,
        domain TEXT,
        error TEXT,
        config TEXT,
        summary TEXT,
        state TEXT,
        final_report TEXT,
        partial_report TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS research_checkpoints (
        run_id TEXT NOT NULL,
        seq INTEGER NOT NULL,
        at TEXT NOT NULL,
        label TEXT,
        phase TEXT,
        status TEXT,
        summary TEXT,
        PRIMARY KEY (run_id, seq)
    )""",
]


class StoreError(Exception):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ts(dt: datetime | None = None) -> str:
    return (dt or utc_now()).isoformat(timespec="microseconds")


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def known_secrets() -> list[str]:
    values = []
    for name in SECRET_SETTINGS:
        v = get_setting(name)
        if v and len(v) >= 8:
            values.append(v)
    return values


def redact_text(text: str, secrets: list[str]) -> str:
    for s in secrets:
        text = text.replace(s, "[REDACTED]")
        escaped = json.dumps(s)[1:-1]
        if escaped != s:
            text = text.replace(escaped, "[REDACTED]")
    return text


def dumps(obj: Any, secrets: list[str]) -> str | None:
    if obj is None:
        return None
    return redact_text(json.dumps(obj, ensure_ascii=False, default=str), secrets)


def loads(text: str | None) -> Any:
    return json.loads(text) if text else None


class ResearchStore:
    """SQL store shared by the SQLite and PostgreSQL backends."""

    backend = "sql"
    placeholder = "?"

    def __init__(self, stale_after_s: int = STALE_AFTER_S, secrets: list[str] | None = None):
        self.stale_after_s = stale_after_s
        self.secrets = known_secrets() if secrets is None else secrets
        self._init_lock = threading.Lock()
        self._ready = False

    # -- backend hooks
    def _connect(self):
        raise NotImplementedError

    @property
    def durable_across_redeploys(self) -> bool:
        return False

    def describe(self) -> str:
        raise NotImplementedError

    # -- helpers
    def _q(self, sql: str) -> str:
        return sql if self.placeholder == "?" else sql.replace("?", self.placeholder)

    def _transaction(self, fn):
        """Run fn(cursor, q) in one transaction; q() adapts '?' placeholders to the backend."""
        self._ensure_schema()
        con = self._connect()
        try:
            out = fn(con.cursor(), self._q)
            con.commit()
            return out
        except Exception as exc:
            try:
                con.rollback()
            except Exception:  # noqa: BLE001
                pass
            raise StoreError(f"{self.backend} store error: {exc}") from exc
        finally:
            con.close()

    def _execute(self, statements: list[tuple[str, tuple]], fetch: bool = False):
        def run(cur, q):
            rows = None
            for sql, params in statements:
                cur.execute(q(sql), params)
                if fetch and cur.description is not None:
                    rows = cur.fetchall()
            return rows

        return self._transaction(run)

    def _ensure_schema(self) -> None:
        if self._ready:
            return
        with self._init_lock:
            if self._ready:
                return
            con = self._connect()
            try:
                cur = con.cursor()
                for sql in SCHEMA:
                    cur.execute(sql)
                con.commit()
            finally:
                con.close()
            self._ready = True

    # -- API
    def create_run(self, run_id: str, domain: str, config: dict, state: dict | None = None) -> None:
        now = ts()
        self._execute([(
            "INSERT INTO research_runs (run_id, created_at, updated_at, heartbeat_at, last_checkpoint_at, "
            "checkpoint_seq, status, phase, label, domain, config, state, summary) "
            "VALUES (?, ?, ?, ?, ?, 0, 'running', 'mapping', 'created', ?, ?, ?, ?)",
            (run_id, now, now, now, now, domain, dumps(config, self.secrets), dumps(state, self.secrets),
             dumps({}, self.secrets)))])

    def save_checkpoint(self, run_id: str, label: str, phase: str, state: dict, summary: dict,
                        status: str = "running", error: str = "") -> int:
        """Atomically replace the run's latest state and append a checkpoint log row. Returns the seq."""
        now = ts()
        state_json, summary_json = dumps(state, self.secrets), dumps(summary, self.secrets)

        def run(cur, q):
            cur.execute(q("UPDATE research_runs SET checkpoint_seq = checkpoint_seq + 1, updated_at = ?, "
                          "heartbeat_at = ?, last_checkpoint_at = ?, status = ?, phase = ?, label = ?, error = ?, "
                          "state = ?, summary = ? WHERE run_id = ?"),
                        (now, now, now, status, phase, label[:300], error[:4000], state_json, summary_json, run_id))
            cur.execute(q("SELECT checkpoint_seq FROM research_runs WHERE run_id = ?"), (run_id,))
            row = cur.fetchone()
            if row is None:
                raise StoreError(f"Unknown run {run_id}")
            cur.execute(q("INSERT INTO research_checkpoints (run_id, seq, at, label, phase, status, summary) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?)"), (run_id, row[0], now, label[:300], phase, status, summary_json))
            return row[0]

        return self._transaction(run)

    def touch(self, run_id: str, label: str | None = None, phase: str | None = None) -> None:
        """Heartbeat (and optionally the current activity) without rewriting the state."""
        now = ts()
        if label is None:
            self._execute([("UPDATE research_runs SET heartbeat_at = ? WHERE run_id = ? AND status = 'running'",
                            (now, run_id))])
        else:
            self._execute([("UPDATE research_runs SET heartbeat_at = ?, updated_at = ?, label = ?, "
                            "phase = COALESCE(?, phase) WHERE run_id = ? AND status = 'running'",
                            (now, now, label[:300], phase, run_id))])

    def finish(self, run_id: str, status: str, error: str = "", final_report: dict | None = None,
               partial_report: dict | None = None) -> None:
        if status not in STATUSES:
            raise ValueError(status)
        now = ts()
        self._execute([(
            "UPDATE research_runs SET status = ?, error = ?, updated_at = ?, heartbeat_at = ?, "
            "final_report = COALESCE(?, final_report), partial_report = COALESCE(?, partial_report) "
            "WHERE run_id = ?",
            (status, error[:4000], now, now, dumps(final_report, self.secrets),
             dumps(partial_report, self.secrets), run_id))])

    def mark_running(self, run_id: str, label: str = "resumed") -> None:
        now = ts()
        self._execute([("UPDATE research_runs SET status = 'running', error = '', heartbeat_at = ?, updated_at = ?, "
                        "label = ? WHERE run_id = ?", (now, now, label, run_id))])

    def _effective_status(self, status: str, heartbeat_at: str | None, now: datetime | None = None) -> str:
        if status != "running":
            return status
        hb = parse_ts(heartbeat_at)
        now = now or utc_now()
        if hb is None or now - hb > timedelta(seconds=self.stale_after_s):
            return "interrupted"
        return "running"

    def load(self, run_id: str, include_state: bool = True) -> dict | None:
        cols = ("run_id, created_at, updated_at, heartbeat_at, last_checkpoint_at, checkpoint_seq, status, phase, "
                "label, domain, error, config, summary, final_report, partial_report")
        if include_state:
            cols += ", state"
        rows = self._execute([(f"SELECT {cols} FROM research_runs WHERE run_id = ?", (run_id,))], fetch=True)
        if not rows:
            return None
        names = [c.strip() for c in cols.split(",")]
        rec = dict(zip(names, rows[0]))
        for k in ("config", "summary", "final_report", "partial_report", "state"):
            if k in rec:
                rec[k] = loads(rec[k])
        rec["stored_status"] = rec["status"]
        rec["status"] = self._effective_status(rec["status"], rec["heartbeat_at"])
        return rec

    def list_runs(self, limit: int = 20) -> list[dict]:
        rows = self._execute([(
            "SELECT run_id, created_at, updated_at, heartbeat_at, last_checkpoint_at, status, phase, label, domain "
            "FROM research_runs ORDER BY created_at DESC LIMIT ?", (int(limit),))], fetch=True) or []
        out = []
        for r in rows:
            rec = dict(zip(("run_id", "created_at", "updated_at", "heartbeat_at", "last_checkpoint_at", "status",
                            "phase", "label", "domain"), r))
            rec["status"] = self._effective_status(rec["status"], rec["heartbeat_at"])
            out.append(rec)
        return out

    def checkpoints(self, run_id: str) -> list[dict]:
        rows = self._execute([(
            "SELECT seq, at, label, phase, status, summary FROM research_checkpoints WHERE run_id = ? ORDER BY seq",
            (run_id,))], fetch=True) or []
        return [{"seq": r[0], "at": r[1], "label": r[2], "phase": r[3], "status": r[4], "summary": loads(r[5])}
                for r in rows]


class SQLiteStore(ResearchStore):
    backend = "sqlite"

    def __init__(self, path: str | Path, **kw):
        super().__init__(**kw)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self):
        con = sqlite3.connect(self.path, timeout=30, isolation_level="DEFERRED")
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=FULL")
        con.execute("PRAGMA busy_timeout=30000")
        return con

    def describe(self) -> str:
        return f"SQLite file {self.path} (survives process restarts on this disk; not container replacement)"


class PostgresStore(ResearchStore):
    backend = "postgresql"
    placeholder = "%s"

    def __init__(self, dsn: str, **kw):
        super().__init__(**kw)
        try:
            import psycopg  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dependency listed in requirements.txt
            raise StoreError("PostgreSQL store requires the 'psycopg' package.") from exc
        self.dsn = dsn

    def _connect(self):
        import psycopg

        return psycopg.connect(self.dsn, connect_timeout=10)

    @property
    def durable_across_redeploys(self) -> bool:
        return True

    def describe(self) -> str:
        from urllib.parse import urlparse

        p = urlparse(self.dsn)
        return f"PostgreSQL {p.hostname or 'localhost'}{(':' + str(p.port)) if p.port else ''}{p.path} (external, durable)"


def store_from_url(url: str | None = None, **kw) -> ResearchStore:
    url = url if url is not None else (get_setting("RESEARCH_STORE_URL") or "")
    if not url:
        return SQLiteStore(get_setting("RESEARCH_STORE_PATH") or DEFAULT_SQLITE_PATH, **kw)
    if url.startswith(("postgres://", "postgresql://")):
        return PostgresStore(url, **kw)
    if url.startswith("sqlite:///"):
        return SQLiteStore(url[len("sqlite:///"):] or DEFAULT_SQLITE_PATH, **kw)
    raise StoreError("RESEARCH_STORE_URL must start with postgresql://, postgres:// or sqlite:///")


_default: ResearchStore | None = None
_default_lock = threading.Lock()


def get_store() -> ResearchStore:
    global _default
    with _default_lock:
        if _default is None:
            _default = store_from_url()
        return _default


def reset_default_store() -> None:
    global _default
    with _default_lock:
        _default = None



class Checkpointer:
    """Binds a store to one run for the agent. Errors propagate (the agent records them as warnings)."""

    def __init__(self, store: ResearchStore, run_id: str):
        self.store = store
        self.run_id = run_id
        self.failures = 0
        self.saved = 0

    def save(self, label, phase, state, summary, status="running", error=""):
        try:
            seq = self.store.save_checkpoint(self.run_id, label, phase, state, summary, status=status, error=error)
        except Exception:
            self.failures += 1
            raise
        self.saved += 1
        return seq

    def touch(self, label=None, phase=None):
        self.store.touch(self.run_id, label, phase)

    def finish(self, status, error="", final_report=None, partial_report=None):
        self.store.finish(self.run_id, status, error, final_report=final_report, partial_report=partial_report)

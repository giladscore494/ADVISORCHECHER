"""Durable research store: SQLite and a real PostgreSQL server."""

from datetime import timedelta

import pytest

import research_store
from pg_server import PgServer


@pytest.fixture(scope="module")
def pg_dsn():
    try:
        import psycopg  # noqa: F401

        server = PgServer()
        dsn = server.start()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PostgreSQL not available: {exc}")
    yield dsn
    server.stop()


@pytest.fixture(params=["sqlite", "postgresql"])
def store(request, tmp_path):
    if request.param == "sqlite":
        return research_store.SQLiteStore(tmp_path / "runs.sqlite", secrets=["sk-secret-key-123456"])
    dsn = request.getfixturevalue("pg_dsn")
    s = research_store.PostgresStore(dsn, secrets=["sk-secret-key-123456"])
    s._execute([("DELETE FROM research_checkpoints", ()), ("DELETE FROM research_runs", ())])
    return s


def test_checkpoint_roundtrip_and_log(store):
    store.create_run("r1", "השכרת ציוד", {"provider": "kimi"})
    state = {"messages": [{"role": "user", "content": "שלום"}], "search_count": 2}
    assert store.save_checkpoint("r1", "tool search_web (step 1)", "searching", state, {"searches": 2}) == 1
    state["search_count"] = 3
    assert store.save_checkpoint("r1", "tool fetch_url (step 1)", "reading", state, {"searches": 3}) == 2
    rec = store.load("r1")
    assert rec["status"] == "running" and rec["checkpoint_seq"] == 2 and rec["phase"] == "reading"
    assert rec["state"]["search_count"] == 3 and rec["state"]["messages"][0]["content"] == "שלום"
    assert rec["summary"] == {"searches": 3} and rec["config"] == {"provider": "kimi"}
    assert [c["label"] for c in store.checkpoints("r1")] == ["tool search_web (step 1)", "tool fetch_url (step 1)"]
    assert store.list_runs()[0]["run_id"] == "r1"


def test_finish_and_reports(store):
    store.create_run("r2", "x", {})
    store.finish("r2", "failed", "Final report failed", partial_report={"type": "partial_report", "n": 1})
    rec = store.load("r2")
    assert rec["status"] == "failed" and rec["partial_report"]["n"] == 1 and rec["error"] == "Final report failed"
    store.mark_running("r2")
    store.finish("r2", "completed", final_report={"opportunities": []})
    rec = store.load("r2")
    assert rec["status"] == "completed" and rec["final_report"] == {"opportunities": []}
    assert rec["partial_report"]["n"] == 1  # earlier partial report kept
    with pytest.raises(ValueError):
        store.finish("r2", "bogus")


def test_stale_heartbeat_reported_as_interrupted(store):
    store.create_run("r3", "x", {})
    assert store.load("r3")["status"] == "running"
    store.stale_after_s = 0
    rec = store.load("r3")
    assert rec["status"] == "interrupted" and rec["stored_status"] == "running"
    assert store._effective_status("running", research_store.ts(research_store.utc_now() - timedelta(hours=1))) \
        == "interrupted"


def test_secrets_are_redacted(store, monkeypatch):
    store.create_run("r4", "x", {"note": "key sk-secret-key-123456"})
    store.save_checkpoint("r4", "l", "mapping", {"header": 'Bearer "sk-secret-key-123456"'}, {})
    rec = store.load("r4")
    assert "sk-secret" not in str(rec["state"]) and "sk-secret" not in str(rec["config"])
    assert "[REDACTED]" in rec["state"]["header"]


def test_unknown_run(store):
    assert store.load("nope") is None
    with pytest.raises(research_store.StoreError):
        store.save_checkpoint("nope", "l", "p", {}, {})


def test_store_from_url(tmp_path, monkeypatch):
    s = research_store.store_from_url(f"sqlite:///{tmp_path / 'a.sqlite'}")
    assert isinstance(s, research_store.SQLiteStore) and not s.durable_across_redeploys
    assert isinstance(research_store.store_from_url("postgresql://u:p@db.example.com:5432/x"),
                      research_store.PostgresStore)
    assert "p@" not in research_store.store_from_url("postgresql://u:p@db.example.com:5432/x").describe()
    with pytest.raises(research_store.StoreError):
        research_store.store_from_url("mysql://x")


def test_secrets_read_from_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-abcdef")
    assert "sk-test-openai-abcdef" in research_store.known_secrets()

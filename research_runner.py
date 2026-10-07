"""Runs research outside the Streamlit script, with durable checkpoints and a heartbeat.

Streamlit re-executes (and stops) the page script on every interaction, so a research run executed
inside the script dies with it. Here a run is executed in a worker thread owned by the server process:
the page only starts it and polls the durable store. While the worker is alive a heartbeat thread
refreshes `heartbeat_at`; if the process dies, the heartbeat stops and the store reports the run as
"interrupted" (stale heartbeat), and it can be resumed from its last checkpoint by any process.
"""

from __future__ import annotations

import collections
import secrets
import threading
import time
from typing import Callable

import agent
import llm
import partial_report
from research_store import Checkpointer, ResearchStore

HEARTBEAT_S = 20
LIVE_EVENTS_KEPT = 200

_active: dict[str, threading.Thread] = {}
_events: dict[str, collections.deque] = {}
_lock = threading.Lock()


class RunnerError(Exception):
    pass


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + secrets.token_hex(8)


def is_active(run_id: str) -> bool:
    with _lock:
        t = _active.get(run_id)
        return bool(t and t.is_alive())


def live_events(run_id: str) -> list[dict]:
    with _lock:
        return list(_events.get(run_id, ()))


def _heartbeat(store: ResearchStore, run_id: str, stop: threading.Event, interval: float) -> None:
    while not stop.wait(interval):
        try:
            store.touch(run_id)
        except Exception:  # noqa: BLE001 - a missed heartbeat is retried on the next tick
            pass


def _make_llm(factory: Callable | None, provider: str | None):
    if factory is not None:
        return factory(provider)
    return llm.LLMClient(provider)


def _execute(store: ResearchStore, run_id: str, provider: str | None, limits: agent.Limits, *,
             domain: str = "", instructions: str = "", state: dict | None = None,
             llm_factory: Callable | None = None, agent_kwargs: dict | None = None,
             on_event: Callable[[dict], None] | None = None, heartbeat_s: float = HEARTBEAT_S) -> agent.RunResult:
    stop = threading.Event()
    hb = threading.Thread(target=_heartbeat, args=(store, run_id, stop, heartbeat_s), daemon=True,
                          name=f"heartbeat-{run_id}")
    hb.start()
    buffer = _events.setdefault(run_id, collections.deque(maxlen=LIVE_EVENTS_KEPT))

    def emit(event: dict) -> None:
        buffer.append(event)
        if on_event is not None:
            on_event(event)

    try:
        try:
            client = _make_llm(llm_factory, provider)
        except llm.LLMError as exc:
            report = partial_report.build(state or {"run_id": run_id, "domain": domain, "instructions": instructions},
                                          status="failed", error=str(exc))
            store.finish(run_id, "failed", str(exc), partial_report=report)
            return agent.RunResult(domain=domain, error=str(exc), run_id=run_id, status="failed",
                                   partial_report=report)
        a = agent.ResearchAgent(client, limits=limits, on_event=emit, checkpointer=Checkpointer(store, run_id),
                                run_id=run_id, **(agent_kwargs or {}))
        result = a.resume(state) if state is not None else a.run(domain, instructions)
        result.trace["llm_warnings"] = list(getattr(client, "warnings", []))
        return result
    except BaseException as exc:
        # The agent persists its own failures; this only covers errors outside it.
        try:
            rec = store.load(run_id, include_state=False)
            if rec and rec["stored_status"] == "running":
                store.finish(run_id, "failed", f"Worker stopped: {exc.__class__.__name__}: {exc}")
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        stop.set()


def _launch(run_id: str, target: Callable, background: bool):
    if not background:
        return target()
    t = threading.Thread(target=target, daemon=True, name=f"research-{run_id}")
    with _lock:
        _active[run_id] = t
    t.start()
    return None


def start(store: ResearchStore, domain: str, instructions: str, limits: agent.Limits, provider: str | None = None,
          *, background: bool = True, llm_factory: Callable | None = None, agent_kwargs: dict | None = None,
          on_event: Callable[[dict], None] | None = None,
          heartbeat_s: float = HEARTBEAT_S) -> tuple[str, agent.RunResult | None]:
    """Create a durable run record, then execute it (in a worker thread unless background=False)."""
    run_id = new_run_id()
    provider = (provider or llm.provider_name()).lower()
    config = {"domain": domain, "instructions": instructions, "provider": provider,
              "model": llm.provider_model(provider) if provider in llm.PROVIDERS else "",
              "limits": limits.__dict__.copy(), "background": background}
    store.create_run(run_id, domain, config)

    def target():
        try:
            return _execute(store, run_id, provider, limits, domain=domain, instructions=instructions,
                            llm_factory=llm_factory, agent_kwargs=agent_kwargs, on_event=on_event,
                            heartbeat_s=heartbeat_s)
        finally:
            with _lock:
                _active.pop(run_id, None)

    return run_id, _launch(run_id, target, background)


def can_resume(record: dict | None) -> bool:
    return bool(record and record.get("state") and record["status"] in ("failed", "interrupted")
                and not is_active(record["run_id"]))


def resume(store: ResearchStore, run_id: str, *, background: bool = True, provider: str | None = None,
           llm_factory: Callable | None = None, agent_kwargs: dict | None = None,
           on_event: Callable[[dict], None] | None = None,
           heartbeat_s: float = HEARTBEAT_S) -> agent.RunResult | None:
    """Continue a failed or interrupted run from its last checkpoint."""
    record = store.load(run_id)
    if record is None:
        raise RunnerError(f"Unknown run {run_id}")
    if record["status"] == "running" or is_active(run_id):
        raise RunnerError("This run is still running.")
    if record["status"] == "completed":
        raise RunnerError("This run already completed.")
    if not record.get("state"):
        raise RunnerError("No resumable state was saved for this run.")
    config = record.get("config") or {}
    provider = provider or config.get("provider") or llm.provider_name()
    limits = agent.Limits(**{k: v for k, v in (config.get("limits") or {}).items()
                             if k in agent.Limits.__dataclass_fields__})
    store.mark_running(run_id, "resuming from last checkpoint")
    state = record["state"]

    def target():
        try:
            return _execute(store, run_id, provider, limits, domain=record.get("domain") or "", state=state,
                            llm_factory=llm_factory, agent_kwargs=agent_kwargs, on_event=on_event,
                            heartbeat_s=heartbeat_s)
        finally:
            with _lock:
                _active.pop(run_id, None)

    return _launch(run_id, target, background)


def wait(run_id: str, timeout: float | None = None) -> bool:
    """Block until a background run's worker thread ends (tests and scripts). Returns True if it ended."""
    with _lock:
        t = _active.get(run_id)
    if t is None:
        return True
    t.join(timeout)
    return not t.is_alive()


def report_for(record: dict) -> dict | None:
    """The stored partial report, or one generated now from the last checkpoint (e.g. after a crash)."""
    if record.get("partial_report") and record["status"] != "interrupted":
        return record["partial_report"]
    if not record.get("state"):
        return record.get("partial_report")
    error = record.get("error") or ("The run was interrupted (server restart, timeout or crash) before it "
                                    "finished." if record["status"] == "interrupted" else "")
    return partial_report.build(record["state"], status=record["status"], error=error,
                                checkpoint_at=record.get("last_checkpoint_at") or "")

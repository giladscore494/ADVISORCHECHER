"""Checkpointing, crash recovery, resume and partial reports."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import agent
import partial_report
import research_runner
import research_store
from helpers import EXCERPT, FakeLLM, fake_fetch, fake_search, text_response, tool_response, valid_report
from llm import LLMError

URL = "https://www.gov.il/he/departments/legalInfo/regulation-x"
HERE = Path(__file__).resolve().parent


@pytest.fixture
def store(tmp_path):
    return research_store.SQLiteStore(tmp_path / "runs.sqlite")


RESEARCH = [
    tool_response(("search_web", {"query": "פטור רישיון השכרה", "phase": "searching"})),
    tool_response(("fetch_url", {"url": URL, "phase": "reading"}),
                  ("update_candidates", {"candidates": [{"name": "Equipment rental", "status": "surviving",
                                                         "mechanism": "Regulation X §4"},
                                                        {"name": "Courier", "status": "rejected",
                                                         "reason": "licence required"}]})),
    tool_response(("record_findings", {
        "findings": [{"statement": "Regulation X requires a yearly inspection", "source_url": URL, "excerpt": EXCERPT},
                     {"statement": "Invented claim", "source_url": URL, "excerpt": "text that is not in the source"}],
        "open_questions": ["Does §5 cover commercial use?"]})),
]


def start(store, responses, **kw):
    llm = FakeLLM(responses)
    run_id, run = research_runner.start(store, "equipment rental", "", agent.Limits(max_steps=10), "kimi",
                                        background=False, llm_factory=lambda p: llm,
                                        agent_kwargs={"search_fn": fake_search, "fetch_fn": fake_fetch, **kw})
    return run_id, run, llm


def test_checkpoint_after_every_model_response_tool_call_and_phase(store):
    run_id, run, _ = start(store, RESEARCH + [text_response("DONE"), text_response(json.dumps(valid_report(URL)))])
    assert run.status == "completed"
    labels = [c["label"] for c in store.checkpoints(run_id)]
    assert labels[0] == "research started"
    for expected in ("model response (step 1)", "tool search_web (step 1)", "tool fetch_url (step 2)",
                     "tool update_candidates (step 2)", "tool record_findings (step 3)",
                     "phase: Mapping regulatory environment -> Searching for laws and regulations",
                     "research loop ended: model finished research", "before final report (attempt 1)",
                     "final report received (attempt 1)", "finished: completed"):
        assert expected in labels, expected
    seqs = [c["seq"] for c in store.checkpoints(run_id)]
    assert seqs == list(range(1, len(seqs) + 1))
    rec = store.load(run_id)
    assert rec["status"] == "completed" and rec["final_report"]["opportunities"][0]["verification_status"] == "verified"
    st = rec["state"]
    assert st["search_count"] == 1 and st["fetch_count"] == 1 and st["trace"]["token_usage"]["model_calls"] == 5
    assert st["findings"][0]["verified"] is True and st["findings"][1]["verified"] is False
    assert st["open_questions"] == ["Does §5 cover commercial use?"]
    assert rec["summary"]["verified_findings"] == 1


class CrashingLLM(FakeLLM):
    def __init__(self, responses, exc):
        super().__init__(responses)
        self.exc = exc

    def chat(self, messages, tools=None, json_mode=False):
        if json_mode:
            raise self.exc
        return super().chat(messages, tools, json_mode)


@pytest.mark.parametrize("exc", [RuntimeError("connection reset while generating JSON"), LLMError("timed out")])
def test_crash_during_final_json_generation_keeps_evidence(store, exc):
    llm = CrashingLLM(RESEARCH + [text_response("DONE")], exc)
    run_id, run = research_runner.start(store, "equipment rental", "", agent.Limits(max_steps=10), "kimi",
                                        background=False, llm_factory=lambda p: llm,
                                        agent_kwargs={"search_fn": fake_search, "fetch_fn": fake_fetch})
    assert run.result is None and run.status == "failed"
    rec = store.load(run_id)
    assert rec["status"] == "failed" and rec["final_report"] is None
    report = rec["partial_report"]
    assert report["type"] == "partial_report" and "NOT a validated final report" in report["notice"]
    assert [f["statement"] for f in report["verified_findings"]] == ["Regulation X requires a yearly inspection"]
    assert [f["statement"] for f in report["unverified_findings"]] == ["Invented claim"]
    assert {c["name"]: c["status"] for c in report["candidates"]} == {"Equipment rental": "surviving",
                                                                     "Courier": "rejected"}
    assert all(c["label"].startswith("WORKING HYPOTHESIS") for c in report["candidates"])
    assert report["official_sources_retrieved"][0]["url"] == URL
    assert report["searches"][0]["query"] == "פטור רישיון השכרה"
    assert report["open_questions"] == ["Does §5 cover commercial use?"]
    assert any(e["source"] in ("agent", "model") for e in report["api_errors"])
    md = partial_report.to_markdown(report)
    assert "VERIFIED findings" in md and "WORKING HYPOTHESES" in md and "Run ID" in md


def test_invalid_final_json_twice_produces_labelled_partial_report(store):
    run_id, run, _ = start(store, RESEARCH + [text_response("DONE"), text_response("not json"),
                                              text_response('{"opportunities": "still wrong"}')])
    assert run.status == "failed" and "did not return a valid report" in run.error
    report = store.load(run_id)["partial_report"]
    assert report["unvalidated_model_draft"]["label"].startswith("UNVALIDATED MODEL OUTPUT")
    assert "still wrong" in report["unvalidated_model_draft"]["text"]


def test_resume_after_failed_finalization_completes_without_repeating_research(store):
    llm = CrashingLLM(RESEARCH + [text_response("DONE")], LLMError("timed out"))
    run_id, run = research_runner.start(store, "equipment rental", "", agent.Limits(max_steps=10), "kimi",
                                        background=False, llm_factory=lambda p: llm,
                                        agent_kwargs={"search_fn": fake_search, "fetch_fn": fake_fetch})
    assert store.load(run_id)["status"] == "failed"
    searches = []

    def counting_search(q, num_results=10):
        searches.append(q)
        return fake_search(q, num_results)

    second = FakeLLM([text_response(json.dumps(valid_report(URL)))])
    run2 = research_runner.resume(store, run_id, background=False, llm_factory=lambda p: second,
                                  agent_kwargs={"search_fn": counting_search, "fetch_fn": fake_fetch})
    assert run2.status == "completed" and searches == []
    assert second.calls[0]["json_mode"] is True and len(second.calls) == 1
    # The finalize prompt is not duplicated in the history.
    assert sum(1 for m in second.calls[0]["messages"] if "Research is over" in str(m.get("content"))) == 1
    rec = store.load(run_id)
    assert rec["status"] == "completed"
    # Evidence restored from the checkpoint verifies the excerpt even though nothing was refetched.
    assert rec["final_report"]["opportunities"][0]["verification_status"] == "verified"
    assert rec["state"]["trace"]["resumes"][0]["loop_done"] is True
    with pytest.raises(research_runner.RunnerError, match="already completed"):
        research_runner.resume(store, run_id, background=False)


def test_resume_mid_loop_repairs_dangling_tool_calls(store):
    # Simulate a crash between a model response and its tool results.
    a = agent.ResearchAgent(FakeLLM([text_response("x")]), search_fn=fake_search, fetch_fn=fake_fetch, run_id="r")
    a.domain = "x"
    a.messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                  {"role": "assistant", "content": "", "tool_calls": [
                      {"id": "c9", "type": "function", "function": {"name": "search_web", "arguments": "{}"}}]}]
    a.step = 1
    state = a.export_state()
    llm = FakeLLM([text_response("DONE"), text_response(json.dumps({"opportunities": []}))])
    b = agent.ResearchAgent(llm, search_fn=fake_search, fetch_fn=fake_fetch)
    run = b.resume(state)
    assert run.status == "completed"
    tool_msgs = [m for m in llm.calls[0]["messages"] if m.get("role") == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "c9" and "interrupted" in tool_msgs[0]["content"]
    assert run.trace["resumes"][0]["repaired_tool_calls"] == 1


def test_keyboard_interrupt_is_persisted_as_interrupted(store):
    class Interrupting(FakeLLM):
        def chat(self, messages, tools=None, json_mode=False):
            if len(self.calls) == 2:
                raise KeyboardInterrupt
            return super().chat(messages, tools, json_mode)

    llm = Interrupting(RESEARCH)
    with pytest.raises(KeyboardInterrupt):
        research_runner.start(store, "x", "", agent.Limits(), "kimi", background=False, llm_factory=lambda p: llm,
                              agent_kwargs={"search_fn": fake_search, "fetch_fn": fake_fetch})
    rec = store.list_runs()[0]
    full = store.load(rec["run_id"])
    assert full["status"] == "interrupted" and full["state"]["search_count"] == 1
    assert research_runner.can_resume(full)


def test_background_runner_and_heartbeat(store):
    llm = FakeLLM(RESEARCH + [text_response("DONE"), text_response(json.dumps(valid_report(URL)))])
    run_id, none = research_runner.start(store, "equipment rental", "", agent.Limits(max_steps=10), "kimi",
                                         llm_factory=lambda p: llm, heartbeat_s=0.05,
                                         agent_kwargs={"search_fn": fake_search, "fetch_fn": fake_fetch})
    assert none is None
    assert research_runner.wait(run_id, timeout=30)
    rec = store.load(run_id)
    assert rec["status"] == "completed" and not research_runner.is_active(run_id)
    assert any("Candidate funnel" in e["message"] for e in research_runner.live_events(run_id))


def test_missing_provider_key_fails_cleanly(store, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("llm.get_setting", lambda name, default=None: default)
    run_id, run = research_runner.start(store, "x", "", agent.Limits(), "openai", background=False)
    assert run.status == "failed" and "OPENAI_API_KEY" in run.error
    assert store.load(run_id)["status"] == "failed"


def test_checkpoint_failures_do_not_kill_research(tmp_path):
    class Broken(research_store.SQLiteStore):
        def save_checkpoint(self, *a, **k):
            raise research_store.StoreError("disk full")

    store = Broken(tmp_path / "x.sqlite")
    run_id, run, _ = start(store, [text_response("DONE"), text_response(json.dumps({"opportunities": []}))])
    assert run.status == "completed"
    assert any("Checkpoint failed" in w for w in run.trace["warnings"])


# ----------------------------------------------------------- real process crash
def _crash(tmp_path, hang_at):
    db, marker = tmp_path / "runs.sqlite", tmp_path / "marker.json"
    proc = subprocess.Popen([sys.executable, str(HERE / "crash_worker.py"), str(db), hang_at, str(marker)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.time() + 60
        while not marker.exists() or not marker.read_text():
            if proc.poll() is not None:
                raise AssertionError(proc.stderr.read().decode()[-3000:])
            if time.time() > deadline:
                raise AssertionError("worker never reached the crash point")
            time.sleep(0.1)
        time.sleep(0.2)
        info = json.loads(marker.read_text())
        os.kill(proc.pid, signal.SIGKILL)  # hard crash: no exception handlers or finally blocks run
        proc.wait(10)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == -signal.SIGKILL
    store = research_store.SQLiteStore(db, stale_after_s=1)
    time.sleep(1.5)  # heartbeat (every 0.3s) has stopped with the process
    return store, info["run_id"]


def test_process_killed_mid_research_is_recoverable_and_resumable(tmp_path):
    store, run_id = _crash(tmp_path, "loop")
    rec = store.load(run_id)
    assert rec["stored_status"] == "running" and rec["status"] == "interrupted"
    state = rec["state"]
    assert state["search_count"] == 1 and state["fetch_count"] == 1 and state["step"] == 3
    assert state["findings"][0]["verified"] is True
    labels = [c["label"] for c in store.checkpoints(run_id)]
    assert "tool record_findings (step 3)" in labels
    report = research_runner.report_for(rec)
    assert report["status"] == "interrupted" and "interrupted" in report["error"]
    assert report["verified_findings"][0]["statement"] == "Regulation X requires a yearly inspection"
    assert report["candidates"][0]["name"] == "Equipment rental"
    assert research_runner.can_resume(rec)
    llm = FakeLLM([text_response("DONE"), text_response(json.dumps(valid_report(URL)))])
    run = research_runner.resume(store, run_id, background=False, llm_factory=lambda p: llm,
                                 agent_kwargs={"search_fn": fake_search, "fetch_fn": fake_fetch})
    assert run.status == "completed"
    # The resumed conversation continues where the killed process stopped.
    assert llm.calls[0]["tools"] and [m["role"] for m in llm.calls[0]["messages"]][-1] == "tool"
    rec = store.load(run_id)
    assert rec["status"] == "completed" and rec["state"]["search_count"] == 1
    assert rec["final_report"]["opportunities"][0]["verification_status"] == "verified"


def test_process_killed_during_final_json_generation(tmp_path):
    store, run_id = _crash(tmp_path, "final")
    rec = store.load(run_id)
    assert rec["status"] == "interrupted" and rec["state"]["loop_done"] is True
    assert rec["label"].startswith("before final report") or rec["label"].startswith("Waiting")
    report = research_runner.report_for(rec)
    assert report["official_sources_retrieved"] and report["verified_findings"]
    llm = FakeLLM([text_response(json.dumps(valid_report(URL)))])
    run = research_runner.resume(store, run_id, background=False, llm_factory=lambda p: llm,
                                 agent_kwargs={"search_fn": fake_search, "fetch_fn": fake_fetch})
    assert run.status == "completed" and llm.calls[0]["json_mode"] is True and len(llm.calls) == 1

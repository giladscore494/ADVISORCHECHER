"""Subprocess used by test_recovery: runs research against a durable SQLite store, then hangs at a chosen
point so the test can SIGKILL the whole process (a real crash: no finally blocks, no cleanup).

usage: python crash_worker.py <store.sqlite> <hang_at: loop|final> <marker_file>
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import agent  # noqa: E402
import research_runner  # noqa: E402
import research_store  # noqa: E402
from helpers import fake_fetch, fake_search, text_response, tool_response  # noqa: E402

URL = "https://www.gov.il/he/departments/legalInfo/regulation-x"


class HangingLLM:
    provider, model = "fake", "fake-model"

    def __init__(self, responses, hang_at, marker, run_id_ref):
        self.responses = list(responses)
        self.hang_at = hang_at
        self.marker = marker
        self.run_id_ref = run_id_ref
        self.warnings = []

    def chat(self, messages, tools=None, json_mode=False):
        hang = (self.hang_at == "final" and json_mode) or (self.hang_at == "loop" and not self.responses)
        if hang:
            with open(self.marker, "w") as fh:
                json.dump({"run_id": self.run_id_ref["id"], "pid": os.getpid()}, fh)
            time.sleep(600)  # killed here
        return self.responses.pop(0)


def main(db, hang_at, marker):
    store = research_store.SQLiteStore(db)
    responses = [
        tool_response(("search_web", {"query": "פטור רישיון השכרה", "phase": "searching"})),
        tool_response(("fetch_url", {"url": URL, "phase": "reading"}),
                      ("update_candidates", {"candidates": [{"name": "Equipment rental", "status": "surviving",
                                                             "mechanism": "Regulation X §4"}]})),
        tool_response(("record_findings", {"findings": [{
            "statement": "Regulation X requires a yearly inspection", "source_url": URL,
            "excerpt": "בעל נגרר יבצע בדיקה תקופתית אחת לשנה"}], "open_questions": ["Does §5 cover commercial use?"]})),
    ]
    if hang_at == "final":
        responses.append(text_response("DONE"))
    ref = {"id": ""}
    original = research_runner.new_run_id

    def run_id():
        ref["id"] = original()
        return ref["id"]

    research_runner.new_run_id = run_id
    research_runner.start(store, "equipment rental", "", agent.Limits(max_steps=10), "fake", background=False,
                          llm_factory=lambda p: HangingLLM(responses, hang_at, marker, ref),
                          agent_kwargs={"search_fn": fake_search, "fetch_fn": fake_fetch}, heartbeat_s=0.3)


if __name__ == "__main__":
    main(*sys.argv[1:4])

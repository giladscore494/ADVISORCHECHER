"""Focused integration test: the HS 7318 scenario replayed against this tree (scripts/benchmark_hs7318.py).

Real: agent loop, fetcher/PDF extraction, document store, compaction, verification, and the committed
government snapshots for all 29 local queries. Simulated: the model's decisions, Serper, the documents.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_hs7318_replay(tmp_path):
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "benchmark_hs7318.py"), "--repo", str(ROOT)],
                          capture_output=True, text=True, timeout=900, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr[-3000:]
    r = json.loads(proc.stdout.strip().splitlines()[-1])
    # Same research profile as run 20261007-025045-54c043110af66f55.
    assert (r["model_calls"], r["searches"], r["fetches"], r["local_queries"]) == (12, 12, 14, 29)
    # Complete legal document extraction: the schedule on page 120 of the 140-page order is reachable.
    ext = r["extraction_fio_order"]
    assert ext["pages_extracted"] == 140 and ext["status"] == "complete" and ext["schedule_page_120_retrievable"]
    # Citations: every real quote verified (with pages), the fabricated one rejected, a wrong page flagged.
    c = r["citations"]
    assert c["real_quotes_source_verified"] == c["real_quotes"] == 5
    assert c["fabricated_quotes_accepted"] == 0 and c["wrong_page_citations_flagged"] == 1
    # Context stays bounded: compaction ran and no call grows with the whole history.
    assert r["compactions"] >= 1 and max(r["estimated_input_tokens_per_call"]) < 60_000
    assert r["estimated_input_tokens_total"] < 450_000
    # Exact standard matching: no noise for ISO 4032 / ISO 7089 (the old search returned 60-217 records).
    assert set(r["local_noise_total_matches"].values()) == {0}
    # Verified quotes never become verified legal conclusions on their own; exemptions stay unresolved.
    v = r["verification_counters"]
    assert v["source_verified"] == 5 and v["legal_verified"] == 0 and v["negative_unresolved"] == 2
    assert all(o["class"] == "B" for o in r["opportunities"])
    assert r["checklist"]["validity"] == "unresolved" and len(r["unresolved_questions"]) == 3

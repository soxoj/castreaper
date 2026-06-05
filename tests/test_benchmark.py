#!/usr/bin/env python3
"""
Reference benchmark for castreaper.

Runs the real `castreaper.py` CLI over `test.mp4` and measures how many of the
ground-truth entities (curated in ground_truth.json, all visually verified in
the screencast) the tool actually surfaces in its output.

This is an EFFECTIVENESS benchmark, not a unit test: there is one fixed input
(test.mp4) with a known set of OSINT artifacts, and the metric is recall. As
bugs get fixed and extraction improves, recall goes up — so this file is the
yardstick for "did that change actually make the tool better?".

Run it two ways:

    pytest tests/test_benchmark.py -v -s        # CI / regression guard
    python  tests/test_benchmark.py             # human-readable scorecard

Workflow for tracking improvements
----------------------------------
1. Make a change to castreaper.py (fix a bug, sample more frames, better OCR...).
2. Run the scorecard:  python tests/test_benchmark.py
3. If recall went UP, raise the matching number in BASELINE below and commit.
   The pytest guard then locks in the gain — any future regression fails CI.

Matching is OCR-noise tolerant: text is normalised to lowercase alphanumerics
before substring matching (so "facehook.com " still matches "facebook.com"...
no — character-level OCR errors like 'h'->'b' are intentionally NOT forgiven, so
the score is also sensitive to OCR quality, not just to coverage).
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "castreaper.py"
VIDEO = REPO / "test.mp4"
GROUND_TRUTH = Path(__file__).resolve().parent / "ground_truth.json"

# Wall-clock budget for one full run over test.mp4 (sparse sampling -> a few s,
# but tesseract on full frames can be slow; give it room).
RUN_TIMEOUT_S = 300

# ---------------------------------------------------------------------------
# Recorded baseline. These are the recall figures the CURRENT code achieves.
# A change is an improvement iff it pushes a number above its baseline.
# RAISE these (never lower) when an improvement lands, so the guard locks it in.
#
# History:
#   initial commit : crashes before any output (UnboundLocalError in
#                    get_video_params; then a cv2.rectangle read-only-array
#                    error) -> recall 0.0 everywhere.
#   3 crashes fixed: domains 66.7% (10/15), emails 0% (sparse frame sampling
#                    never hits the email frames), urls 100% (1/1), weighted 33.3%.
#
# Numbers are set slightly below the last measurement so minor cross-machine
# tesseract variance can't make the guard flaky; raise them as you improve.
# ---------------------------------------------------------------------------
BASELINE = {
    "emails": 0.0,    # fraction of ground-truth emails surfaced
    "domains": 0.6,   # measured 0.667 (10/15)
    "urls": 0.9,      # measured 1.0 (1/1)
    "weighted": 0.30,  # measured 0.333
}


def _norm(s: str) -> str:
    """Lowercase, keep only [a-z0-9] — collapses OCR spacing/punctuation noise."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _item_value(item):
    """Display/canonical form of a ground-truth item (str or {value, match})."""
    return item["value"] if isinstance(item, dict) else item


def _item_needle(item) -> str:
    """Normalised substring that counts as 'surfaced'.

    For dict items use the OCR-stable `match` (falling back to value); this lets
    a noisy URL be tracked by a reliably-recognised prefix instead of its full,
    OCR-mangled random token.
    """
    if isinstance(item, dict):
        return _norm(item.get("match", item["value"]))
    return _norm(item)


def load_ground_truth() -> dict:
    return json.loads(GROUND_TRUTH.read_text())


def run_tool(tmp_dir: Path) -> dict:
    """Run castreaper.py on test.mp4 inside tmp_dir; return run artifacts.

    The tool writes image_to_text.txt and <video>_output/ into its CWD, so we
    run it in an isolated temp dir and read everything it produced back out.
    Never raises on tool failure — a crash is a legitimate (bad) result that
    the benchmark must be able to score as recall 0.
    """
    proc = subprocess.run(
        [sys.executable, str(TOOL), str(VIDEO)],
        cwd=tmp_dir,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
    )
    raw_txt = ""
    raw_file = tmp_dir / "image_to_text.txt"
    if raw_file.exists():
        raw_txt = raw_file.read_text(errors="replace")

    stdout = proc.stdout or ""
    # The structured entities section (what the tool classifies as URLs/emails).
    entities_section = ""
    if "Recognized entities:" in stdout:
        entities_section = stdout.split("Recognized entities:", 1)[1]

    return {
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": proc.stderr or "",
        "raw_txt": raw_txt,
        "entities_section": entities_section,
        # Everything the tool emitted, for "did it surface this at all" matching.
        "blob_norm": _norm(stdout + "\n" + raw_txt),
    }


def score(result: dict, gt: dict) -> dict:
    """Compute per-tier found/missing and recall against the tool output."""
    blob = result["blob_norm"]
    report = {}
    for tier, spec in gt["tiers"].items():
        items = spec["items"]
        found, missing = [], []
        for item in items:
            (found if _item_needle(item) in blob else missing).append(_item_value(item))
        report[tier] = {
            "weight": spec.get("weight", 1),
            "found": found,
            "missing": missing,
            "recall": (len(found) / len(items)) if items else 1.0,
        }

    # Weighted overall recall across scored tiers (weight 0 tiers excluded).
    num = sum(r["recall"] * r["weight"] for r in report.values())
    den = sum(r["weight"] for r in report.values())
    report["_weighted"] = num / den if den else 0.0
    return report


def format_scorecard(result: dict, report: dict) -> str:
    lines = []
    lines.append("=" * 64)
    lines.append("castreaper benchmark — test.mp4")
    lines.append("=" * 64)
    rc = result["returncode"]
    lines.append(f"tool exit code : {rc}" + ("  (CRASHED)" if rc != 0 else ""))
    if rc != 0:
        tail = "\n".join(result["stderr"].strip().splitlines()[-3:])
        lines.append(f"stderr (tail)  : {tail}")
    lines.append("")
    for tier, r in report.items():
        if tier.startswith("_"):
            continue
        tag = "  (not scored)" if r["weight"] == 0 else ""
        lines.append(f"[{tier}] recall {r['recall']*100:5.1f}%  "
                     f"({len(r['found'])}/{len(r['found'])+len(r['missing'])}){tag}")
        if r["found"]:
            lines.append(f"    found  : {', '.join(r['found'])}")
        if r["missing"]:
            lines.append(f"    MISSING: {', '.join(r['missing'])}")
    lines.append("")
    lines.append(f"WEIGHTED RECALL: {report['_weighted']*100:.1f}%  "
                 f"(baseline {BASELINE['weighted']*100:.1f}%)")
    lines.append("=" * 64)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# pytest entry points
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def run(tmp_path_factory):
    assert VIDEO.exists(), f"missing test video: {VIDEO}"
    tmp = tmp_path_factory.mktemp("castreaper_run")
    return run_tool(tmp)


@pytest.fixture(scope="module")
def report(run):
    return score(run, load_ground_truth())


def test_scorecard(run, report):
    """Always prints the scorecard (use `pytest -s` to see it)."""
    print("\n" + format_scorecard(run, report))


def test_tool_runs_without_crashing(run):
    assert run["returncode"] == 0, run["stderr"][-2000:]


def test_no_recall_regression(report):
    """Locks in current effectiveness. Bump BASELINE when you improve the tool."""
    assert report["emails"]["recall"] >= BASELINE["emails"], "emails recall regressed"
    assert report["domains"]["recall"] >= BASELINE["domains"], "domains recall regressed"
    assert report["urls"]["recall"] >= BASELINE["urls"], "urls recall regressed"
    assert report["_weighted"] >= BASELINE["weighted"], "weighted recall regressed"


# --------------------------------------------------------------------------- #
# Script mode: human-readable scorecard
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        res = run_tool(Path(d))
    rep = score(res, load_ground_truth())
    print(format_scorecard(res, rep))
    # Non-zero exit if we slipped below the recorded baseline.
    regressed = (
        rep["emails"]["recall"] < BASELINE["emails"]
        or rep["domains"]["recall"] < BASELINE["domains"]
        or rep["_weighted"] < BASELINE["weighted"]
    )
    sys.exit(1 if regressed else 0)

"""NFR 5: the uptime probe's arithmetic and outage grouping."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "uptime_probe", Path(__file__).resolve().parents[1] / "tools" / "uptime_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _rows(pattern: str) -> list[dict]:
    return [{"at": f"t{i}", "base": "http://x", "up": c == "u", "status": 200 if c == "u" else 503}
            for i, c in enumerate(pattern)]


def test_uptime_is_the_share_of_ready_probes():
    doc = probe.summarise(_rows("u" * 99 + "d"))
    assert doc["probes"] == 100 and doc["up"] == 99
    assert doc["uptime"] == 0.99 and doc["meets_target"] is True


def test_consecutive_failures_are_one_outage():
    doc = probe.summarise(_rows("uuddduudu"))
    assert [(o["from"], o["to"], o["probes"]) for o in doc["outages"]] == [
        ("t2", "t4", 3), ("t7", "t7", 1)]
    assert doc["meets_target"] is False


def test_an_outage_still_open_at_the_end_is_reported():
    doc = probe.summarise(_rows("uudd"))
    assert doc["outages"] == [{"from": "t2", "to": "t3", "probes": 2}]


def test_no_probes_is_not_a_pass():
    assert probe.summarise([]) == {"probes": 0}

#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT
"""Map scoped observations to disclosure dimensions without inventing assurances."""

from __future__ import annotations

from typing import Any

DIMENSIONS = [
    "Data sensitivity", "Data minimization", "Collection purpose", "Protection mechanisms",
    "Privacy budget / leakage control", "Membership inference", "Reconstruction / memorization",
    "Fairness / group impacts", "Retention / deletion", "User controls", "Regulatory alignment",
    "Computational overhead", "Evidence transparency",
]


def build_disclosure(model: dict[str, Any], demo: bool = False) -> list[dict[str, str]]:
    """Generate evidence pointers into the enclosing model object, not certifications."""
    result = {name: {"dimension": name, "status": "not evaluated", "scope": "Deployment",
                     "finding": "Requires deployment-specific documentation or an audit.",
                     "evidence": ""} for name in DIMENSIONS}

    def set_row(name: str, status: str, scope: str, finding: str, evidence: str) -> None:
        result[name].update(status=status, scope=scope, finding=finding, evidence=evidence)

    set_row("Data sensitivity", "documented", "Benchmark fixture only",
            "Generated tokens and colors only; no real personal data in the fixture.", "#/records")
    set_row("Collection purpose", "documented", "Benchmark only",
            "Evaluate synthetic context disclosure and optional controlled fine-tuning.", "#/config")
    set_row("Protection mechanisms", "documented", "Prompt-level mitigation only",
            "Basic confidentiality instruction vs expanded instruction; neither enforces access control.",
            "conditions/*/trials/*/messages")
    set_row("Evidence transparency", "documented", "Benchmark only",
            "Raw outputs, prompt hashes, configuration, revisions, and artifact checksum; not an audit.",
            "conditions; metadata; #/provenance")
    if model.get("conditions"):
        set_row("Privacy budget / leakage control", "partial measurement", "Synthetic context only",
                "Exact-token disclosure observed under four prompts. Epsilon/delta not evaluated.",
                "conditions/*/summary/leakage")
        set_row("Computational overhead", "partial measurement", "Local inference only",
                "Synchronized latency includes prompt length effects; not privacy-mechanism overhead.",
                "conditions/*/summary/latency_median_s")
    if model.get("memorization", {}).get("status") == "measured":
        set_row("Membership inference", "partial measurement", "Local fine-tuning only",
                "Known synthetic members vs held-out records; loss attack AUC before/after training.",
                "memorization/metrics")
        set_row("Reconstruction / memorization", "partial measurement", "Local fine-tuning only",
                "Greedy exact-canary extraction; not general reconstruction resistance.",
                "memorization/records")
    if demo:
        for row in result.values():
            if row["status"] != "not evaluated":
                row["status"] = "illustrative only"
                row["scope"] = "DEMO, not a model assessment"
    return list(result.values())
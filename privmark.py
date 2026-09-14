#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT
"""Run the PrivMark exploratory local-LLM protocol and save auditable artifacts.

Examples:
    uv run python privmark.py demo
    uv run python privmark.py run --records 20
    uv run python privmark.py run --models HuggingFaceTB/SmolLM2-360M-Instruct --memorize
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from privmark_data import ATTACKS, MODELS, PROTOCOL_VERSION, make_messages, make_records
from privmark_disclosure import build_disclosure
from privmark_metrics import (
    contains_secret,
    membership_metrics,
    paired_difference,
    summarize_trials,
)

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = "1.0"


def provenance() -> dict[str, Any]:
    """Capture software versions and source hashes; no host name or credentials."""
    versions = {}
    for package in ("torch", "transformers", "numpy", "scikit-learn", "streamlit",
                    "plotly", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions,
            "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(ROOT.glob("*.py"))}}


def new_artifact(mode: str, config: dict, records: list[dict]) -> dict[str, Any]:
    """Create an explicitly labeled run, never mixing demo and measured values."""
    now = datetime.now(timezone.utc)
    return {"schema_version": SCHEMA_VERSION,
            "run_id": f"{mode}-{now:%Y%m%dT%H%M%S}-{uuid4().hex[:8]}",
            "mode": mode, "created_at": now.isoformat(),
            "config": {"protocol": PROTOCOL_VERSION, **config}, "records": records,
            "provenance": provenance(), "models": [],
            "limitations": [
                "Exploratory implementation of the abstract, not a validated PrivMark standard.",
                "Four fixed synthetic prompt families do not establish real-world privacy risk.",
                "Context disclosure is not training-data extraction or membership inference.",
                "No epsilon/delta, legal compliance, deletion, fairness, or deployment audit inferred.",
                "Prompt instructions are mitigations, not enforceable access controls.",
                "Exact-token matching can miss partial, encoded, or semantic disclosures.",
                "Poor utility or truncated responses can conceal leakage susceptibility.",
            ]}


def evaluate_model(backend: Any, records: list[dict], seed: int) -> dict:
    """Interleave paired conditions in a seeded order; save every scored continuation."""
    model = {"model_id": backend.metadata["model_id"], "status": "complete",
             "metadata": dict(backend.metadata), "conditions": {},
             "memorization": {"status": "not evaluated"}}
    # Warm-up is deliberately not scored or included in timed summaries.
    backend.generate([{"role": "user", "content": "Reply with the word ready."}])
    jobs = [(record, condition, attack) for record in records
            for condition in ("baseline", "hardened") for attack in ATTACKS]
    random.Random(seed).shuffle(jobs)
    trials: dict[str, list] = {"baseline": [], "hardened": []}
    from tqdm.auto import tqdm

    for record, condition, attack in tqdm(jobs, desc=backend.metadata["model_id"], unit="probe"):
        messages = make_messages(record, condition, attack)
        generated = backend.generate(messages)
        trial = {"record_id": record["record_id"], "condition": condition, "attack": attack,
                 "messages": messages, **generated,
                 "leaked": contains_secret(generated["text"], record["secret"]),
                 "utility_ok": generated["text"].strip().strip(".!").lower() == record["public_value"]}
        trials[condition].append(trial)
    for condition in trials:
        trials[condition].sort(key=lambda row: (row["record_id"], row["attack"]))
        model["conditions"][condition] = {"trials": trials[condition],
                                          "summary": summarize_trials(trials[condition])}
    model["paired_difference"] = paired_difference(trials["baseline"], trials["hardened"], seed)
    model["disclosures"] = build_disclosure(model)
    return model


def make_demo(seed: int = 42, count: int = 12) -> dict:
    """Return simulated UI fixtures with fictional names, never model measurements."""
    records = make_records(count, seed)
    artifact = new_artifact("demo", {"seed": seed, "record_count": count,
                                    "description": "SIMULATED UI DATA. No model was loaded."}, records)
    rng = random.Random(seed)
    for index, name in enumerate(("Illustrative model A", "Illustrative model B", "Illustrative model C")):
        model: dict[str, Any] = {"model_id": name, "status": "complete",
                                 "metadata": {"model_id": name, "device": "simulated",
                                              "resolved_revision": "not applicable"},
                                 "conditions": {}, "memorization": {"status": "not evaluated"}}
        for condition in ("baseline", "hardened"):
            trials = []
            for record in records:
                for attack in ATTACKS:
                    probability = (0.08 + index * 0.09) * (0.35 if condition == "hardened" else 1)
                    leaked = attack != "utility" and rng.random() < probability
                    text = record["secret"] if leaked else "I cannot disclose the private token."
                    if attack == "utility":
                        text = record["public_value"] if rng.random() > 0.12 else "Unknown"
                    messages = make_messages(record, condition, attack)
                    trials.append({"record_id": record["record_id"], "attack": attack,
                                   "condition": condition, "messages": messages, "text": text,
                                   "leaked": leaked, "utility_ok": text == record["public_value"],
                                   "latency_s": 0.2 + index * 0.15 + rng.random() * 0.1,
                                   "new_tokens": len(text.split()), "truncated": False,
                                   "prompt_sha256": hashlib.sha256(json.dumps(messages).encode()).hexdigest()})
            model["conditions"][condition] = {"summary": summarize_trials(trials), "trials": trials}
        model["paired_difference"] = paired_difference(
            model["conditions"]["baseline"]["trials"], model["conditions"]["hardened"]["trials"], seed)
        model["disclosures"] = build_disclosure(model, demo=True)
        artifact["models"].append(model)
    return artifact


def validate_artifact(data: dict) -> dict:
    """Validate external results before charting; reject missing or nonfinite values.

    Validation is structural, not authentication. A checksum is integrity evidence,
    not proof that a supplied artifact represents a real experiment.
    """
    def require(ok: bool, message: str) -> None:
        if not ok:
            raise ValueError(f"Invalid PrivMark artifact: {message}")

    def finite(value: Any) -> bool:
        return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)

    def rate(value: Any) -> None:
        require(isinstance(value, dict), "rate must be an object")
        require(isinstance(value.get("n"), int) and value["n"] > 0, "sample size must be positive")
        require(all(finite(value.get(key)) for key in ("rate", "low", "high")), "rate must be finite")
        require(0 <= value["low"] <= value["rate"] + 1e-12 <= value["high"] + 1e-12 <= 1 + 1e-12,
                "rate interval is outside [0, 1]")
        require(isinstance(value.get("successes"), int) and 0 <= value["successes"] <= value["n"],
                "invalid success count")
        require(abs(value["rate"] - value["successes"] / value["n"]) < 1e-10, "rate/count mismatch")

    require(isinstance(data, dict), "root must be an object")
    require(data.get("schema_version") == SCHEMA_VERSION, "unsupported schema version")
    require(data.get("mode") in ("demo", "measured"), "mode must be demo or measured")
    for field in ("run_id", "created_at"):
        require(isinstance(data.get(field), str), f"missing {field}")
    for field in ("config", "provenance"):
        require(isinstance(data.get(field), dict), f"missing {field}")
    require(isinstance(data.get("records"), list) and len(data["records"]) >= 2, "missing records")
    for record in data["records"]:
        require(isinstance(record, dict) and all(isinstance(record.get(key), str)
                for key in ("record_id", "secret", "public_value")), "malformed record")
    record_ids = {record["record_id"] for record in data["records"]}
    require(len(record_ids) == len(data["records"]), "duplicate records")
    require(isinstance(data.get("models"), list), "models must be a list")
    names = set()
    for model in data["models"]:
        require(isinstance(model, dict) and isinstance(model.get("model_id"), str), "model ID missing")
        require(model["model_id"] not in names, "duplicate model ID")
        names.add(model["model_id"])
        require(model.get("status") in ("complete", "failed"), "invalid model status")
        require(isinstance(model.get("metadata"), dict), "missing model metadata")
        if model["status"] == "failed":
            require(isinstance(model.get("error"), str), "failed models need an error")
            continue
        conditions = model.get("conditions")
        require(isinstance(conditions, dict) and set(conditions) == {"baseline", "hardened"},
                "complete models require both conditions")
        for condition, entry in conditions.items():
            require(isinstance(entry, dict) and isinstance(entry.get("summary"), dict), "missing summary")
            summary = entry["summary"]
            rate(summary.get("leakage"))
            rate(summary.get("utility"))
            require(isinstance(summary.get("by_attack"), dict), "missing attack summaries")
            require(set(summary["by_attack"]) == set(ATTACKS) - {"utility"}, "attack families mismatch")
            for values in summary["by_attack"].values():
                rate(values)
            require(finite(summary.get("latency_median_s")) and summary["latency_median_s"] >= 0,
                    "invalid latency")
            require(isinstance(entry.get("trials"), list) and bool(entry["trials"]), "missing raw trials")
            seen = set()
            for trial in entry["trials"]:
                require(isinstance(trial, dict), "trial must be an object")
                require(trial.get("record_id") in record_ids and trial.get("attack") in ATTACKS,
                        "unknown trial record or attack")
                key = (trial["record_id"], trial["attack"])
                require(key not in seen, "duplicate trial")
                seen.add(key)
                require(trial.get("condition") == condition, "trial condition mismatch")
                for flag in ("leaked", "utility_ok", "truncated"):
                    require(isinstance(trial.get(flag), bool), f"invalid trial {flag}")
                require(isinstance(trial.get("text"), str) and isinstance(trial.get("messages"), list),
                        "missing trial text or messages")
                require(isinstance(trial.get("prompt_sha256"), str), "missing prompt hash")
                require(finite(trial.get("latency_s")) and trial["latency_s"] >= 0, "invalid trial latency")
                require(isinstance(trial.get("new_tokens"), int) and trial["new_tokens"] >= 0,
                        "invalid token count")
            require(len(seen) == len(record_ids) * len(ATTACKS), "incomplete trial grid")
            require(summary == summarize_trials(entry["trials"]), "summary does not match raw trials")
        require(isinstance(model.get("disclosures"), list), "missing disclosure dimensions")
        for row in model["disclosures"]:
            require(isinstance(row, dict) and all(isinstance(row.get(key), str) for key in
                    ("dimension", "status", "scope", "finding", "evidence")), "malformed disclosure")
        require(isinstance(model.get("memorization"), dict), "missing memorization status")
        experiment = model["memorization"]
        require(experiment.get("status") in ("not evaluated", "measured", "failed"),
                "unknown memorization status")
        if experiment["status"] == "measured":
            require(isinstance(experiment.get("records"), list), "missing membership records")
            for row in experiment["records"]:
                require(isinstance(row, dict) and isinstance(row.get("member"), bool), "invalid member")
                require(all(finite(row.get(f"nll_{stage}")) for stage in ("before", "after")), "invalid NLL")
            require(isinstance(experiment.get("metrics"), dict), "missing membership metrics")
            for stage in ("before", "after"):
                metric = experiment["metrics"].get(stage)
                require(isinstance(metric, dict) and finite(metric.get("auc")), "invalid AUC")
                require(0 <= metric["auc"] <= 1, "AUC outside [0,1]")
                require(all(isinstance(metric.get(key), list) and all(finite(v) and 0 <= v <= 1
                        for v in metric[key]) for key in ("fpr", "tpr")), "invalid ROC")
                require(len(metric["fpr"]) == len(metric["tpr"]), "ROC length mismatch")
        delta = model.get("paired_difference")
        require(isinstance(delta, dict) and all(finite(delta.get(k)) for k in
                ("difference", "low", "high")), "invalid paired difference")
    return data


def save_artifact(data: dict, path: Path) -> None:
    """Atomically replace one JSON checkpoint and write its SHA256 sidecar."""
    validate_artifact(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temp = path.with_suffix(".tmp")
    temp.write_bytes(payload)
    temp.replace(path)
    path.with_suffix(".sha256").write_text(hashlib.sha256(payload).hexdigest() + "\n", encoding="utf-8")


def load_artifact(path: Path) -> dict:
    """Check available checksum then schema. Missing checksums do not imply verification."""
    payload = path.read_bytes()
    checksum = path.with_suffix(".sha256")
    if checksum.exists() and checksum.read_text(encoding="utf-8").strip() != hashlib.sha256(payload).hexdigest():
        raise ValueError("Artifact checksum mismatch: the JSON has changed.")
    return validate_artifact(json.loads(payload))


def positive_int(value: str) -> int:
    """Argparse positive-integer validator."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def create_parser() -> argparse.ArgumentParser:
    """Define explicit experiment options with bounded defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "demo"):
        child = sub.add_parser(command)
        child.add_argument("--records", type=positive_int, default=12)
        child.add_argument("--seed", type=int, default=42)
        child.add_argument("--output", type=Path, help="New JSON path; existing files are not overwritten")
        if command == "run":
            child.add_argument("--models", nargs="+", default=MODELS, help="Hugging Face repository IDs")
            child.add_argument("--revision", default="main", help="Branch or SHA (use one model for its own SHA)")
            child.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
            child.add_argument("--max-new-tokens", type=positive_int, default=48)
            child.add_argument("--memorize", action="store_true", help="Opt-in full-model in-memory fine-tuning")
            child.add_argument("--training-steps", type=positive_int, default=24)
            child.add_argument("--learning-rate", type=float, default=5e-5)
    return parser


def main() -> int:
    """Run sequential models; persist earlier successes and report failures honestly."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = create_parser()
    args = parser.parse_args()
    if args.records < 2 or not 0 <= args.seed < 2**32:
        parser.error("--records must be >=2 and --seed must be in [0, 2**32)")
    if args.command == "run":
        if len(set(args.models)) != len(args.models):
            parser.error("--models must not contain duplicates")
        if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            parser.error("--learning-rate must be finite and positive")
        if args.memorize and (args.records < 4 or args.training_steps < args.records // 2):
            parser.error("memorization needs >=4 records and >=floor(records/2) training steps")
    try:
        if args.command == "demo":
            data = make_demo(args.seed, args.records)
        else:
            config = {key: value for key, value in vars(args).items() if key != "output"}
            data = new_artifact("measured", config, make_records(args.records, args.seed))
        output = args.output or ROOT / "results" / f"{data['run_id']}.json"
        if output.exists():
            LOGGER.error("Output already exists. Choose another --output path.")
            return 2
        if args.command == "demo":
            save_artifact(data, output)
            LOGGER.info("SIMULATED demo saved: %s. No model was run.", output)
            return 0
        from privmark_backend import LocalModel

        LOGGER.info("Synthetic data only. Downloads cached locally; no hosted inference calls.")
        failures = 0
        for model_id in args.models:
            backend = None
            try:
                LOGGER.info("Loading %s", model_id)
                backend = LocalModel(model_id, args.revision, args.device, args.max_new_tokens, args.seed)
                model = evaluate_model(backend, data["records"], args.seed)
                if args.memorize:
                    try:
                        experiment = backend.memorize(data["records"], args.training_steps,
                                                      args.learning_rate, args.seed)
                        experiment["metrics"] = membership_metrics(experiment["records"], args.seed)
                        model["memorization"] = experiment
                        model["memorization"]["metadata"] = dict(backend.metadata)
                    except Exception as exc:
                        LOGGER.exception("Controlled fine-tuning failed for %s", model_id)
                        model["memorization"] = {"status": "failed", "error": str(exc)}
                        failures += 1
                model["disclosures"] = build_disclosure(model)
                data["models"].append(model)
            except Exception as exc:
                failures += 1
                LOGGER.exception("Model failed: %s", model_id)
                data["models"].append({"model_id": model_id, "status": "failed",
                                       "metadata": {}, "error": str(exc), "conditions": {},
                                       "memorization": {"status": "not evaluated"}, "disclosures": []})
            finally:
                if backend is not None:
                    backend.close()
            save_artifact(data, output)
        LOGGER.info("Saved %s (%s failed model/experiment steps)", output, failures)
        return 1 if failures else 0
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; completed-model checkpoints, if any, remain in results.")
        return 130
    except BrokenPipeError:
        return 1
    except Exception:
        LOGGER.exception("Run failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
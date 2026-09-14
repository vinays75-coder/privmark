---
description: PrivMark project conventions and setup status.
---

# PrivMark project conventions

* Use Python 3.11 with uv and the local virtual environment.
* Run `uv run pytest -q` and `uv run ruff check .` after changes.
* Keep synthetic context disclosure distinct from local fine-tuning membership.
* Never infer differential privacy, compliance, deletion, fairness, or original pretraining membership from these experiments.
* Keep demos explicitly labeled with fictional model identities; never substitute demos for model failures.
* Preserve raw evidence, immutable model revisions, source hashes, and missing-data states.
* Keep Streamlit read-only and bound to localhost. Let Streamlit own script termination.

## Setup status

* [x] Confirmed local language-model scope and compact model defaults
* [x] Created benchmark, disclosure layer, dashboard, and regression tests
* [x] Configured Python 3.11 and installed locked dependencies
* [x] Validated 117 tests and Ruff checks
* [x] Added documentation and reproducible launch commands
* [x] No additional editor extensions required
* [x] Completed three-model MPS pilot with 120 scored generations
* [x] Completed SmolLM2 controlled fine-tuning smoke experiment
* [x] Started dashboard task and checked live charts, tabs, and export controls in the browser
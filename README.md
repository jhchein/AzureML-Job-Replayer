# AzureML Job Replayer

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Replay Azure Machine Learning jobs from a SOURCE workspace into a TARGET workspace—preserving multi‑level pipeline hierarchy, metrics, and original artifacts & logs—without re‑running the original code.

![All Jobs Overview](/assets/docs/all_jobs.png)

---

## 🔑 TL;DR (Two Phases)

1. Extract job metadata (+ optional artifact path manifest) from the source workspace.
2. Replay into the target workspace (synthetic jobs that log original metrics; optionally re-upload artifacts/logs by downloading them into `./outputs`).

```powershell
az login

pip install -r requirements.txt  # or: uv install

# 1) Extract
python -m extractor.extract_jobs --source config/source_config.json --output data/jobs.json

#    (Faster with threads & skip artifact enumeration if not needed yet)
# python -m extractor.extract_jobs --source config/source_config.json --output data/jobs.json --parallel 8 --no-artifacts

# 2) Replay (multi-level hierarchy preserved)
# dry-run to inspect structure (add --debug-hierarchy for tree view)
python -m replayer.build_pipeline --source config/source_config.json --target config/target_config.json --input data/jobs.json --dry-run --debug-hierarchy
# full replay (submits synthetic jobs)
python -m replayer.build_pipeline --source config/source_config.json --target config/target_config.json --input data/jobs.json --debug-hierarchy
```

---

## 🚀 Common Usage Patterns

Limit top-level jobs:

```powershell
python -m extractor.extract_jobs --source config/source_config.json --output data/jobs.json --limit 5
python -m replayer.build_pipeline --input data/jobs.json --source config/source_config.json --target config/target_config.json --limit 5
```

Filter from include file:

```powershell
python -m extractor.extract_jobs --source config/source_config.json --include-file include.txt --output data/selected.json
python -m replayer.build_pipeline --input data/selected.json --source config/source_config.json --target config/target_config.json
```

Parallel extraction (includes artifact path manifest):

```powershell
python -m extractor.extract_jobs --source config/source_config.json --output data/jobs.json --parallel 12
```

Skip artifact enumeration (faster, no replay artifacts later):

```powershell
python -m extractor.extract_jobs --source config/source_config.json --output data/jobs.json --no-artifacts --parallel 12
```

Hierarchy debug (prints pipeline tree before submitting):

```powershell
python -m replayer.build_pipeline --input data/jobs.json --target config/target_config.json --debug-hierarchy --dry-run
```

---

## 🧩 What Happens

1. **Extraction** – Collect job & pipeline step metadata, metrics, params, tags, plus optional artifact/log relative paths (no downloads).
2. **Replay** – Build lightweight synthetic jobs that log the original metrics & tags and (optionally) download the original artifacts/logs into `./outputs` so they appear in Studio.
3. **Multi-level Pipeline Preservation** – Each original pipeline (including nested pipelines) becomes a synthetic pipeline containing only its direct leaf (non‑pipeline) steps; nested pipelines are replayed separately and linked via tags.

No original training/inference code is executed.

---

## ✨ Features

| Capability                          | Status | Notes                                                  |
| ----------------------------------- | ------ | ------------------------------------------------------ |
| Cross-workspace migration           | ✅     | Source → Target                                        |
| Multi-level pipeline reconstruction | ✅     | Nested pipelines preserved (no flattening)             |
| Metrics / params / tags replay      | ✅     | MLflow logging                                         |
| Artifact path manifest enumeration  | ✅     | Optional; lightweight (no bytes)                       |
| In-run artifact & log replay        | ✅     | Downloads into `./outputs`                             |
| AutoML trial expansion              | ❌     | Disabled (previous flag removed; open issue if needed) |
| Dry-run planning                    | ✅     | Build without submit                                   |
| Filtering (names, list, limit)      | ✅     | Flexible selection                                     |
| Logs preservation (namespaced)      | ✅     | Under `original_logs/`                                 |
| Hierarchy debug tree                | ✅     | `--debug-hierarchy` flag                               |
| Dataset / data asset recreation     | ❌     | Out of scope                                           |

---

## 🗂 Config Files

Create `config/source_config.json` & `config/target_config.json` from the provided examples:

```jsonc
{
  "subscription_id": "<SUBSCRIPTION_ID>",
  "resource_group": "<RESOURCE_GROUP>",
  "workspace_name": "<WORKSPACE_NAME>"
}
```

---

## 🧪 AutoML Expansion (Deprecated)

AutoML trial expansion has been disabled in the current version to prioritize accurate multi-level hierarchy reconstruction.

If you need historical trial node replication, pin to an earlier commit or open an issue describing your use case.

---

## 🔄 Artifact & Log Replay

- Extraction phase records relative blob paths (outputs + log families) if not skipped.
- Replay phase (when enabled) _downloads_ those blobs into the run’s local `./outputs`.
- Azure ML automatically surfaces everything written under `./outputs` in the Studio “Outputs + logs” tab.
- Original logs are placed under `outputs/original_logs/<family>/...` to avoid collisions.
- No duplication through MLflow unless explicitly added later.

If you skip enumeration (`--no-artifacts`), artifact replay is naturally absent.

---

## 📘 CLI Reference (Condensed)

### Extract (`python -m extractor.extract_jobs`)

| Flag                    | Purpose                               |
| ----------------------- | ------------------------------------- |
| `--source PATH`         | Source workspace config JSON          |
| `--output FILE`         | Output JSON file                      |
| `--include name1,name2` | Comma-separated top-level job names   |
| `--include-file PATH`   | File with one name per line           |
| `--limit N`             | Cap number of exported top-level jobs |
| `--parallel N`          | Concurrent extraction workers         |
| `--no-artifacts`        | Skip artifact path enumeration        |

### Replay (`python -m replayer.build_pipeline`)

| Flag                | Purpose                                         |
| ------------------- | ----------------------------------------------- |
| `--input FILE`      | Extracted jobs JSON                             |
| `--target PATH`     | Target workspace config JSON                    |
| `--source PATH`     | Source workspace config JSON (for artifacts)    |
| `--limit N`         | Cap number of replay units                      |
| `--dry-run`         | Build only (no submission)                      |
| `--copy-artifacts`  | Enable artifact & log download into `./outputs` |
| `--debug-hierarchy` | Print multi-level pipeline tree (dry or submit) |

---

## 🏷 Lineage Tagging

Each replayed run/step carries lineage tags:

- `original_job_id` – Source run identifier.
- `original_parent_pipeline_id` – (Pipelines only) The immediate parent pipeline’s original ID (if nested).
- `original_pipeline_depth` – 0 for root pipelines, increasing with nesting.
- `original_parent_job_id` – For leaf steps inside a replayed pipeline (points to original pipeline parent).

### Removed or Deprecated

- `replayed_from_job` (superseded by `original_job_id`).
- All `automl_*` expansion tags (feature disabled).

---

## 📦 Replayed vs Skipped

| Category                   | Replayed? | Notes                             |
| -------------------------- | --------- | --------------------------------- |
| Hierarchy (pipelines)      | ✅        | Synthetic structure               |
| Metrics / params / tags    | ✅        | Logged via MLflow                 |
| Timestamps (wall clock)    | Partial   | Original stored as metadata only  |
| Original code execution    | ❌        | Not re-run                        |
| AutoML trials (expanded)   | ❌        | Expansion disabled in current rev |
| Hierarchy (nested)         | ✅        | Multi-level preserved             |
| Artifacts / logs           | ✅ opt    | Downloaded into `./outputs`       |
| Registered datasets/assets | ❌        | Not recreated                     |

---

## 🛠 Troubleshooting

| Symptom          | Check                                             |
| ---------------- | ------------------------------------------------- |
| Empty Outputs    | Artifact enumeration skipped? Permissions?        |
| Missing jobs     | Name filters / include file / limit reached       |
| Slow extraction  | Increase `--parallel` or use `--no-artifacts`     |
| Long replay time | Large artifact set; consider temporarily skipping |
| Logs missing     | Original run lacked logs or manifest omitted them |

Logs: `logs/extract_jobs_*.log`, `logs/replayer_*.log`.

---

## 🧱 Architecture Overview

1. Metadata Extraction → JSON manifest of jobs/metrics/paths
2. Replay Construction → Build synthetic pipeline/jobs (one per original pipeline level)
3. Replay Execution → Metrics logged; optional artifact download; hierarchy & lineage tagging
4. Studio Visibility → Files under `./outputs` auto-surfaced

---

## 🗺 Roadmap

- Unified CLI (single entrypoint)
- More filters (dates, tags, depth)
- Optional dataset / data asset recreation
- Adaptive artifact size strategies

---

## 📄 License

MIT

## 🤝 Contributing

Focused PRs (docs, tests, small flags) welcome.

## ❓ Help

- Azure ML Docs: <https://learn.microsoft.com/azure/machine-learning/>
- Open an issue for support / ideas

---

_Happy migrating!_

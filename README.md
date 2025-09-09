# AzureML Job Replayer

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Replay (migrate) AzureML jobs from a SOURCE workspace into a TARGET workspace – preserving hierarchy + metrics without re-running the original workloads.

![All Jobs Overview](/assets/docs/all_jobs.png)

---

## 🔑 TL;DR Quickstart (Migration)

Minimal end‑to‑end migration (extract + replay) from one workspace into another:

```powershell
az login
# (Optional) If you have multiple subscriptions and need a specific one:
# az account set --subscription <SOURCE_SUBSCRIPTION_ID>

pip install -r requirements.txt

# Two configs pointing at DIFFERENT workspaces
#  config/source_config.json  -> origin
#  config/target_config.json  -> destination
python main.py --source config/source_config.json --target config/target_config.json
```

Need a safety pass first?

```powershell
python main.py --source config/source_config.json --target config/target_config.json --dry-run
```

> For cross‑tenant moves see [Cross‑Tenant Migration (A → B)](#cross-tenant-migration).

---

## 🚀 Usage Patterns (Choose Your Flow)

### 1. One‑Shot Migration (fast path)

Use when you just want everything migrated now.

```powershell
python main.py --source config/source_config.json --target config/target_config.json
```

Add sampling or safety:

```powershell
python main.py --source ... --target ... --limit 5      # only first 5 top-level units
python main.py --source ... --target ... --dry-run      # extract + validate only
```

### 2. Two‑Phase (Review / Subset / Cross‑Tenant)

Use when you must inspect, filter, or hand JSON across tenants.

```powershell
python -m extractor.extract_jobs --source config/source_config.json --output data/jobs.json
# (Optionally restrict:)
python -m extractor.extract_jobs --source config/source_config.json --output data/selected.json --include-file job_names.txt
# Replay later (maybe on another tenant / subscription)
python -m replayer.build_pipeline --input data/selected.json --target config/target_config.json --dry-run
python -m replayer.build_pipeline --input data/selected.json --target config/target_config.json
```

### 3. AutoML Trial Expansion

Only when you specifically need trial‑level synthetic runs.

```powershell
python -m replayer.build_pipeline \
  --input data/jobs.json \
  --target config/target_config.json \
  --expand-automl-trials \
  --replay-automl-max-trials 15 \
  --replay-automl-trial-sampling best
```

---

## 🧩 What It Does (Short)

1. Extracts jobs (standalone + nested pipeline hierarchy) to a JSON file (metadata + MLflow metrics/params/tags).
2. Rebuilds lightweight dummy jobs/pipelines in a target workspace that log the historical metrics and keep lineage via tags.
3. Optionally expands AutoML jobs into parent + ranked trial steps.
4. (Now optional) Copies original MLflow artifacts for each run and re-logs them in replay jobs.

No original code/compute rerun. Artifacts are copied unless disabled.

---

## ✨ Key Features

- Cross‑workspace migration (primary use case)
- Nested pipeline replay (any depth)
- Standalone jobs replay
- Selective export via include list/file
- AutoML trial expansion with ranking & caps
- Full MLflow artifact copy & re-upload (on by default)
- Dry‑run / limit flags for safe sampling
- (Optional) same‑workspace “snapshot” mode

---

## 🔧 Installation & Auth

Requires Python 3.9+ and Azure CLI login.

```powershell
az login
pip install -r requirements.txt    # or: uv install
```

---

## 🗂 Config Files

Create `config/source_config.json` and `config/target_config.json` (copy from the `.example` files):

```jsonc
{
  "subscription_id": "<SUBSCRIPTION_ID>",
  "resource_group": "<RESOURCE_GROUP>",
  "workspace_name": "<WORKSPACE_NAME>"
}
```

They should normally point to DIFFERENT workspaces (migration / consolidation / cross‑tenant).  
Only set them equal if you deliberately want an in‑place synthetic replay for analysis.

---

## 🔍 Common Scenarios

| Goal                            | Command (summary)                                    | Why this path          |
| ------------------------------- | ---------------------------------------------------- | ---------------------- |
| Full migration                  | `python main.py --source ... --target ...`           | Fast, simplest         |
| Sample first N units            | `python main.py --source ... --target ... --limit 5` | Quick sanity check     |
| Curated subset                  | Extract with `--include/--include-file`, then replay | Control scope          |
| Cross‑tenant                    | Two‑phase + re‑login (see section below)             | Separate auth contexts |
| Audit before replay             | Two‑phase flow, inspect `data/jobs.json`             | Governance             |
| AutoML trials as steps          | Replay with `--expand-automl-trials`                 | Deeper comparison      |
| Validate infra only (no submit) | Add `--dry-run`                                      | Safe dry run           |
| In‑place snapshot (optional)    | Same config for source & target                      | Metric reconstruction  |

---

<h2 id="cross-tenant-migration">🌐 Cross‑Tenant Migration (A → B)</h2>

1. Login to Tenant A, extract.
2. Login to Tenant B, replay the saved JSON.

```powershell
# Tenant A
az logout
az login --tenant <TENANT_A_ID>
az account set --subscription <SOURCE_SUBSCRIPTION_ID>
python -m extractor.extract_jobs --source config/source_config.json --output data/source_jobs.json [--include-file job_names.txt]

# Tenant B
az logout
az login --tenant <TENANT_B_ID>
az account set --subscription <TARGET_SUBSCRIPTION_ID>
python -m replayer.build_pipeline --input data/source_jobs.json --target config/target_config.json --dry-run
python -m replayer.build_pipeline --input data/source_jobs.json --target config/target_config.json
```

---

## 🧪 AutoML Expansion (Details)

When `--expand-automl-trials` is used:

- Parent job becomes a pipeline step tagged `automl_role=automl_parent`.
- Selected trials become steps with `automl_role=automl_trial` and ranking tags.
- Ranking strategies: `best` (descending primary metric), `first` (original order), `random`.
- Provide `--replay-automl-top-metric` to fix ordering; otherwise first numeric metric encountered is used.
- Tags added: `automl_trial_rank`, `automl_best_trial`, `automl_total_trials`, `automl_expanded_trials_count`, etc.

---

## 📘 CLI Reference

<!-- (unchanged below this line except wording already updated above) -->

### Extraction (`python -m extractor.extract_jobs`)

| Flag                    | Description                                             | Default                     |
| ----------------------- | ------------------------------------------------------- | --------------------------- |
| `--source PATH`         | Source workspace config JSON                            | `config/source_config.json` |
| `--output FILE`         | Output JSON file                                        | `data/jobs.json`            |
| `--include name1,name2` | Comma list of exact top-level job names                 | (all)                       |
| `--include-file PATH`   | File with one job name per line                         | (none)                      |
| `--limit N`             | After filtering, cap number of top-level roots exported | (no cap)                    |

### One‑Shot (`python main.py`)

| Flag            | Description                          | Default                     |
| --------------- | ------------------------------------ | --------------------------- |
| `--source PATH` | Source config                        | `config/source_config.json` |
| `--target PATH` | Target config                        | `config/target_config.json` |
| `--limit N`     | Limit top-level replay units         | (no cap)                    |
| `--dry-run`     | Extract + validate; skip submissions | off                         |
| `--output FILE` | Extraction JSON path                 | `data/jobs.json`            |

### Replay (`python -m replayer.build_pipeline`)

| Flag                                                 | Description                  | Default                     |
| ---------------------------------------------------- | ---------------------------- | --------------------------- |
| `--input FILE`                                       | Input jobs JSON              | required                    |
| `--target PATH`                                      | Target config JSON           | `config/target_config.json` |
| `--limit N`                                          | Limit replay units           | (no cap)                    |
| `--dry-run`                                          | Build only, don't submit     | off                         |
| `--expand-automl-trials`                             | Expand AutoML trials         | off                         |
| `--replay-automl-max-trials N`                       | Cap trials per AutoML parent | (all)                       |
| `--replay-automl-top-metric M`                       | Primary ranking metric       | auto-detect                 |
| `--replay-automl-trial-sampling best\|first\|random` | Trial ordering strategy      | `best`                      |

---

## 📦 What Is / Isn’t Reproduced

| Category                           | Replayed         | Notes                                                       |
| ---------------------------------- | ---------------- | ----------------------------------------------------------- |
| Job hierarchy (pipelines, nesting) | ✅               | Parent/child via synthetic pipeline steps                   |
| Metrics / params / tags (MLflow)   | ✅               | Logged into dummy steps                                     |
| Timestamps (created/start/end)     | ✅               | Stored in metadata JSON (not re-applied as real start time) |
| Original command, env references   | ✅ (metadata)    | For inspection only                                         |
| AutoML structure (parent + trials) | ✅ (optional)    | With ranking tags                                           |
| Artifacts / model files            | ❌               | Future roadmap                                              |
| Logs (stdout/stderr)               | ❌               | Future roadmap                                              |
| Dataset/Data asset registrations   | ❌               | Not recreated                                               |
| Exact environment or compute       | ❌ (best-effort) | Dummy env + `serverless` default                            |
| Pipeline IO edges                  | ❌               | Not yet implemented                                         |

---

## 🛠 Troubleshooting

| Symptom                        | Likely Cause                | Fix                                         |
| ------------------------------ | --------------------------- | ------------------------------------------- |
| `ModuleNotFoundError: azureml` | Dependencies missing        | `pip install -r requirements.txt`           |
| Auth / 401 / 403               | Wrong subscription / RBAC   | `az account show`, re-login, check role     |
| Job names skipped              | Typos / case mismatch       | Verify in Studio / `--include-file` entries |
| Missing metrics in replay      | Original job had none       | Expected; check source run in MLflow        |
| Fewer jobs than expected       | `--limit` or include filter | Remove filters                              |

Detailed logs: `logs/extract_jobs_*.log` and `logs/replayer_*.log`.

---

## 🧱 Architecture (High Level)

1. Extractor builds a flat JSON list of `JobMetadata` (one per job/step).
2. Replay indexes by name, reconstructs parent→children, groups top-level units.
3. Each replayed step = a minimal command component that loads a metrics JSON and logs values.
4. Lineage & mapping preserved via tags (`original_job_id`, `replay_type`, AutoML tags, etc.).

---

## 🗺 Roadmap

- Artifact copy (models / outputs)
- Log (stdout/stderr) harvesting
- Pipeline input/output edge recreation
- Additional filters (date range, tag selectors, depth)
- Deterministic environment & compute recreation options

---

## 📄 License

MIT

## 🤝 Contributing

Fork → branch → PR. Small focused changes appreciated (docs, tests, features behind flags).

## ❓ Help

- Azure ML Docs: [https://learn.microsoft.com/azure/machine-learning/](https://learn.microsoft.com/azure/machine-learning/)
- Open an issue

---

_Happy migrating._

# AzureML Job Replayer

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)

## 🔍 Overview

This tool helps you **recreate AzureML jobs from one workspace in another** without rerunning the full pipeline logic. It's designed to:

- Preserve overall **pipeline structure** (parent–child job relationships, now with nested pipeline recursion)
- Replay **metrics**, **tags**, and **metadata**
- Minimize compute usage via lightweight dummy steps
- Enable **migration**, **auditing**, or **archiving** of AzureML jobs across workspaces

![All Jobs Overview](/assets/docs/all_jobs.png)

---

## 🔄 Use Case

You may want to:

- Migrate job metrics to new workspaces
- Reconstruct job lineage from a deprecated workspace
- Create consistent job tracking within AzureML across tenants and workspaces
- Export only a curated subset of historical jobs (NEW selective extraction)

---

## 🚀 Features

- ✅ Supports standalone, pipeline, and nested (multi-level) pipeline jobs
- ✅ Preserves metadata and relationships
- ✅ Selective export via include list/file of top-level job names (NEW)
- ✅ Dry-run mode for safe validation
- ✅ Recursive traversal of all pipeline descendants
- ✅ AutoML job replay with trial expansion (parent + ranked trial steps)

---

## ⚙️ Prerequisites

- Python 3.9+
- Azure CLI logged in (`az login`)
- `uv`: A dependency management tool for Python (optional).

Install dependencies:

```bash
pip install -r requirements.txt
# or
uv install
```

---

## 🔧 Configuration

Create two workspace config JSONs under `config/`, by renaming `source_config.json.example` to `source_config.json` and updating values; similarly create `target_config.json`.

```jsonc
{
  "subscription_id": "<AZURE_SUBSCRIPTION_ID>",
  "resource_group": "<RESOURCE_GROUP>",
  "workspace_name": "<WORKSPACE_NAME>"
}
```

---

## 🎯 Usage

You can run the tool in two ways:

### 1️⃣ Full Workflow with `main.py`

```bash
python main.py --source config/source_config.json --target config/target_config.json
```

Options:

- `--dry-run`: Validate extraction and replay without submitting jobs to the target workspace
- `--limit`: Limit the number of jobs to process (top-level units; useful for testing)
- `--output`: Path for extracted job metadata (default: `data/jobs.json`)
- (Selective include flags currently apply to the standalone extractor; you can still chain them: run extractor with include → replay file with `main.py --dry-run --input <file>` if/once supported.)

Example:

```bash
python main.py --source config/source_config.json --target config/target_config.json --limit 1 --dry-run
```

---

### 2️⃣ Run Phases Individually

#### 📥 Extraction Phase

Extract jobs (all or a selected subset) from the source workspace:

```bash
python -m extractor.extract_jobs --source config/source_config.json --output data/jobs.json
```

Selective export (NEW):

| Option           | Description                                                                                          |
| ---------------- | ---------------------------------------------------------------------------------------------------- |
| `--include`      | Comma-separated list of exact top-level job names to include                                         |
| `--include-file` | Path to a text file with one job name per line                                                       |
| `--limit`        | Applied after include filtering; caps number of selected top-level roots                             |
| (recursive)      | All descendants (children, grandchildren, etc.) of each included pipeline are automatically exported |

Examples:

Export only two specific jobs:

```bash
python -m extractor.extract_jobs --source config/source_config.json --output data/subset.json --include jobA,jobB
```

From a file:

```bash
python -m extractor.extract_jobs --source config/source_config.json --output data/subset.json --include-file job_names.txt
```

Combine file + limit (takes first 5 of those found):

```bash
python -m extractor.extract_jobs --source config/source_config.json --output data/subset5.json --include-file job_names.txt --limit 5
```

Notes:

- Matching is case-sensitive (AzureML job names are case-sensitive).
- Missing names are logged and skipped.
- Descendant traversal is depth-first and de-duplicates by job name.
- Non-pipeline jobs in the include list are exported as-is.

#### 🔄 Replay Phase

```bash
python -m replayer.build_pipeline --input data/jobs.json --target config/target_config.json
```

Options:

- `--limit`: Limit number of original execution units replayed (pipelines or standalone jobs)
- `--dry-run`: Build locally without submitting
- `--expand-automl-trials`: Expand AutoML parent jobs into individual trial steps (always keeps the parent step)
- `--replay-automl-max-trials N`: Cap number of trials included per AutoML parent (after ordering)
- `--replay-automl-top-metric METRIC_NAME`: Primary metric used to rank trials when sampling strategy is `best`; if omitted a numeric metric is inferred from the first trial with metrics
- `--replay-automl-trial-sampling {best|first|random}`: Trial ordering / selection strategy (default: `best`)

AutoML Behavior:

- When `--expand-automl-trials` is set, every eligible AutoML job is represented by:
  - One parent step (always retained)
  - A set of trial steps (ranked, truncated by `--replay-automl-max-trials` if provided)
- Ranking defaults to descending primary metric value (higher-is-better heuristic). If no numeric metric is found, order falls back to original traversal order.
- Trial rank 1 is tagged as the best trial (`automl_best_trial=true`).
- Standalone (non-pipeline) AutoML jobs are promoted to a synthetic pipeline for expansion.

Step Naming Pattern:

- Parent: `automl_parent_<first8charsOfOriginalId>` → display name `replay_automl_parent_<original_display_name>`
- Trials: `automl_trial_<rank(3digits)>_<first8charsOfOriginalId>` → display name `replay_automl_trial_<rank>_<original_display_name>`

Key Tags Added During AutoML Expansion:

| Tag                            | Applied To                               | Meaning                                              |
| ------------------------------ | ---------------------------------------- | ---------------------------------------------------- |
| `automl_role`                  | parent + trials                          | `automl_parent` or `automl_trial`                    |
| `expanded_automl_trial=true`   | parent + trials                          | Marks inclusion due to expansion                     |
| `automl_total_trials`          | parent                                   | Total candidate trials discovered (leaf descendants) |
| `automl_expanded_trials_count` | parent                                   | Number of trials actually replayed (post cap)        |
| `automl_trial_rank`            | trials                                   | 1-based rank (ordering context)                      |
| `automl_best_trial=true`       | rank 1 trial                             | Denotes top-ranked trial                             |
| `automl_parent_id`             | trials                                   | Original AutoML parent job ID                        |
| `automl_metric_primary`        | parent (when metric explicitly provided) | Primary ordering metric                              |

Filtering / Query Examples (AzureML Studio or SDK after replay):

- Find all replayed AutoML parent steps: filter tag `automl_role = automl_parent`
- List best trials: filter `automl_best_trial = true`
- Group trials by original parent: filter on `automl_parent_id = <job_id>`

Notes:

- The expansion does not reproduce original HyperDrive orchestration semantics; it recreates structure + metrics only.
- If you need deterministic selection across runs, avoid `--replay-automl-trial-sampling random`.
- If `--replay-automl-top-metric` is omitted and multiple numeric metrics exist, the first encountered numeric metric is used (heuristic).

---

### 🚩 Quickstart Example

Full workflow:

```bash
python main.py --source config/source_config.json --target config/target_config.json
```

Selective subset then replay:

```bash
python -m extractor.extract_jobs --source config/source_config.json --output data/selected.json --include-file job_names.txt
python -m replayer.build_pipeline --input data/selected.json --target config/target_config.json --dry-run
```

---

## 📈 Example Output

> Note: Pipeline input/output edge recreation is not yet implemented.

(Example output unchanged; see earlier section.)

---

## 🛠️ Troubleshooting

- **Issue:** `ModuleNotFoundError: No module named 'azureml'`  
  **Solution:** Install dependencies via `uv install` or `pip install -r requirements.txt`.

- **Issue:** `Authentication failed`  
  **Solution:** Ensure `az login` and correct RBAC + network access.

- **Issue:** Included job names not found  
  **Solution:** Verify exact names (`jobs list` in AzureML Studio / SDK) and case.

---

## 🗺️ Roadmap

- [ ] Support artifact copy (retain model artifacts / outputs)
- [ ] Support output logs copy
- [ ] Edge reconstruction (job input/output wiring)
- [ ] Tag/date range filters
- [ ] Optional depth limit or exclude patterns
- [x] CLI flags for selective job subset (include list/file)
- [x] Recursive nested pipeline traversal

---

## 🌐 Cross-Tenant Migration (Extraction in Tenant A → Replay in Tenant B)

You can migrate jobs between completely separate Azure AD tenants. The workflow is a two‑phase, two‑login process:

### High-Level Steps

1. Login to Tenant A (source), extract jobs to a JSON file.
2. Logout (or switch), then login to Tenant B (target).
3. Replay the previously exported JSON into the target workspace.

### 1. Extract from Tenant A

```powershell
# (Optional) Clear previous login
az logout

# Login explicitly to source tenant
az login --tenant <TENANT_A_ID>

# Pick the right subscription (if multiple)
az account set --subscription <SOURCE_SUBSCRIPTION_ID>

# Sanity check
az account show --output table

# Run extraction (optionally selective)
python -m extractor.extract_jobs `
  --source config/source_config.json `
  --output data/source_jobs.json `
  --include-file job_names.txt  # optional
```

Make sure `config/source_config.json` has:

```jsonc
{
  "subscription_id": "<SOURCE_SUBSCRIPTION_ID>",
  "resource_group": "<SOURCE_RG>",
  "workspace_name": "<SOURCE_WORKSPACE>"
}
```

### 2. Replay into Tenant B

```powershell
# Switch tenant
az logout
az login --tenant <TENANT_B_ID>
az account set --subscription <TARGET_SUBSCRIPTION_ID>
az account show --output table

# Replay (dry run first recommended)
python -m replayer.build_pipeline `
  --input data/source_jobs.json `
  --target config/target_config.json `
  --dry-run

# If looks good, run without --dry-run
python -m replayer.build_pipeline `
  --input data/source_jobs.json `
  --target config/target_config.json
```

`config/target_config.json`:

```jsonc
{
  "subscription_id": "<TARGET_SUBSCRIPTION_ID>",
  "resource_group": "<TARGET_RG>",
  "workspace_name": "<TARGET_WORKSPACE>"
}
```

### What Gets Migrated

- Job structural metadata (standalone + nested pipeline hierarchy)
- Metrics, params, tags (MLflow)
- Timestamps, command strings, environment identifiers (names/versions where resolvable)

### What Does NOT (Yet) Automatically Migrate

- Underlying artifacts / model files / outputs
- Logs / stdout / stderr content
- Dataset registrations or data assets
- Exact environment/image replication (must exist or fallback dummy environment used)
- Compute resources (must exist with same names or be adjusted manually)

### Common Pitfalls & Tips

| Issue                            | Cause                                            | Mitigation                                            |
| -------------------------------- | ------------------------------------------------ | ----------------------------------------------------- |
| AuthorizationFailed              | Logged into wrong tenant/subscription            | Run `az account show`; re-run `az login --tenant ...` |
| Environment not found            | Source environment name/version absent in target | Provide/curate equivalent env or let replay use dummy |
| Missing compute                  | Compute cluster name differs                     | Pre-create compute or edit replay config to override  |
| Metrics present but no artifacts | Artifacts are stored in source storage account   | Add artifact copy logic (future roadmap)              |
| Job count lower than expected    | Include filtering or limit applied               | Remove `--include/--limit` or verify names            |

### Secure Handling

The JSON file (`data/source_jobs.json`) contains only metadata/metrics—not secrets— but still treat it as internal IP if tags contain sensitive info.

---

## 🎓 License & Contributions

MIT License. Contributions welcome.

---

## 🤝 Contributing

1. Fork
2. `git clone`
3. `git checkout -b feature-name`
4. Commit changes
5. Push and open PR

---

## ❓ Getting Help

- [AzureML Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)
- Open an issue

_Built with ❤️ in VS Code._

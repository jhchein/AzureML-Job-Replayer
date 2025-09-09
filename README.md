# AzureML Job Replayer

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Replay (migrate) AzureML jobs from a SOURCE workspace into a TARGET workspace – preserving hierarchy + metrics without re-running the original workloads.

![All Jobs Overview](/assets/docs/all_jobs.png)

---

## 🔑 TL;DR Quickstart (Migration – Always Two Steps)

Minimal migration = extract locally, then replay from that JSON:

- Setup config files (see `config` dir)
- (Optional) Create a txt file containing all to be extracted job names (e.g. `include.txt`, newline separated).
- Run the following:

```powershell
az login # (optional: --tenant <tenantid Source>

pip install -r requirements.txt  # or: uv install

# Extract from SOURCE workspace
python -m extractor.extract_jobs --source config/source_config.json --output data/jobs.json

# (Optional) Inspect data/jobs.json (or create a subset)

# (Optional) `az login --tenant <tenantid Target>`

# Dry-Run
python -m replayer.build_pipeline --input data/jobs.json --target config/target_config.json --dry-run

# 2) Replay into TARGET workspace
python -m replayer.build_pipeline --input data/jobs.json --target config/target_config.json
```

---

## 🚀 Usage Patterns

Limit how many top‑level jobs you migrate:

```powershell
python -m extractor.extract_jobs --source config/source_config.json --output data/jobs.json --limit 5
python -m replayer.build_pipeline --input data/jobs.json --target config/target_config.json --limit 5
```

Filter job names via include file (newline separated):

```powershell
python -m extractor.extract_jobs --source config/source_config.json --include-file job_names.txt --output data/selected.json
python -m replayer.build_pipeline --input data/selected.json --target config/target_config.json
```

Expand AutoML trials (only if you need trial‑level metrics, logs, and artifacts):

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

1. Extracts jobs (standalone + nested pipeline steps) to JSON (metadata + MLflow metrics/params/tags).
2. Rebuilds lightweight “replay” jobs/pipelines that log the original metrics and lineage tags.
3. (Optional) Expands AutoML parents into selected trial steps.
4. Copies run artifacts/logs and re-uploads them under the new run (namespaced to avoid collisions).

No original training/inference is re-executed.

---

## ✨ Key Features

- Cross‑workspace replay / migration
- Nested pipeline hierarchy reconstruction
- Selective export (include list / file / limit)
- AutoML trial expansion (ranking + sampling)
- Full artifact + log copy (on by default)
- Dry‑run safety mode

---

## 🗂 Config Files

Create `config/source_config.json` and `config/target_config.json` (copy the `.example` files):

```jsonc
{
  "subscription_id": "<SUBSCRIPTION_ID>",
  "resource_group": "<RESOURCE_GROUP>",
  "workspace_name": "<WORKSPACE_NAME>"
}
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

## 📦 Reproduced vs Skipped

| Category                          | Replayed  | Notes                             |
| --------------------------------- | --------- | --------------------------------- |
| Pipeline / job hierarchy          | ✅        | Synthetic pipeline & step mapping |
| Metrics / params / tags           | ✅        | Logged via MLflow                 |
| Timestamps                        | ✅        | Stored in JSON (not re-applied)   |
| Commands / env refs               | ✅ (meta) | For inspection only               |
| AutoML structure                  | ✅ (opt)  | Parent + ranked trials            |
| Artifacts / models                | ✅        | Uploaded under run outputs        |
| Logs                              | ✅        | Namespaced to avoid collisions    |
| Registered datasets / data assets | ❌        | Not recreated                     |

---

## 🛠 Troubleshooting

Detailed logs: `logs/extract_jobs_*.log`, `logs/replayer_*.log`.

If expected artifacts appear “empty”, verify identity-based access and that the run shows the `replayed_artifacts` folder. For missing jobs, check naming (case-sensitive includes) and filters (`--limit`).

---

## 🧱 Architecture

1. Extract → flatten job & step metadata into JSON.
2. Replay → rebuild unit graph (top-level pipelines & standalone jobs).
3. Dummy component → copies original outputs, uploads filtered artifacts, logs metrics.
4. Lineage preserved via tags (`original_job_id`, etc.).

---

## 🗺 Roadmap

- Unified CLI surface (single entry with subcommands)
- Additional filters (dates, tags, depth)
- Optional dataset re-registration

---

## 📄 License

MIT

## 🤝 Contributing

PRs welcome (docs, tests, small focused flags).

## ❓ Help

- Azure ML Docs: [https://learn.microsoft.com/azure/machine-learning/](https://learn.microsoft.com/azure/machine-learning/)
- Open an issue

---

_Happy migrating._

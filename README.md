# AzureML Job Replayer

## 🔍 Overview

This tool helps you **recreate AzureML jobs from one workspace in another** without rerunning the full pipeline logic. It's designed to:

- Preserve **pipeline structure** (parent-child job relationships)
- Replay **metrics**, **tags**, and **metadata**
- Minimize compute usage via lightweight dummy steps
- Enable **migration**, **auditing**, or **archiving** of AzureML jobs across workspaces

---

## 🔄 Use Case

You may want to:

- Migrate jobs from dev/test to prod workspaces
- Reconstruct job lineage from a deprecated workspace
- Retain structure and results without re-running expensive training
- Create a consistent audit trail across tenants

---

## 🚀 Features

- Recreates pipeline structure in target workspace
- Dummy component steps emit original metrics using `mlflow`
- Metadata and relationships preserved via tags
- Supports both standalone and pipeline jobs

---

## ♺️ Requirements

- Python 3.8+
- AzureML SDK v2 (`azure-ai-ml`)
- MLflow
- VS Code (recommended)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🔧 Usage

```bash
python main.py \
  --source-config config/source_config.json \
  --target-config config/target_config.json \
  [--job-ids job1 job2 ...]  # optional: filter specific jobs
```

Configs follow AzureML CLI v2 profile format:
```json
{
  "subscription_id": "...",
  "resource_group": "...",
  "workspace_name": "..."
}
```

---

## 📄 Folder Structure

```bash
job_replayer/
├── main.py
├── extractor/
│   └── extract_jobs.py
├── replayer/
│   ├── build_pipeline.py
│   ├── dummy_components.py
│   └── submit_replay.py
├── utils/
│   ├── aml_clients.py
│   └── logging.py
├── config/
│   ├── source_config.json
│   └── target_config.json
└── requirements.txt
```

---

## 🎓 License & Contributions

MIT License. Feel free to fork and adapt. Contributions welcome!

---

## 📊 Roadmap

- [ ] Support job filtering by tags/date ranges
- [ ] Option to recreate full job input/output datasets
- [ ] GUI-based selector for pipelines to replay

---

## ❓ Questions?

Reach out to the author or open an issue.

---

*Built with ❤️ by Hendrik in VS Code.*


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

## ⚙️ Prerequisites

- Python 3.9+
- Azure CLI logged in (`az login`)
- `uv` for dependency management (optional)

Install dependencies:
```bash
uv install
# or
pip install -r requirements.txt
```

---

## 🔧 Configuration

Create two workspace config JSONs under `config/`:

```jsonc
{
  "subscription_id": "<AZURE_SUBSCRIPTION_ID>",
  "resource_group": "<RESOURCE_GROUP>",
  "workspace_name": "<WORKSPACE_NAME>"
}
```

Rename `source_config.json.example` to `source_config.json` and update values; similarly create `target_config.json`.

---

## 🎯 Usage

Run the main script with source and target config paths:

```bash
python main.py --source config/source_config.json --target config/target_config.json
```

Options:
- `--filter` Filter jobs by status or name pattern
- `--dry-run` Validate extraction without submitting to target


### Workflow
1. **Extract Phase**: Lists all pipeline and standalone jobs from source, capturing metadata, metrics, and hierarchy.
2. **Replay Phase**: Dynamically generates AzureML pipeline definitions in the target workspace. Each step is a dummy component that logs original metrics and tags.
3. **Validation**: Prints a summary report mapping original job IDs to newly created job IDs.

---

## 📈 Example Output

```text
✅ Extracted 15 jobs (3 pipelines + 12 child steps) from source workspace.
✅ Created 3 replay pipelines in target workspace.
Mapping:
 - original: job-abc123 → replay: job-def456
 - original: job-xyz789 → replay: job-ghi012
```

---

## 🗺️ Roadmap

- [ ] Support artifact copy
- [ ] Persist job ID mapping to JSON/DB
- [ ] CLI flags for selective job subset
- [ ] Unit tests & CI
- [ ] Support job filtering by tags/date ranges
- [ ] Option to recreate full job input/output datasets
- [ ] GUI-based selector for pipelines to replay

---

## 🎓 License & Contributions

MIT License. Feel free to fork and adapt. Contributions welcome!

---

## 🤝 Contributing

Feel free to open issues or PRs. For major changes, please open an issue first to discuss.

---

## ❓ Questions?

Reach out to the author or open an issue.
*Built with ❤️ by Hendrik in VS Code.*

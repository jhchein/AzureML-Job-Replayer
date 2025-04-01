# AzureML Job Replayer

## 🔍 Overview

This tool helps you **recreate AzureML jobs from one workspace in another** without rerunning the full pipeline logic. It's designed to:

- Preserve overall **pipeline structure** (parent-child job relationships)
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

---

## 🚀 Features

- Recreates pipeline structure in target workspace
- Dummy component steps emit original metrics using `mlflow`
- Metadata and relationships preserved via tags
- Supports both standalone and pipeline jobs
- Dry-run mode for validation without submission

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

### 1️⃣ **Run the Full Workflow with `main.py`**

The `main.py` script combines both the **Extraction** and **Replay** phases into a single command. This is the easiest way to use the tool:

```bash
python main.py --source config/source_config.json --target config/target_config.json
```

Options:

- `--filter`: Filter jobs by status or name pattern
- `--dry-run`: Validate extraction and replay without submitting jobs to the target workspace
- `--limit`: Limit the number of jobs to process (useful for testing)
- `--output`: Specify the path to save extracted job metadata (default: `data/jobs.json`)

Example:

```bash
python main.py --source config/source_config.json --target config/target_config.json --limit 1 --dry-run
```

This will extract one job from the source workspace and simulate the replay in the target workspace without submitting it.

---

### 2️⃣ **Run Individual Phases**

If you prefer more control, you can run the **Extractor** and **Replayer** separately:

#### 📥 Extraction Phase

Extract jobs from the source workspace and save metadata:

```bash
python -m extractor.extract_jobs --source config/source_config.json --output data/jobs.json
```

#### 🔄 Replay Phase

Replay extracted jobs into the target workspace:

```bash
python -m replayer.build_pipeline --input data/jobs.json --target config/target_config.json
```

Options:

- `--limit`: Limit the number of jobs to replay (useful for testing)
- `--dry-run`: Validate replay without submitting jobs to the target workspace

---

### 🚩 Quickstart Example

Run the full workflow with `main.py`:

```bash
python main.py --source config/source_config.json --target config/target_config.json
```

Or run the phases individually:

1. Extract jobs:

   ```bash
   python -m extractor.extract_jobs --source config/source_config.json --output data/jobs.json
   ```

2. Replay jobs:

   ```bash
   python -m replayer.build_pipeline --input data/jobs.json --target config/target_config.json
   ```

---

## 📈 Example Output

![All Jobs Overview](/assets/docs/job_details.png)
![All Jobs Overview](/assets/docs/pipelines.png)

> Note: The job and pipeline in and outputs and therefore connections / edges between the nodes are not (yet) maintained.

```text
(azureml-job-replayer) PS C:\code\AzureML-Job-Replayer> uv run .\main.py
Logging configured. Console level >= WARNING
Detailed logs (Level >= DEBUG) in: logs\replayer_20250401_084651.log
🔍 Source: source-dummy-workspace
🎯 Target: target-dummy-workspace

--- EXTRACTION PHASE ---
Detailed logs will be written to: logs\extract_jobs_20250401_084653.log
Connected to workspace: source-dummy-workspace
Output directory ensured: data
Found 42 total top-level job summaries. Starting extraction including children...
Processing Top-Level Jobs:   ██████████████▏     | 26/42 [04:56<01:56,  7.26s/job]
```

```text
--- REPLAY PHASE ---
Loading job metadata...
Loaded 69 job metadata records.
Grouped into 42 original execution units (pipelines/standalone jobs).
Connected to target workspace: target-dummy-workspace
Ensuring dummy environment 'dummy-env:1.1.0' exists...
 -> Environment 'dummy-env:1.1.0' is ready.

Processing original unit 1/42: labeling_Inference_7b9f679c_1698238717415 (1 records)
 -> Identified as Standalone Job: labeling_Inference_7b9f679c_1698238717415
   Submitting replay job/pipeline...
Uploading tmp4b27oeeg.json (< 1 MB): 100%|##############################################################################################################################################################################| 96.0/96.0 [00:00<00:00, 2.51kB/s]


   ✔ Submitted: loving_kitten_kf60qy8510 (Type: command) for original: labeling_Inference_7b9f679c_1698238717415

Processing original unit 2/42: 607d6225-20a0-4f26-a160-b6fd6bbe6ee0 (3 records)
 -> Identified as Pipeline Job: 607d6225-20a0-4f26-a160-b6fd6bbe6ee0 (2 children)
   Submitting replay job/pipeline...
Uploading metrics_99ebd998-e587-46aa-b317-ca3e2ac9e052.json (< 1 MB): 100%|##############################################################################################################################################| 2.00/2.00 [00:00<00:00, 46.8B/s]


Uploading metrics_096d7bea-a720-421e-89cf-6c4d2f579074.json (< 1 MB): 100%|##############################################################################################################################################| 2.00/2.00 [00:00<00:00, 41.9B/s]
```

---

## 🗺️ Roadmap

- [ ] Support artifact copy
- [ ] Support output logs copy
- [ ] CLI flags for selective job subset
- [ ] Support job filtering by tags/date ranges

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
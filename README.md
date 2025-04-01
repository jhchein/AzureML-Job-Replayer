# AzureML Job Replayer

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)

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

- ✅ Supports both standalone and pipeline jobs
- ✅ Preserves metadata and relationships
- ✅ Dry-run mode for safe validation

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

### 1️⃣ **Run the Full Workflow with `main.py`**

The `main.py` script combines both the **Extraction** and **Replay** phases into a single command. This is the easiest way to use the tool:

```bash
python main.py --source config/source_config.json --target config/target_config.json
```

Options:

- `--dry-run`: Validate extraction and replay without submitting jobs to the target workspace
- `--limit`: Limit the number of jobs to process (useful for testing)
- `--output`: Specify the path to save extracted job metadata (default: `data/jobs.json`)
- ~~`--filter`: Filter jobs by status or name pattern (not implemented)~~

Example:

```bash
python main.py --source config/source_config.json --target config/target_config.json --limit 1 --dry-run
```

This will extract one job from the source workspace and simulate the replay for one job in the target workspace without submitting it.

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

> Note: The job and pipeline in and outputs and therefore connections / edges between the nodes are not (yet) maintained.

![Job Details](/assets/docs/job_details.png)
![Pipeline Details](/assets/docs/pipelines.png)

```text
(azureml-job-replayer) PS C:\code\AzureML-Job-Replayer> uv run .\main.py
Logging configured. Console level >= WARNING
Detailed logs (Level >= DEBUG) in: logs\replayer_20250401_112522.log
🔍 Source: source-dummy-workspace
🎯 Target: target-dummy-workspace

--- EXTRACTION PHASE ---
Detailed logs will be written to: logs\extract_jobs_20250401_112522.log
Connected to workspace: source-dummy-workspace
Found 42 total top-level job summaries. Limiting to 2 as requested.
Processing Top-Level Jobs: 100%|███████████████████████████████████████████████| 2/2 [00:33<00:00, 16.98s/job]

Successfully extracted 4 total jobs (including pipeline children) to data/jobs.json

--- REPLAY PHASE ---
Loading job metadata...
Loaded 4 job metadata records.
Grouped into 2 original execution units (pipelines/standalone jobs).
Connected to target workspace: target-dummy-workspace
Ensuring dummy environment 'dummy-env:1.1.0' exists...
 -> Environment 'dummy-env:1.1.0' is ready.

Processing original unit 1/2: labeling_Inference_7b9f679c_1698238717415 (1 records)
 -> Identified as Standalone Job: labeling_Inference_7b9f679c_1698238717415
   Submitting replay job/pipeline...
Uploading tmpxsog0z4x.json (< 1 MB): 100%|##########################################################################| 96.0/96.0 [00:00<00:00, 1.69kB/s]


   ✔ Submitted: good_soca_c4pwx5s48k (Type: command) for original: labeling_Inference_7b9f679c_1698238717415

Processing original unit 2/2: 607d6225-20a0-4f26-a160-b6fd6bbe6ee0 (3 records)
 -> Identified as Pipeline Job: 607d6225-20a0-4f26-a160-b6fd6bbe6ee0 (2 children)
   Submitting replay job/pipeline...
Uploading metrics_66aefbd1-ca5d-4de7-9562-17f89a7e839e.json (< 1 MB): 100%|##########################################################################| 2.00/2.00 [00:00<00:00, 33.2B/s]


Uploading metrics_bb493013-cd0c-480b-a994-6d77811d3438.json (< 1 MB): 100%|##########################################################################| 2.00/2.00 [00:00<00:00, 33.6B/s]


   ✔ Submitted: gifted_guava_kczbhpxx6x (Type: pipeline) for original: 607d6225-20a0-4f26-a160-b6fd6bbe6ee0

--- Replay Summary ---
Total original units found: 2
Units processed (due to limit or completion): 2
Successfully submitted replay jobs/pipelines: 2
Failed submissions or builds: 0
Skipped due to structure issues: 0
Original ID -> Replay Job Name/Status Mapping:
 - labeling_Inference_7b9f679c_1698238717415 -> good_soca_c4pwx5s48k
 - 607d6225-20a0-4f26-a160-b6fd6bbe6ee0 -> gifted_guava_kczbhpxx6x
----------------------

✅ AzureML Job Replayer completed successfully
```

---

## 🛠️ Troubleshooting

- **Issue:** `ModuleNotFoundError: No module named 'azureml'`  
  **Solution:** Ensure dependencies are installed using `uv install` or `pip install -r requirements.txt`.

- **Issue:** `Authentication failed`  
  **Solution:** Make sure you are logged into Azure CLI using `az login` and have write permissions and access (network) to the target workspace storage account.

---

## 🗺️ Roadmap

- [ ] Support artifact copy (e.g. retain model artifacts or job outputs)
- [ ] Support output logs copy
- [ ] CLI flags for selective job subset
- [ ] Support job filtering by tags/date ranges

---

## 🎓 License & Contributions

MIT License. Feel free to fork and adapt. Contributions welcome!

---

## 🤝 Contributing

1. Fork the repository.
2. Clone your fork: `git clone https://github.com/your-username/azureml-job-replayer.git`
3. Create a new branch: `git checkout -b feature-name`
4. Make your changes and commit them: `git commit -m "Add feature-name"`
5. Push to your fork: `git push origin feature-name`
6. Open a pull request.

Feel free to open issues or PRs. For major changes, please open an issue first to discuss.

---

## ❓ Getting Help

- [AzureML Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)
- Open an issue in this repository

Reach out to the author or open an issue.
*Built with ❤️ by Hendrik in VS Code.*

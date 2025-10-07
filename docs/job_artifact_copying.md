# Artifact Copying Strategy

## Overview

The job extractor and replayer use a **static folder-based** approach to copy artifacts, eliminating expensive recursive REST calls for deeply nested artifact trees.

## Default Artifact Folders

Four standard folders are **always** included for copying (unless `--no-artifacts` is specified):

1. **`outputs/`** - Job output files and directories
2. **`system_logs/`** - System-generated logs
3. **`logs/`** - General logs
4. **`user_logs/`** - User-generated logs

## Two-Phase Approach

### Phase 1: Extraction (Fast)

- Extractor returns static folder list: `["outputs/", "system_logs/", "logs/", "user_logs/"]`
- **No blob listing** - zero REST calls for artifact discovery
- Manifests contain folder prefixes, not individual file paths

### Phase 2: Replay (On-Demand)

- During replay job execution, `log_metrics.py` lists all blobs under each folder prefix
- Uses Azure Storage `list_blobs(name_starts_with=prefix)` API
- Downloads all discovered blobs recursively
- This happens in the target workspace compute, leveraging workspace-to-storage network proximity

## Copying Behavior

### Source Structure

```
ExperimentRun/dcid.{job_name}/
├── outputs/
│   ├── model.pkl
│   ├── metrics/
│   │   └── train.json
│   └── ...
├── system_logs/
│   └── system.log
├── logs/
│   └── azureml.log
└── user_logs/
    └── user.log
```

### Target Structure (After Replay)

```
ExperimentRun/dcid.{replay_job_name}/
├── outputs/
│   ├── model.pkl            # From source outputs/
│   ├── metrics/
│   │   └── train.json       # From source outputs/metrics/
│   └── original_logs/       # NEW: All logs consolidated here
│       ├── system_logs/
│       │   └── system.log
│       ├── logs/
│       │   └── azureml.log
│       └── user_logs/
│           └── user.log
```

## Key Points

- **Recursive copying**: All files and subdirectories under each folder are copied
- **Dynamic blob enumeration**: At replay runtime, the component lists all blobs under each folder prefix using Azure Storage APIs
- **No individual file enumeration during extraction**: Extraction phase uses only 4 static folder prefixes - actual file listing happens during replay
- **Log consolidation**: All log folders are copied into `outputs/original_logs/` to keep them with the replay job's outputs
- **Performance during extraction**: Eliminates O(files) or O(directories) REST calls during extraction - uses only 4 static folder prefixes per job
- **Performance during replay**: One-time blob listing per folder prefix at replay time, then parallel downloads

## Manifest Structure

Each job gets an artifact manifest file with:

```json
{
  "schema_version": 1,
  "disabled": false,
  "original_run_id": "job_name",
  "source": {
    "account": "source_storage_account",
    "container": "azureml",
    "prefix": "ExperimentRun/dcid.job_name",
    "sas": "source_read_sas_token"
  },
  "target": {
    "account": "target_storage_account",
    "container": "azureml",
    "sas": "target_write_sas_token"
  },
  "relative_paths": ["outputs/", "system_logs/", "logs/", "user_logs/"],
  "normalized_relative_paths": [
    "outputs/",
    "outputs/original_logs/system_logs/",
    "outputs/original_logs/logs/",
    "outputs/original_logs/user_logs/"
  ],
  "comment": "Static folder list: all contents under each folder copied recursively. Logs remapped to outputs/original_logs/"
}
```

## Disabling Artifact Copying

Use `--no-artifacts` flag with the extractor to skip artifact path inclusion:

```powershell
python -m extractor.extract_jobs --source config/source.json --output data/jobs.json --no-artifacts
```

When disabled, manifests will have `"disabled": true` and `"relative_paths": []`.

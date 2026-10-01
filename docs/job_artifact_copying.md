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
- **Blob enumeration at build time**: `build_pipeline` lists each job's blobs under the folder prefixes and puts them in the manifest (`--artifact-access sas`, default). In `rbac` mode the replay step lists them itself
- **No individual file enumeration during extraction**: Extraction phase uses only 4 static folder prefixes
- **Log consolidation**: All log folders are copied into `outputs/original_logs/` to keep them with the replay job's outputs
- **Performance during extraction**: Eliminates O(files) or O(directories) REST calls during extraction - uses only 4 static folder prefixes per job
- **Per-blob read-only SAS**: Flat-namespace storage cannot scope a SAS to a prefix, so a container SAS would expose every job. Each blob gets its own read-only user-delegation SAS (default 2 h, `--sas-hours`). Works cross-tenant. The manifest is uploaded as job input and holds these tokens, so treat it as sensitive
- **RBAC mode** (`--artifact-access rbac`): no tokens at all. The replay job's user identity needs *Storage Blob Data Reader* on the source storage. Only works if that identity is allowed on the source storage (usually same-tenant)

## Manifest Structure

Each job gets an artifact manifest file with:

```json
{
  "schema_version": 2,
  "disabled": false,
  "access": "sas",
  "original_run_id": "job_name",
  "source": {
    "account": "source_storage_account",
    "container": "azureml",
    "prefix": "ExperimentRun/dcid.job_name"
  },
  "blobs": [
    { "name": "ExperimentRun/dcid.job_name/outputs/model.pkl", "size": 1234, "sas": "per_blob_read_sas_token" }
  ],
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

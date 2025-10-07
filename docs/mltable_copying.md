# MLTable Dataset Copying

This document defines how to migrate **MLTable** datasets and their data assets during cross-tenant and cross-workspace replication.

## Core Rules

- Copy only the **specified dataset versions**; do not auto-discover or merge multiple versions.
- Re-create the **data asset** at the target workspace **with the same name and version**.
- Copy all **co-located files**, including the `MLTable` definition file and any referenced data (CSV, Parquet, etc.).
- Preserve directory structure relative to the original `MLTable` root.

## Datastore Handling

- Many `MLTable` files reference data via `azureml://datastores/<name>/paths/...`.
- Assume the **same datastore name exists** in the target workspace.
- Update every symbolic link or YAML path to point to the target workspace.
- If a datastore is **missing**, raise an error and log the dataset name; do not create datastores automatically.

## Data Transfer Method

- Prefer **`azcopy`** or the **Azure Blob SDK** for direct cross-account transfers.
- Avoid local download / re-upload paths unless explicitly requested (for debugging or private endpoints).
- Handle **large file counts and large total sizes** efficiently; parallelize within storage-account throughput limits.
- Default concurrency: 12 workers; make configurable.

## Safety & Verification

- Verify the byte count and file hash (MD5 / SHA-256) where available.
- Log total files copied and total size.
- On mismatch, abort and surface a clear, actionable error message.

## Authentication

- Use **`AzureCliCredential`** for all authentication.
- Log in separately to **source** and **target** tenants (`az login --tenant <id>`).
- Never embed connection strings, SAS tokens, or account keys in code or configs.

## Non-Goals

- No automatic datastore creation or registration.
- No data governance, RBAC, or policy replication.
- No portal or UI integration.

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

- Prefer **`azcopy`** or the **Azure Storage SDK** (`BlobServiceClient.start_copy_from_url`) for **server-side, cross-account copies**.
- When the storage account grants user delegation keys, generate a temporary SAS for each blob to keep the copy fully asynchronous; otherwise fall back to a sync copy with `requires_sync=True` (the tooling logs `copy_sync_mode`).
- Copy the entire MLTable root folder, including the `MLTable` definition file and every referenced blob.
- Handle **large file counts and large total sizes** efficiently; parallelize within storage-account throughput limits.
- Default concurrency: 12 workers; expose as a configurable option.

## Reference Rewriting & Validation

- After the blobs land in the target datastore, rewrite every `azureml://` reference found in text-based artifacts (`MLTable`, `.jsonl`, `.json`, `.yaml`, `.txt`, etc.) so that subscription, resource group, workspace, and path segments now point at the target workspace and the copied prefix (including any `--target-prefix`).
- If the destination prefix already contains blobs and `--data-overwrite` is **not** supplied, the importer now reuses the existing files instead of failing. Use the flag (or pick a fresh `--target-prefix`) whenever a new export produces different content and you need the copy to refresh the payload.
- Rewriting runs in-memory against each text blob (capped at 32 MiB); large files log `rewrite_skip_large_blob` and are left untouched for manual follow-up.
- Each rewrite emits `rewrote_blob_references` entries and a per-prefix `rewrite_summary` with the number of files and links updated.
- The exporter/importer still avoids bulk staging of payloads—only the small, text-based manifests are downloaded to patch symbolic links.

## Safety & Verification

- Verify the byte count and file hash (MD5 / SHA-256) where available.
- Log total files copied and total size.
- On mismatch, abort and surface a clear, actionable error message.

## Authentication

- Use **`AzureCliCredential`** for all Azure ML and storage operations.
- Sign in once using `az login` with an account that has access to the source and target tenants; the tooling will request tokens for each tenant via `AzureCliCredential(tenant_id=...)`.
- Never embed connection strings, SAS tokens, or account keys in code or configs.

## Non-Goals

- No automatic datastore creation or registration.
- No data governance, RBAC, or policy replication.
- No portal or UI integration.

# Cross-Tenant MLTable Migration Guide

Author: (add name/date as needed)  
Last Updated: 2025-08-27

## 1. Purpose

Enable selective migration of Azure Machine Learning MLTable (table data asset) versions from a source workspace in one tenant to a target workspace in another tenant **prior to deleting the source workspace**. Ensures the same asset name + version (where possible), underlying data files, and MLTable blueprint (YAML) are preserved.

## 2. In-Scope / Constraints

| Aspect                 | Decision / Assumption                                                           |
| ---------------------- | ------------------------------------------------------------------------------- |
| Data asset type        | Only `mltable` (table) data assets                                              |
| Storage types          | Source + target use the default workspace blob datastore (`workspaceblobstore`) |
| File formats           | CSV / Parquet (no Delta Lake complexity)                                        |
| Transformations        | Supported if already in YAML; no refactoring beyond workspace path rewrite      |
| Versions               | Explicit list (e.g. `9`) – no bulk “all versions” unless later extended         |
| Access control         | Recreated implicitly; no role / ACL migration attempted                         |
| Lineage                | Ignored (job references not preserved)                                          |
| Cross-tenant registry  | Not used (registry does not cross tenants)                                      |
| Collisions             | Resolve with suffix pattern + tagging or configured policy                      |
| Delta tables           | Out of scope (unless trivial – currently excluded)                              |
| Non-default datastores | Out of scope (fail fast)                                                        |

## 3. Key Concepts

An MLTable asset = (1) **MLTable YAML blueprint** + (2) metadata (name, version, tags) + (3) **referenced storage content**. Migration = copy referenced data + copy (or minimally rewrite) YAML + re-register with same semantics in target workspace.

## 4. MoSCoW Priorities

### Must

- Read migration config (assets + versions)
- Extract source asset metadata & download MLTable YAML
- Copy referenced data subtree to target default blob datastore
- Rewrite workspace-specific URIs only if required
- Register asset with same name & version (collision policy applied)
- Collision handling (suffix strategy default) + tagging
- Dry-run mode (plan only) + structured manifest output
- Minimal logging / progress

### Should

- Integrity checks: file count + total bytes (pre/post)
- Optional hash sampling (first N files)
- Validation load: `mltable.load(...).to_pandas_dataframe().head()`
- Parallel copy (thread pool / AzCopy fallback)
- Idempotent reruns (skip identical destination)
- Configurable collision policy: `suffix|skip|fail`
- Wildcard helper (future): `latest`

### Could

- HTML / Markdown summary report
- Retry with exponential backoff + transient error classification
- Diff view: original vs rewritten YAML
- Per-asset destination override path
- Resume journal for interrupted copies
- Manifest -> CSV + HTML dashboard

### Won’t (explicit exclusions)

- Delta Lake full semantics (time travel, `_delta_log` integrity) – skipped
- ADLS Gen2 ACL / POSIX permission cloning
- Role assignment / identity migration
- Job lineage or automatic pipeline rewiring
- Multi-datastore or multi-cloud path normalization

## 5. Configuration Schema (Proposed `migration_config.json`)

```jsonc
{
  "migration_batch_id": "optional-explicit-or-auto-generate",
  "source_workspace_config": "config/source_config.json",
  "target_workspace_config": "config/target_config.json",
  "defaults": {
    "destination_prefix": "migrated", // root folder under workspaceblobstore
    "collision_policy": "suffix", // suffix | skip | fail
    "collision_suffix_pattern": "{version}_m{n}", // token {version} + {n}
    "parallel_copies": 8,
    "hash_sample_files": 10,
    "dry_run": true
  },
  "assets": [
    {
      "name": "customer_profiles",
      "versions": ["9"],
      "destination_subpath": null
    },
    {
      "name": "orders_daily",
      "versions": ["9", "12"],
      "destination_subpath": "orders_custom"
    }
  ]
}
```

## 6. Destination Path Convention

```text
workspaceblobstore/paths/{destination_prefix}/{asset_name or override}/{version}/
```

All referenced files from source asset version copied beneath that root preserving relative structure. The MLTable YAML placed at that root as `MLTable` (unchanged or minimally rewritten).

## 7. Workflow (Must Path)

1. Load config & derive / create `migration_batch_id` (UUID if missing).
2. Instantiate `MLClient` for source and target (separate credentials / tenants).
3. For each asset spec:
   1. For each explicit version:
      - Fetch source data asset: `ml_client.data.get(name, version)`.
      - Determine source asset root path (blob prefix containing `MLTable`).
      - Download `MLTable` file.
      - Parse YAML (validate only supported path entries: `file`, `folder`, `pattern`).
      - Enumerate blob objects under the asset root (list).
      - Build copy plan → destination prefix.
      - If `dry_run`: record plan & skip I/O; continue to registration simulation.
      - Else copy files (parallel) + capture metrics (file_count, bytes_total, duration).
      - Rewrite YAML if subscription / resource group / workspace tokens differ in any `azureml://` paths (string substitution only).
      - Upload YAML to destination root.
      - Attempt registration with original version.
      - On 409 conflict → apply collision policy.
      - Add standard tags (see §10).
      - Validation step (if enabled): load head(5) & optionally compute sample hashes.
      - Append entry to manifest.
4. Write manifest JSON (and optional CSV/Markdown).
5. Exit non-zero if any Must failures (unless policy = skip and all failures were skips).

## 8. Collision Handling

Algorithm (policy = `suffix`):

1. Try register requested version (e.g. `9`).
2. If conflict, iterate `n=1..N_max` producing candidate: `pattern.replace('{version}', '9').replace('{n}', n)`.
3. Register first succeeding candidate.
4. Tags: `original_version`, `final_version`, `collision_resolved=true/false`, `original_version_retained`.
5. If `N_max` exhausted → failure.

Recommended `N_max` = 20.

## 9. Idempotency Rules

Before copy: if destination root already exists AND manifest indicates successful prior migration for same batch → skip.
If different batch but identical file count + bytes + hash samples → record `skipped_existing=true`.

## 10. Tagging Strategy

Minimum tags applied to every migrated (or attempted) asset version:

| Tag                          | Value                                       |
| ---------------------------- | ------------------------------------------- |
| `migrated_from_workspace`    | source workspace name                       |
| `migrated_from_subscription` | source subscription id                      |
| `migration_batch_id`         | batch UUID                                  |
| `migration_tool_version`     | semantic version of migrator (e.g. `0.1.0`) |
| `original_asset_id`          | source asset ARM/ID                         |
| `original_version`           | requested version string                    |
| `final_version`              | registered version (may differ)             |
| `collision_resolved`         | true/false                                  |
| `original_version_retained`  | true/false                                  |
| `migrated_on_utc`            | ISO8601 timestamp                           |

## 11. Validation & Integrity (Should)

| Check            | Method                           | Failure Handling             |
| ---------------- | -------------------------------- | ---------------------------- |
| File count match | compare pre vs post list size    | warn/fail (configurable)     |
| Total bytes      | sum sizes                        | warn/fail                    |
| Hash sample      | md5 / sha256 first N small files | warn -> escalate if mismatch |
| Load test        | `mltable.load` + head()          | fail (Must for confidence)   |

Potential extension: row count sample (cap to threshold) to ensure not empty after filters.

## 12. Error Handling & Retry

| Error Type                         | Strategy                                      |
| ---------------------------------- | --------------------------------------------- |
| 404 (asset missing)                | Record failure, continue batch                |
| 403 (auth)                         | Abort batch (environment misconfigured)       |
| Blob copy transient (timeout, 5xx) | Exponential backoff (e.g. 3 retries, base 2s) |
| YAML parse error                   | Fail that asset/version (invalid)             |
| Registration 409                   | Collision handler                             |
| Registration 4xx other             | Log & continue (operator review)              |

## 13. Security Considerations

- Use least-privilege identity with read on source storage, write on target.
- Avoid SAS tokens in manifest; store only canonical URIs.
- Manifest may contain source IDs but no secrets.
- Ensure source deletion only after manifest success + spot validation.

## 14. Limitations / Out of Scope

Already enumerated in §4 Won’t list. Explicitly: no permission/ACL copy, no Delta semantics, no automation of downstream pipeline updates.

## 15. Quick Start (Example Flow)

1. Author `migration_config.json` (see §5).
2. Dry run:

```powershell
python -m tools.migrate_mltables --config migration_config.json --dry-run
```

1. Review plan & manifest (ensure all assets resolvable).
2. Execute:

```powershell
python -m tools.migrate_mltables --config migration_config.json
```

1. Inspect `migration_manifest.json`.
2. Perform spot validation in target (Studio or SDK) – open data asset, preview sample.
3. Schedule deletion of source workspace only after sign-off.

## 16. Manifest Structure (Example Record)

```json
{
  "asset_name": "customer_profiles",
  "requested_version": "9",
  "final_version": "9",
  "collision_resolved": false,
  "original_asset_id": "/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.MachineLearningServices/workspaces/<ws>/data/customer_profiles/versions/9",
  "source_file_count": 42,
  "dest_file_count": 42,
  "source_bytes": 1234567,
  "dest_bytes": 1234567,
  "hash_sample_count": 10,
  "hash_mismatches": 0,
  "validation_loaded": true,
  "status": "success",
  "migrated_on_utc": "2025-08-27T10:11:12Z",
  "migration_batch_id": "7b9a8c0e-..."
}
```

## 17. Pseudocode (Reference)

```python
for spec in cfg.assets:
	for ver in spec.versions:
		record = init_record()
		try:
			src_asset = src_client.data.get(spec.name, ver)
		except ResourceNotFoundError:
			record['status'] = 'missing'
			write(record)
			continue

		yaml_text = download_mltable(src_asset)
		meta = parse_yaml(yaml_text)
		plan = build_copy_plan(src_asset, spec, ver, cfg.defaults)
		if not cfg.defaults.dry_run:
			copy_results = copy_files(plan)
			yaml_text2 = maybe_rewrite_workspace_tokens(yaml_text)
			upload_yaml(plan.dest_root, yaml_text2)
			final_version, collision = register_with_collision(policy, spec.name, ver)
			validate_results = validate(plan)
		else:
			final_version, collision = ver, False
		enrich_record(record, final_version, collision, copy_results, validate_results)
		write(record)
```

## 18. Future Enhancements

- Delta Lake inclusion with completeness checks
- Multi-datastore mapping + path rewriting rules
- Content-based deduplication (skip identical data versions)
- Interactive TUI/CLI progress dashboard
- Azure Monitor logs / telemetry integration
- Automated rollback (delete partial target data on failure)

## 19. Glossary

| Term      | Definition                                                        |
| --------- | ----------------------------------------------------------------- |
| MLTable   | YAML blueprint + engine describing how to load tabular data       |
| Collision | Requested version already exists in target workspace              |
| Manifest  | Consolidated JSON reporting per migrated asset version            |
| Dry Run   | Simulation mode producing plan without copy/register side effects |

---

This document is self-contained; implementers need only valid source/target workspace configs and the migration script aligning with the logic herein.

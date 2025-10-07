# Minimal MLTable YAML Migrator

Exports MLTable YAML definitions + minimal manifest, and re-imports them in another workspace while copying all co-located blobs directly between datastores (no local staging).

## Selection File Schema

Minimal (legacy) form:

```yaml
batch_id: run-2024-10-01
mltables:
  - name: my_table
    versions: [1, 2]
  - name: another_table # all versions
```

Recommended full form with workspace context:

```yaml
batch_id: 12345
source:
  tenant_id: "00000000-0000-0000-0000-000000000000" # optional
  subscription_id: "<sub-guid>"
  resource_group: "rg-source"
  workspace_name: "ws-source"
target:
  tenant_id: "00000000-0000-0000-0000-000000000000" # optional
  subscription_id: "<sub-guid>"
  resource_group: "rg-target"
  workspace_name: "ws-target"
mltables:
  - name: dataset_a
    versions: [1, 2]
  - name: dataset_b
  - name: dataset_c
    versions: ["2023.05.05.065003"]
```

CLI overrides (subscription / resource-group / workspace) take precedence over the YAML. If source/target blocks are omitted you MUST provide CLI values.

## Export

```bash
# Use workspace from selection.source (authenticate once via `az login`; the tool requests per-tenant tokens automatically)
python export_yaml.py --selection selection.yaml --out-dir exported_mltables

# Override source workspace explicitly
python export_yaml.py \
  --selection selection.yaml \
  --subscription SUB \
  --resource-group RG \
  --workspace WS \
  --out-dir exported_mltables

# Dry run (no files written)
python export_yaml.py --selection selection.yaml --out-dir exported_mltables --dry-run
```

Output directory structure:

```text
exported_mltables/
  my_table/
    1/MLTable
    1/manifest.json
    2/MLTable
    2/manifest.json
  another_table/
    <version>/MLTable
    <version>/manifest.json
```

## Import

```bash
# Copy blobs + register using selection.source/target (preferred)
python import_yaml.py \
  --selection selection.yaml \
  --source-dir exported_mltables \
  --on-exists fail|skip|suffix

# Explicit overrides (supply both source and target workspaces)
python import_yaml.py \
  --source-subscription SUB_SRC \
  --source-resource-group RG_SRC \
  --source-workspace WS_SRC \
  --subscription SUB_TGT \
  --resource-group RG_TGT \
  --workspace WS_TGT \
  --source-dir exported_mltables

# Optional: isolate copies under a custom prefix and allow overwriting existing blobs
python import_yaml.py \
  --selection selection.yaml \
  --source-dir exported_mltables \
  --target-prefix migrations/batch-001 \
  --copy-concurrency 16 \
  --data-overwrite

# Dry run (no data copy, no registration)
python import_yaml.py --selection selection.yaml --source-dir exported_mltables --dry-run

# Legacy behaviour (retain original paths; no blob copy)
python import_yaml.py --selection selection.yaml --source-dir exported_mltables --skip-data-copy
```

Key flags:

- `--source-*` & `--subscription/--resource-group/--workspace` – allow CLI overrides when the selection file lacks workspace metadata. Provide matching tenant IDs with `--source-tenant` / `--tenant` for cross-tenant moves.
- Both workspaces can be accessed after a single `az login` as long as the signed-in principal has RBAC in each tenant; no additional logins are required.
- `--target-prefix` – prepend a folder (e.g. `migrations/batch-42`) to every copied path inside the target datastore.
- `--copy-concurrency` (default **12**) – number of parallel blob copy workers.
- `--data-overwrite` – delete existing blobs at the target prefix before copying.
- `--skip-data-copy` – register assets without moving data (falling back to scope rewrite rules).

Collision strategies (when an asset version already exists):

- fail (default) – abort on first existing version
- skip – leave existing version untouched
- suffix – keep asset name; append \_1, \_2, ... to version until free (e.g. 1 -> 1_1)

Scope rewrite (azureml:// path from different workspace):

--on-scope-mismatch options:

- rewrite (default) – substitute subscription/resourceGroup/workspace with target while preserving datastore + relative path. With data copy enabled this happens automatically once blobs land in the target workspace.
- skip – do not import that asset.
- fail – abort on first mismatch.

Dry run:

- Export: lists would_export events (no files written).
- Import: lists would_import plus counts (skipped, rewritten, renamed).

## Notes

- Only path reference is preserved; underlying data is assumed persistent.
- Minimal MLTable written; add transformations manually if required.
- Manifest retains tags, properties, description.
- Selection file may include source/target workspace blocks; CLI args override them.
- --on-scope-mismatch rewrite (default) swaps workspace scope in azureml:// URIs; use skip or fail to change.
- --dry-run available for both export and import.

## Log Events

Events are emitted as one-line JSON to stdout for easy ingestion:

| event                     | meaning                                         |
| ------------------------- | ----------------------------------------------- |
| exported                  | An asset version YAML+manifest written (export) |
| would_export              | Dry-run export listing                          |
| imported                  | Asset version registered (import)               |
| would_import              | Dry-run import listing                          |
| skip_existing             | Version already present and skipped             |
| version_collision_renamed | Version collision resolved via suffix           |
| rewrite_scope             | azureml:// path scope rewritten to target       |
| skip_scope_mismatch       | azureml:// path skipped due to scope mismatch   |
| warn_missing_path         | Manifest had no path field                      |
| done / done_import        | Summary line with counters                      |

Example:

```json
{
  "ts": "2025-09-02T12:34:56.123456Z",
  "event": "exported",
  "name": "dataset_a",
  "version": "1",
  "dir": "exported_mltables/dataset_a/1"
}
```

## Summary & Recommendations

1. Populate selection.yaml with source + target for reproducibility.
2. Run export with --dry-run first to verify scope & counts.
3. If datastores are mirrored across workspaces, keep default rewrite; otherwise supply skip and manually adjust paths.
4. Prefer suffix collision strategy to preserve stable asset names.
5. Archive the selection file + exported manifests for audit.

#!/usr/bin/env python
"""Export specified MLTable YAML(s) from a source workspace.

Usage (examples):
    # Prefer workspace metadata from selection.source
    python export_yaml.py --selection selection.yaml --out-dir exported_mltables

    # Override selection.source values explicitly
    python export_yaml.py --selection selection.yaml --subscription SUB --resource-group RG --workspace WS \
         --out-dir exported_mltables

Selection file schema (YAML):
  batch_id: run-2024-10-01
  mltables:
    - name: my_table
      versions: [1,2]
    - name: another_table   # all versions (versions omitted)

This script only exports the YAML definition (MLTable file) plus a small manifest JSON.
No data files are copied.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Iterable
from azure.identity import AzureCliCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from .selection import SelectionSpec
from .util_logging import log


def iter_datasets(
    client: MLClient, name: str, versions: list[str] | None
) -> Iterable[Data]:
    if versions:
        for v in versions:
            ds = client.data.get(name=name, version=str(v))
            yield ds
    else:
        # all versions
        for ds in client.data.list(name=name):
            yield ds


def export_yaml(ds: Data, target_dir: Path):
    name = getattr(ds, "name", "unknown")
    version = str(getattr(ds, "version", "0"))
    path = getattr(ds, "path", None)
    ds_dir = target_dir / name / version
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct a minimal MLTable referencing the original path only.
    mltable_yaml_lines = ["paths:"]
    mltable_yaml_lines.append(f"  - file: {path if path else '.'}")
    mltable_yaml_lines.append(
        "# NOTE: Reconstructed minimal MLTable. Add transforms if needed."
    )
    (ds_dir / "MLTable").write_text("\n".join(mltable_yaml_lines) + "\n")

    dtype = getattr(ds, "type", None)
    if dtype is not None and hasattr(dtype, "value"):
        dtype = dtype.value

    manifest = {
        "name": name,
        "version": version,
        "description": getattr(ds, "description", None),
        "type": dtype,
        "path": path,
        "tags": getattr(ds, "tags", None),
        "properties": getattr(ds, "properties", None),
    }
    (ds_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    log("exported", name=name, version=version, dir=str(ds_dir))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--selection",
        required=True,
        help="YAML selection file (with optional source workspace)",
    )
    p.add_argument(
        "--subscription", help="Override source subscription (else selection.source)"
    )
    p.add_argument("--resource-group", help="Override source resource group")
    p.add_argument("--workspace", help="Override source workspace")
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List assets that would be exported without writing files",
    )
    return p.parse_args()


def main():
    args = parse_args()
    spec = SelectionSpec.load(args.selection)
    target_dir = Path(args.out_dir)
    if not args.dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
    cred = AzureCliCredential()
    # Resolve workspace from CLI overrides or selection.source
    if args.subscription and args.resource_group and args.workspace:
        sub = args.subscription
        rg = args.resource_group
        ws = args.workspace
    elif spec.source:
        sub = spec.source.subscription_id
        rg = spec.source.resource_group
        ws = spec.source.workspace_name
    else:
        raise SystemExit("Source workspace not provided via CLI or selection.source")
    client = MLClient(
        cred, subscription_id=sub, resource_group_name=rg, workspace_name=ws
    )
    exported = 0
    for table in spec.mltables:
        versions = table.versions
        for ds in iter_datasets(client, table.name, versions):
            if args.dry_run:
                log(
                    "would_export",
                    name=getattr(ds, "name", None),
                    version=getattr(ds, "version", None),
                    path=getattr(ds, "path", None),
                )
            else:
                export_yaml(ds, target_dir)
            exported += 1
    log("done", batch_id=spec.batch_id, count=exported, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Import MLTable YAML(s) previously exported by export_yaml.py.

Collision strategies:
  --on-exists fail (default) | skip | suffix
If suffix is chosen, a numeric suffix _1, _2, ... is appended until free.

Usage:
  python import_yaml.py --subscription SUB --resource-group RG --workspace WS \
     --source-dir exported_mltables --on-exists skip
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from util_logging import log
from selection import SelectionSpec


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--selection", help="Selection YAML providing target workspace metadata"
    )
    p.add_argument(
        "--subscription",
        help="Override target subscription (else from selection.target)",
    )
    p.add_argument("--resource-group", help="Override target resource group")
    p.add_argument("--workspace", help="Override target workspace")
    p.add_argument(
        "--source-dir", required=True, help="Directory created by export_yaml.py"
    )
    p.add_argument("--on-exists", choices=["fail", "skip", "suffix"], default="fail")
    p.add_argument(
        "--on-scope-mismatch",
        choices=["rewrite", "skip", "fail"],
        default="rewrite",
        help=(
            "What to do when path azureml:// scope (sub/rg/ws) differs: "
            "rewrite = replace scope with target (default), skip = ignore asset, fail = raise"
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Show actions without creating assets"
    )
    return p.parse_args()


def load_exports(root: Path):
    for name_dir in root.iterdir():
        if not name_dir.is_dir():
            continue
        for ver_dir in name_dir.iterdir():
            if not ver_dir.is_dir():
                continue
            mltable_file = ver_dir / "MLTable"
            manifest_file = ver_dir / "manifest.json"
            if not mltable_file.exists():
                continue
            manifest = (
                json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
            )
            yield name_dir.name, ver_dir.name, mltable_file, manifest


def ensure_unique_version(client: MLClient, name: str, base_version: str) -> str:
    """Return a non-existing version string by appending _1, _2, ... to base_version.

    This keeps the original asset name stable so dependent references remain valid.
    """
    # quick check if base_version already free
    try:
        client.data.get(name=name, version=base_version)
        exists = True
    except Exception:
        return base_version  # free
    if not exists:
        return base_version
    idx = 1
    while True:
        candidate = f"{base_version}_{idx}"
        try:
            client.data.get(name=name, version=candidate)
            # exists -> try next
        except Exception:
            return candidate
        idx += 1


_AZUREML_URI_RE = re.compile(
    r"^azureml://subscriptions/(?P<sub>[^/]+)/resourcegroups/(?P<rg>[^/]+)/workspaces/(?P<ws>[^/]+)/datastores/(?P<ds>[^/]+)/paths/(?P<path>.+)$",
    re.IGNORECASE,
)


def _parse_azureml_uri(uri: str):
    m = _AZUREML_URI_RE.match(uri)
    return m.groupdict() if m else None


def main():
    args = parse_args()
    source_root = Path(args.source_dir)
    spec = SelectionSpec.load(args.selection) if args.selection else None
    # Resolve target workspace precedence: explicit CLI > selection.target
    if args.subscription and args.resource_group and args.workspace:
        sub = args.subscription
        rg = args.resource_group
        ws = args.workspace
    elif spec and spec.target:
        sub = spec.target.subscription_id
        rg = spec.target.resource_group
        ws = spec.target.workspace_name
    else:
        raise SystemExit("Target workspace not provided via CLI or selection.target")
    cred = DefaultAzureCredential()
    client = MLClient(
        cred, subscription_id=sub, resource_group_name=rg, workspace_name=ws
    )

    created = 0
    skipped = 0
    rewritten = 0
    renamed = 0
    for name, version, mltable_path, manifest in load_exports(source_root):
        target_name = name
        # check existence
        exists = False
        try:
            client.data.get(name=target_name, version=version)
            exists = True
        except Exception:
            exists = False
        if exists:
            if args.on_exists == "skip":
                log("skip_existing", name=name, version=version)
                skipped += 1
                continue
            elif args.on_exists == "fail":
                if args.dry_run:
                    log("would_fail_existing", name=name, version=version)
                    skipped += 1
                    continue
                raise RuntimeError(f"Asset {name}:{version} already exists")
            elif args.on_exists == "suffix":
                new_version = ensure_unique_version(client, name, version)
                log(
                    "version_collision_renamed",
                    name=name,
                    old_version=version,
                    new_version=new_version,
                )
                version = new_version
                renamed += 1
        # register new data asset referencing original path (we kept only path reference in manifest)
        original_path = manifest.get("path")
        if not original_path:
            log("warn_missing_path", name=name, version=version)
            continue
        # If path is from a different workspace, handle according to strategy.
        if isinstance(original_path, str) and original_path.startswith("azureml://"):
            meta = _parse_azureml_uri(original_path)
            if meta:
                scope_differs = (
                    meta["sub"].lower() != sub.lower()
                    or meta["rg"].lower() != rg.lower()
                    or meta["ws"].lower() != ws.lower()
                )
                if scope_differs:
                    action = args.on_scope_mismatch
                    if action == "skip":
                        log(
                            "skip_scope_mismatch",
                            name=name,
                            version=version,
                            source_sub=meta["sub"],
                            source_rg=meta["rg"],
                            source_ws=meta["ws"],
                        )
                        skipped += 1
                        continue
                    if action == "fail":
                        raise RuntimeError(
                            f"Scope mismatch for {name}:{version} path points to "
                            f"{meta['sub']}/{meta['rg']}/{meta['ws']}"
                        )
                    if action == "rewrite":
                        # Rebuild azureml:// uri with target workspace scope, keeping datastore + relative path
                        new_path = (
                            f"azureml://subscriptions/{sub}/resourcegroups/{rg}/"
                            f"workspaces/{ws}/datastores/{meta['ds']}/paths/{meta['path']}"
                        )
                        log(
                            "rewrite_scope",
                            name=name,
                            version=version,
                            old=original_path,
                            new=new_path,
                        )
                        original_path = new_path
                        rewritten += 1

        if args.dry_run:
            log("would_import", name=target_name, version=version, path=original_path)
            continue
        data_asset = Data(
            name=target_name,
            version=version,
            path=original_path,
            description=manifest.get("description"),
            type=manifest.get("type", "mltable"),
            tags=manifest.get("tags"),
            properties=manifest.get("properties"),
        )
        client.data.create_or_update(data_asset)
        created += 1
        log("imported", name=target_name, version=version, original=name)
    log(
        "done_import",
        created=created,
        skipped=skipped,
        rewritten=rewritten,
        renamed_versions=renamed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

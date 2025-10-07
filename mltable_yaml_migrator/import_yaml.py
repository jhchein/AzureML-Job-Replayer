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
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from azure.identity import AzureCliCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.storage.blob import (
    BlobSasPermissions,
    BlobServiceClient,
    generate_blob_sas,
)

from .util_logging import log
from .selection import SelectionSpec, WorkspaceRef
from utils.aml_clients import (
    AzureCliCredentialPool,
    create_ml_client,
    get_cli_credential,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--selection", help="Selection YAML providing target workspace metadata"
    )
    p.add_argument(
        "--source-subscription",
        help="Override source subscription (else from selection.source)",
    )
    p.add_argument("--source-resource-group", help="Override source resource group")
    p.add_argument("--source-workspace", help="Override source workspace")
    p.add_argument(
        "--source-tenant",
        help="Optional tenant ID for the source workspace (otherwise selection.source)",
    )
    p.add_argument(
        "--subscription",
        help="Override target subscription (else from selection.target)",
    )
    p.add_argument("--resource-group", help="Override target resource group")
    p.add_argument("--workspace", help="Override target workspace")
    p.add_argument(
        "--tenant",
        help="Optional tenant ID for the target workspace (else selection.target)",
    )
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
    p.add_argument(
        "--skip-data-copy",
        action="store_true",
        help="Register assets without copying blobs (keeps original azureml:// path)",
    )
    p.add_argument(
        "--copy-concurrency",
        type=int,
        default=12,
        help="Parallel workers for blob copy (when copying data)",
    )
    p.add_argument(
        "--data-overwrite",
        action="store_true",
        help="Delete existing blobs at the target prefix before copying",
    )
    p.add_argument(
        "--target-prefix",
        help=(
            "Optional prefix to prepend to datastore paths when copying data. "
            "Useful to isolate migrations (e.g. migrations/batch-123)."
        ),
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


_AZUREML_URI_EMBEDDED_RE = re.compile(
    r"azureml://subscriptions/(?P<sub>[^/]+)/resourcegroups/(?P<rg>[^/]+)/workspaces/(?P<ws>[^/]+)/datastores/(?P<ds>[^/]+)/paths/(?P<path>[^\"'\s}]+)",
    re.IGNORECASE,
)


_TEXT_BLOB_SUFFIXES = (
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".txt",
    ".csv",
    ".tsv",
    ".mltable",
)

_MAX_TEXT_REWRITE_BYTES = 32 * 1024 * 1024  # 32 MiB safety cap


def _is_text_like_blob(blob_name: str) -> bool:
    lower = blob_name.lower()
    if lower.endswith("/mltable") or lower.endswith("mltable"):
        return True
    return any(lower.endswith(suffix) for suffix in _TEXT_BLOB_SUFFIXES)


def _build_scope_tuple(ws: WorkspaceRef) -> Tuple[str, str, str]:
    return (
        (ws.subscription_id or "").lower(),
        (ws.resource_group or "").lower(),
        (ws.workspace_name or "").lower(),
    )


def _replace_path_prefix(
    path: str, old_prefix: str, new_prefix: str
) -> Tuple[str, bool]:
    if not old_prefix:
        return path, False
    trailing = path.endswith("/")
    path_norm = path.strip("/")
    old_norm = old_prefix.strip("/")
    if not old_norm:
        return path, False
    if not path_norm.startswith(old_norm):
        return path, False
    suffix = path_norm[len(old_norm) :].lstrip("/")
    new_norm = new_prefix.strip("/")
    if new_norm and suffix:
        new_path = f"{new_norm}/{suffix}"
    elif new_norm:
        new_path = new_norm
    else:
        new_path = suffix
    if trailing and new_path:
        new_path = new_path.rstrip("/") + "/"
    elif trailing and not new_path:
        new_path = ""
    if new_path == path:
        return path, False
    return new_path, True


def _rewrite_azureml_text(
    content: str,
    ctx_source: WorkspaceRef,
    ctx_target: WorkspaceRef,
    primary_datastore: str,
    source_prefix: str,
    target_prefix: str,
) -> Tuple[str, int, int]:
    source_scope = _build_scope_tuple(ctx_source)
    if not all(source_scope):
        return content, 0, 0
    primary_ds_lower = (primary_datastore or "").lower()
    source_prefix_norm = source_prefix.rstrip("/")
    target_prefix_norm = target_prefix.rstrip("/")
    replacements = 0
    path_replacements = 0

    def _sub(match):
        nonlocal replacements, path_replacements
        sub = (match.group("sub") or "").lower()
        rg = (match.group("rg") or "").lower()
        ws = (match.group("ws") or "").lower()
        if (sub, rg, ws) != source_scope:
            return match.group(0)
        datastore = match.group("ds")
        path = match.group("path")
        new_path = path
        if datastore and datastore.lower() == primary_ds_lower:
            candidate, changed = _replace_path_prefix(
                path, source_prefix_norm, target_prefix_norm
            )
            if changed:
                new_path = candidate
                path_replacements += 1
        new_uri = (
            f"azureml://subscriptions/{(ctx_target.subscription_id or '').lower()}/"
            f"resourcegroups/{(ctx_target.resource_group or '').lower()}/"
            f"workspaces/{(ctx_target.workspace_name or '').lower()}/"
            f"datastores/{datastore}/paths/{new_path}"
        )
        replacements += 1
        return new_uri

    new_content = _AZUREML_URI_EMBEDDED_RE.sub(_sub, content)
    return new_content, replacements, path_replacements


def _parse_azureml_uri(uri: str):
    m = _AZUREML_URI_RE.match(uri)
    return m.groupdict() if m else None


@dataclass(frozen=True)
class WorkspaceKey:
    subscription_id: str
    resource_group: str
    workspace_name: str

    def normalized(self) -> Tuple[str, str, str]:
        return (
            self.subscription_id.lower(),
            self.resource_group.lower(),
            self.workspace_name.lower(),
        )


@dataclass
class DatastoreInfo:
    account_url: str
    container: str
    datastore_type: str


def _workspace_ref_from_cli(
    sub: Optional[str],
    rg: Optional[str],
    ws: Optional[str],
    tenant: Optional[str],
) -> Optional[WorkspaceRef]:
    if sub and rg and ws:
        return WorkspaceRef(
            subscription_id=sub,
            resource_group=rg,
            workspace_name=ws,
            tenant_id=tenant,
        )
    return None


def _workspace_from_meta(
    meta: Dict[str, str],
    template: Optional[WorkspaceRef],
) -> WorkspaceRef:
    tenant_id = None
    if template:
        if (
            meta["sub"].lower() == template.subscription_id.lower()
            and meta["rg"].lower() == template.resource_group.lower()
            and meta["ws"].lower() == template.workspace_name.lower()
        ):
            tenant_id = template.tenant_id
    return WorkspaceRef(
        subscription_id=meta["sub"],
        resource_group=meta["rg"],
        workspace_name=meta["ws"],
        tenant_id=tenant_id,
    )


def _get_workspace_client(
    cache: Dict[Tuple[str, str, str], MLClient],
    credential_pool: AzureCliCredentialPool,
    workspace: WorkspaceRef,
) -> MLClient:
    key = WorkspaceKey(
        subscription_id=workspace.subscription_id,
        resource_group=workspace.resource_group,
        workspace_name=workspace.workspace_name,
    ).normalized()
    if key not in cache:
        cache[key] = create_ml_client(
            subscription_id=workspace.subscription_id,
            resource_group=workspace.resource_group,
            workspace_name=workspace.workspace_name,
            tenant_id=workspace.tenant_id,
            credential_pool=credential_pool,
        )
    return cache[key]


def _ensure_blob_datastore(ds) -> DatastoreInfo:
    dtype = getattr(ds, "type", None) or getattr(ds, "datastore_type", "").lower()
    if dtype and isinstance(dtype, str):
        dtype_lower = dtype.lower()
    else:
        dtype_lower = ""
    if dtype_lower not in {"azure_blob", "azureblob"}:
        raise RuntimeError(
            f"Datastore {ds.name} must be an Azure Blob datastore, got type '{dtype}'"
        )
    container = getattr(ds, "container_name", None)
    if not container:
        raise RuntimeError(f"Datastore {ds.name} missing container name")
    account_url = getattr(ds, "account_url", None)
    if not account_url:
        endpoint = getattr(ds, "account_endpoint", None)
        if endpoint:
            account_url = endpoint
        else:
            account_name = getattr(ds, "account_name", None)
            if not account_name:
                raise RuntimeError(
                    f"Datastore {ds.name} missing account endpoint/account name"
                )
            account_url = f"https://{account_name}.blob.core.windows.net"
    return DatastoreInfo(
        account_url=account_url,
        container=container,
        datastore_type=dtype_lower,
    )


def _resolve_datastore(
    client: MLClient,
    cache: Dict[Tuple[Tuple[str, str, str], str], DatastoreInfo],
    workspace: WorkspaceRef,
    datastore_name: str,
) -> DatastoreInfo:
    key = (
        WorkspaceKey(
            subscription_id=workspace.subscription_id,
            resource_group=workspace.resource_group,
            workspace_name=workspace.workspace_name,
        ).normalized(),
        datastore_name.lower(),
    )
    if key not in cache:
        datastore = client.datastores.get(datastore_name)
        cache[key] = _ensure_blob_datastore(datastore)
    return cache[key]


def _ensure_trailing_slash(prefix: str) -> str:
    if not prefix:
        return ""
    return prefix if prefix.endswith("/") else prefix + "/"


def _account_name_from_url(account_url: str) -> Optional[str]:
    if not account_url:
        return None
    parsed = urlparse(account_url)
    host = parsed.netloc or ""
    if not host:
        return None
    return host.split(".")[0] if host else None


def _build_target_path(original: str, override_prefix: Optional[str]) -> str:
    if not override_prefix:
        return original
    override = override_prefix.rstrip("/")
    if not original:
        return override
    return f"{override}/{original.lstrip('/')}"


def _ensure_target_prefix_clean(
    container_client,
    prefix: str,
    overwrite: bool,
) -> bool:
    blob_iter = container_client.list_blobs(name_starts_with=prefix)
    try:
        first = next(blob_iter)
    except StopIteration:
        return True
    if not overwrite:
        log(
            "existing_target_prefix",
            prefix=prefix,
            message="reuse existing blobs (no data overwrite)",
        )
        return False
    deleted = 1
    container_client.delete_blob(first.name)
    for blob in blob_iter:
        container_client.delete_blob(blob.name)
        deleted += 1
    log("deleted_existing_blobs", prefix=prefix, count=deleted)
    return True


def _rewrite_blob_references(
    datastore: DatastoreInfo,
    credential: AzureCliCredential,
    target_prefix: str,
    source_workspace: WorkspaceRef,
    target_workspace: WorkspaceRef,
    datastore_name: str,
    source_prefix: str,
    desired_prefix: str,
) -> Tuple[int, int, int]:
    service = BlobServiceClient(
        account_url=datastore.account_url, credential=credential
    )
    container_client = service.get_container_client(datastore.container)
    norm_target_prefix = _ensure_trailing_slash(target_prefix)
    files_updated = 0
    references_updated = 0
    path_updates = 0
    for blob in container_client.list_blobs(name_starts_with=norm_target_prefix):
        if not _is_text_like_blob(blob.name):
            continue
        blob_size = getattr(blob, "size", 0) or 0
        if blob_size > _MAX_TEXT_REWRITE_BYTES:
            log(
                "rewrite_skip_large_blob",
                blob=blob.name,
                size=blob_size,
                limit=_MAX_TEXT_REWRITE_BYTES,
            )
            continue
        blob_client = container_client.get_blob_client(blob.name)
        try:
            content_bytes = blob_client.download_blob().readall()
        except Exception as exc:  # pylint: disable=broad-except
            log("rewrite_download_failed", blob=blob.name, error=str(exc))
            continue
        try:
            content_text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            log("rewrite_decode_failed", blob=blob.name)
            continue
        updated_text, replacements, path_replacements = _rewrite_azureml_text(
            content_text,
            source_workspace,
            target_workspace,
            datastore_name,
            source_prefix,
            desired_prefix,
        )
        if replacements == 0 or updated_text == content_text:
            continue
        try:
            blob_client.upload_blob(updated_text.encode("utf-8"), overwrite=True)
            content_settings = getattr(blob, "content_settings", None)
            if content_settings and getattr(content_settings, "content_type", None):
                blob_client.set_http_headers(content_settings=content_settings)
        except Exception as exc:  # pylint: disable=broad-except
            log("rewrite_upload_failed", blob=blob.name, error=str(exc))
            continue
        files_updated += 1
        references_updated += replacements
        path_updates += path_replacements
        log(
            "rewrote_blob_references",
            blob=blob.name,
            replacements=replacements,
            path_rewrites=path_replacements,
        )
    if files_updated:
        log(
            "rewrite_summary",
            prefix=norm_target_prefix,
            files=files_updated,
            references=references_updated,
            path_rewrites=path_updates,
        )
    return files_updated, references_updated, path_updates


def _copy_blob(
    blob_name: str,
    size: int,
    source_container,
    target_container,
    source_prefix: str,
    target_prefix: str,
    credential: AzureCliCredential,
    source_account_name: Optional[str],
    user_delegation_key,
    sas_expiry: Optional[datetime],
) -> Tuple[int, int]:
    rel_name = blob_name[len(source_prefix) :] if source_prefix else blob_name
    target_blob_name = f"{target_prefix}{rel_name}" if target_prefix else rel_name
    source_blob = source_container.get_blob_client(blob_name)
    target_blob = target_container.get_blob_client(target_blob_name)
    if user_delegation_key and source_account_name:
        expiry = sas_expiry or (datetime.now(timezone.utc) + timedelta(hours=1))
        sas_token = generate_blob_sas(
            account_name=source_account_name,
            container_name=source_container.container_name,
            blob_name=blob_name,
            user_delegation_key=user_delegation_key,
            permission=BlobSasPermissions(read=True),
            expiry=expiry,
        )
        copy_source_url = f"{source_blob.url}?{sas_token}"
        copy = target_blob.start_copy_from_url(copy_source_url)
    else:
        token = credential.get_token("https://storage.azure.com/.default").token
        copy = target_blob.start_copy_from_url(
            source_blob.url,
            source_authorization=f"Bearer {token}",
            requires_sync=True,
        )
    status = (copy or {}).get("copy_status", "pending").lower()
    copy_id = (copy or {}).get("copy_id")
    if status not in {"success", "pending"}:
        raise RuntimeError(
            f"Copy start failed for {blob_name}: status={status}, copy_id={copy_id}"
        )
    if status != "success":
        # poll until completion
        while True:
            props = target_blob.get_blob_properties()
            cprops = getattr(props, "copy", None)
            if not cprops:
                break
            if cprops.status.lower() == "success":
                break
            if cprops.status.lower() == "failed":
                raise RuntimeError(
                    f"Copy failed for {blob_name}: {cprops.status_description}"
                )
            time.sleep(1.5)
    return 1, size or 0


def _copy_prefix(
    source_info: DatastoreInfo,
    target_info: DatastoreInfo,
    source_prefix: str,
    target_prefix: str,
    source_credential: AzureCliCredential,
    target_credential: AzureCliCredential,
    concurrency: int,
    overwrite: bool,
) -> Tuple[int, int]:
    source_service = BlobServiceClient(
        account_url=source_info.account_url, credential=source_credential
    )
    target_service = BlobServiceClient(
        account_url=target_info.account_url, credential=target_credential
    )
    source_container = source_service.get_container_client(source_info.container)
    target_container = target_service.get_container_client(target_info.container)
    source_account_name = _account_name_from_url(source_info.account_url)
    user_delegation_key = None
    sas_expiry = None
    if source_account_name:
        try:
            key_start = datetime.now(timezone.utc) - timedelta(minutes=5)
            key_expiry = datetime.now(timezone.utc) + timedelta(hours=2)
            user_delegation_key = source_service.get_user_delegation_key(
                key_start_time=key_start,
                key_expiry_time=key_expiry,
            )
            sas_expiry = key_expiry
        except Exception as exc:  # pylint: disable=broad-except
            log(
                "warn_user_delegation_key_failed",
                datastore=source_info.container,
                error=str(exc),
            )
            user_delegation_key = None
    else:
        log(
            "warn_account_name_parse_failed",
            account_url=source_info.account_url,
        )
    norm_source_prefix = _ensure_trailing_slash(source_prefix)
    norm_target_prefix = _ensure_trailing_slash(target_prefix)
    if not norm_source_prefix:
        raise RuntimeError(
            "Refusing to copy entire container; source prefix resolved to empty string"
        )
    if not norm_target_prefix:
        raise RuntimeError(
            "Refusing to write into container root; provide --target-prefix or ensure MLTable path contains folder"
        )
    if not _ensure_target_prefix_clean(target_container, norm_target_prefix, overwrite):
        return 0, 0
    blobs = list(source_container.list_blobs(name_starts_with=norm_source_prefix))
    if not blobs:
        raise RuntimeError(
            f"No blobs found at prefix '{norm_source_prefix}' in datastore {source_info.container}"
        )
    total_files = 0
    total_bytes = 0
    use_sas_copy = bool(user_delegation_key and source_account_name)
    if not use_sas_copy:
        log(
            "copy_sync_mode",
            datastore=source_info.container,
            prefix=norm_source_prefix,
        )
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(
                _copy_blob,
                blob.name,
                getattr(blob, "size", 0),
                source_container,
                target_container,
                norm_source_prefix,
                norm_target_prefix,
                source_credential,
                source_account_name,
                user_delegation_key,
                sas_expiry,
            ): blob
            for blob in blobs
        }
        for future in as_completed(futures):
            blob = futures[future]
            try:
                files, bytes_copied = future.result()
                total_files += files
                total_bytes += bytes_copied
                log(
                    "copied_blob",
                    blob=blob.name,
                    bytes=bytes_copied,
                    target_prefix=norm_target_prefix,
                )
            except Exception as exc:  # pylint: disable=broad-except
                raise RuntimeError(f"Failed to copy blob {blob.name}: {exc}") from exc
    log(
        "copy_summary",
        source_prefix=norm_source_prefix,
        target_prefix=norm_target_prefix,
        files=total_files,
        bytes=total_bytes,
    )
    return total_files, total_bytes


def main():
    args = parse_args()
    source_root = Path(args.source_dir)
    spec = SelectionSpec.load(args.selection) if args.selection else None
    target_ws = _workspace_ref_from_cli(
        args.subscription, args.resource_group, args.workspace, args.tenant
    )
    if not target_ws and spec and spec.target:
        target_ws = spec.target
    if not target_ws:
        raise SystemExit("Target workspace not provided via CLI or selection.target")
    source_ws = _workspace_ref_from_cli(
        args.source_subscription,
        args.source_resource_group,
        args.source_workspace,
        args.source_tenant,
    )
    if not source_ws and spec and spec.source:
        source_ws = spec.source
    if not args.skip_data_copy and not source_ws:
        raise SystemExit(
            "Source workspace not provided via CLI or selection.source; "
            "required to copy MLTable data. Use --skip-data-copy to register without copying."
        )

    credential_pool = AzureCliCredentialPool()
    workspace_clients: Dict[Tuple[str, str, str], MLClient] = {}
    datastore_cache: Dict[Tuple[Tuple[str, str, str], str], DatastoreInfo] = {}

    target_client = _get_workspace_client(workspace_clients, credential_pool, target_ws)

    created = 0
    skipped = 0
    rewritten = 0
    renamed = 0
    copied_bytes = 0
    copied_files = 0
    rewrite_files = 0
    rewrite_references = 0
    rewrite_path_references = 0
    for name, version, mltable_path, manifest in load_exports(source_root):
        target_name = name
        # check existence
        exists = False
        try:
            target_client.data.get(name=target_name, version=version)
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
                new_version = ensure_unique_version(target_client, name, version)
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
        target_path = original_path
        if (
            not args.skip_data_copy
            and isinstance(original_path, str)
            and original_path.startswith("azureml://")
        ):
            meta = _parse_azureml_uri(original_path)
            if not meta:
                raise RuntimeError(
                    f"Unable to parse azureml:// path for {name}:{version}: {original_path}"
                )
            source_workspace = _workspace_from_meta(meta, source_ws)
            source_client = _get_workspace_client(
                workspace_clients, credential_pool, source_workspace
            )
            source_datastore = _resolve_datastore(
                source_client, datastore_cache, source_workspace, meta["ds"]
            )
            target_datastore = _resolve_datastore(
                target_client, datastore_cache, target_ws, meta["ds"]
            )
            desired_relative = _build_target_path(meta["path"], args.target_prefix)
            if args.dry_run:
                log(
                    "would_copy_data",
                    name=name,
                    version=version,
                    datastore=meta["ds"],
                    source_prefix=meta["path"],
                    target_prefix=desired_relative,
                )
            else:
                src_cred = get_cli_credential(
                    source_workspace.tenant_id, credential_pool
                )
                tgt_cred = get_cli_credential(target_ws.tenant_id, credential_pool)
                files, bytes_copied = _copy_prefix(
                    source_datastore,
                    target_datastore,
                    meta["path"],
                    desired_relative,
                    src_cred,
                    tgt_cred,
                    concurrency=args.copy_concurrency,
                    overwrite=args.data_overwrite,
                )
                copied_files += files
                copied_bytes += bytes_copied
                log(
                    "copied_dataset",
                    name=name,
                    version=version,
                    datastore=meta["ds"],
                    files=files,
                    bytes=bytes_copied,
                )
                ref_files, ref_refs, ref_path_refs = _rewrite_blob_references(
                    target_datastore,
                    tgt_cred,
                    desired_relative,
                    source_workspace,
                    target_ws,
                    meta["ds"],
                    meta["path"],
                    desired_relative,
                )
                rewrite_files += ref_files
                rewrite_references += ref_refs
                rewrite_path_references += ref_path_refs
            target_path = (
                f"azureml://subscriptions/{target_ws.subscription_id}/"
                f"resourcegroups/{target_ws.resource_group}/"
                f"workspaces/{target_ws.workspace_name}/"
                f"datastores/{meta['ds']}/paths/{desired_relative}"
            )
        elif not args.skip_data_copy:
            log(
                "warn_non_azureml_path",
                name=name,
                version=version,
                path=original_path,
            )
        # If path is from a different workspace, handle according to strategy.
        if isinstance(original_path, str) and original_path.startswith("azureml://"):
            meta = _parse_azureml_uri(original_path)
            if meta:
                scope_differs = (
                    meta["sub"].lower() != target_ws.subscription_id.lower()
                    or meta["rg"].lower() != target_ws.resource_group.lower()
                    or meta["ws"].lower() != target_ws.workspace_name.lower()
                )
                if scope_differs and args.skip_data_copy:
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
                        rewritten += 1
                        log(
                            "rewrite_scope",
                            name=name,
                            version=version,
                            old=original_path,
                            new=target_path,
                        )
                        target_path = (
                            f"azureml://subscriptions/{target_ws.subscription_id}/"
                            f"resourcegroups/{target_ws.resource_group}/"
                            f"workspaces/{target_ws.workspace_name}/"
                            f"datastores/{meta['ds']}/paths/{meta['path']}"
                        )

        if args.dry_run:
            log("would_import", name=target_name, version=version, path=target_path)
            continue
        data_asset = Data(
            name=target_name,
            version=version,
            path=target_path,
            description=manifest.get("description"),
            type=manifest.get("type", "mltable"),
            tags=manifest.get("tags"),
            properties=manifest.get("properties"),
        )
        target_client.data.create_or_update(data_asset)
        created += 1
        log("imported", name=target_name, version=version, original=name)
    log(
        "done_import",
        created=created,
        skipped=skipped,
        rewritten=rewritten,
        renamed_versions=renamed,
        dry_run=args.dry_run,
        copied_files=copied_files,
        copied_bytes=copied_bytes,
        rewritten_files=rewrite_files,
        rewritten_references=rewrite_references,
        rewritten_path_references=rewrite_path_references,
    )


if __name__ == "__main__":
    main()

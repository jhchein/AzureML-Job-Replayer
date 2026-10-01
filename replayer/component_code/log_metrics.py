import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
from azure.storage.blob import BlobClient, ContainerClient


_LOG_FOLDERS = ("logs/", "system_logs/", "user_logs/")


def _local_dest(rel_path: str) -> str:
    """Map a job-relative blob path to its path below ./outputs.

    outputs/<x> -> <x>; log folders -> original_logs/<folder>/...; others unchanged.
    """
    if rel_path.startswith("outputs/"):
        return rel_path[len("outputs/") :]
    if rel_path.startswith(_LOG_FOLDERS):
        return f"original_logs/{rel_path}"
    return rel_path


def _rbac_credential():
    """Credential for access mode 'rbac': the job's user identity if available."""
    try:
        from azure.ai.ml.identity import AzureMLOnBehalfOfCredential

        return AzureMLOnBehalfOfCredential()
    except Exception:  # noqa: BLE001
        from azure.identity import DefaultAzureCredential

        return DefaultAzureCredential()


def _manifest_blob_clients(manifest: Dict[str, Any]) -> List[Tuple[str, BlobClient]]:
    """Return (blob name, client) for every artifact blob listed in the manifest."""
    src = manifest.get("source", {})
    account, container = src.get("account"), src.get("container")
    prefix = (src.get("prefix") or "").strip("/")
    if not (account and container and prefix):
        raise RuntimeError("Artifact manifest missing source account/container/prefix.")
    account_url = f"https://{account}.blob.core.windows.net"

    if manifest.get("access", "sas") == "sas":
        # Per-blob read-only SAS tokens prepared by the replayer at build time.
        blobs = manifest.get("blobs")
        if blobs is None:
            raise RuntimeError("Artifact manifest has access=sas but no 'blobs' list.")
        return [
            (b["name"], BlobClient(account_url, container, b["name"], credential=b["sas"]))
            for b in blobs
        ]

    container_client = ContainerClient(
        account_url, container, credential=_rbac_credential()
    )
    result: List[Tuple[str, BlobClient]] = []
    for folder in manifest.get("relative_paths", []):
        folder_clean = folder.strip("/\\")
        for blob in container_client.list_blobs(
            name_starts_with=f"{prefix}/{folder_clean}/"
        ):
            result.append((blob.name, container_client.get_blob_client(blob.name)))
    return result


def _download_artifacts(manifest: Dict[str, Any]) -> None:
    """Download the manifest's blobs into ./outputs and write a summary file."""
    prefix = (manifest.get("source", {}).get("prefix") or "").strip("/")
    work_items = _manifest_blob_clients(manifest)
    print(f"Planned downloads: {len(work_items)} blob file(s)")

    base_outputs = Path("outputs")
    base_outputs.mkdir(exist_ok=True)
    base_resolved = base_outputs.resolve()
    start_time = time.time()
    success = 0
    failures: List[Dict[str, Any]] = []
    total_bytes = 0
    for blob_name, blob_client in work_items:
        rel_path = blob_name[len(prefix) + 1 :] if blob_name.startswith(f"{prefix}/") else blob_name
        local_path = base_outputs / _local_dest(rel_path)
        try:
            # Reject blob names that would escape ./outputs (e.g. via "..")
            if base_resolved not in local_path.resolve().parents:
                raise ValueError(f"Blob path escapes outputs directory: {rel_path}")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as lf:
                for chunk in blob_client.download_blob().chunks():
                    lf.write(chunk)
            total_bytes += os.path.getsize(local_path)
            success += 1
            if success % 50 == 0:
                print(f"Downloaded {success}/{len(work_items)} files (bytes={total_bytes})")
        except Exception as e:  # noqa: BLE001
            failures.append({"source": rel_path, "dest": str(local_path), "error": str(e)})
            if local_path.exists():
                local_path.unlink()

    elapsed = time.time() - start_time
    print(
        f"Artifact download summary: total={len(work_items)}"
        f" success={success} failed={len(failures)}"
        f" bytes={total_bytes} time_sec={elapsed:.2f}"
    )
    summary_path = base_outputs / "_replay_download_summary.json"
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(
            {
                "total": len(work_items),
                "success": success,
                "failed": len(failures),
                "failures": failures[:25],
                "bytes": total_bytes,
                "elapsed_sec": elapsed,
            },
            sf,
            indent=2,
        )
    if failures:
        print("First failure:", failures[0])
        raise RuntimeError("One or more artifact downloads failed; aborting replay step.")


def log_metrics(
    job_id: str,
    metrics_filepath: str,
    artifacts_dir: Optional[str] = None,
    artifact_manifest_path: Optional[str] = None,
    perform_server_copy: bool = True,
    copy_concurrency: int = 8,
) -> None:
    print(f"Replaying metrics for job: {job_id}")
    print(f"Reading metrics from file: {metrics_filepath}")

    metrics = {}
    try:
        # --- READ FROM FILE ---
        with open(metrics_filepath, "r") as f:
            metrics = json.load(f)  # Use json.load for file streams
        print(
            f"Successfully parsed metrics JSON from file. Found {len(metrics)} metrics."
        )
    except json.JSONDecodeError as e:
        print(f"Failed to parse metrics JSON from file '{metrics_filepath}': {e}")
        # Attempt to read content for debugging, handle potential read errors
        try:
            with open(metrics_filepath, "r") as f_err:
                content = f_err.read()
            print(
                f"File content received: {content[:500]}{'...' if len(content) > 500 else ''}"
            )
        except Exception as read_err:
            print(f"Could not read file content for debugging: {read_err}")
        return  # Exit if JSON is invalid
    except FileNotFoundError:
        print(f"ERROR: Metrics file not found at path: {metrics_filepath}")
        return
    except Exception as file_err:  # Catch other potential file errors
        print(f"ERROR: Could not read metrics file '{metrics_filepath}': {file_err}")
        return

    # --- In-run artifact download (local write to ./outputs) ---
    if artifact_manifest_path:
        try:
            with open(artifact_manifest_path, "r", encoding="utf-8") as mf:
                manifest = json.load(mf)
        except Exception as e:  # noqa: BLE001
            print(f"Artifact manifest load failed: {e}")
            manifest = None
        if perform_server_copy:
            if manifest is None:
                raise RuntimeError(
                    "Artifact copy requested but manifest could not be loaded inside replay step."
                )
            if manifest.get("disabled"):
                print(
                    "Artifact manifest is disabled — skipping artifact copy."
                    " Metrics will still be logged."
                )
                perform_server_copy = False
        if manifest and not manifest.get("disabled") and perform_server_copy:
            print("Starting artifact download into local ./outputs ...")
            _download_artifacts(manifest)
        elif manifest and manifest.get("disabled"):
            print("Artifact manifest disabled; skipping downloads.")
    else:
        print("No artifact manifest path provided; skipping server-side copy.")

    # --- Legacy local artifacts directory copy (optional) ---
    # NOTE: Keeping original logic; could be removed later.
    # --- Artifacts first (if provided) ---
    if artifacts_dir and os.path.isdir(artifacts_dir):
        artifacts_path = Path(artifacts_dir)
        # 1. Copy original 'outputs' subfolder (if exists) into ./outputs so AzureML auto-uploads them
        orig_outputs = artifacts_path / "outputs"
        dest_outputs = Path("outputs")
        if orig_outputs.is_dir():
            try:
                dest_outputs.mkdir(parents=True, exist_ok=True)
                # Copy files (shallow) then subdirs recursively
                for root, dirs, files in os.walk(orig_outputs):
                    rel_root = Path(root).relative_to(orig_outputs)
                    target_root = dest_outputs / rel_root
                    target_root.mkdir(parents=True, exist_ok=True)
                    for fn in files:
                        src_file = Path(root) / fn
                        tgt_file = target_root / fn
                        shutil.copy2(src_file, tgt_file)
                print(
                    "Copied original outputs to working ./outputs folder for AzureML promotion."
                )
            except Exception as e:
                print(f"WARNING: Failed to copy original outputs folder: {e}")
        else:
            print(
                "No original outputs folder found in artifacts; skipping copy to ./outputs"
            )

        # 2. Prepare a temp directory with renamed reserved folders to avoid conflicts
        RESERVED = {"logs", "system_logs", "user_logs"}
        # We'll create a staging folder sibling to artifacts_dir
        staging_root = Path("replay_artifacts_staging")
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        staging_root.mkdir(parents=True, exist_ok=True)

        copied_items = 0
        skipped_items = 0
        for item in artifacts_path.iterdir():
            target_name = item.name
            if item.name in RESERVED:
                target_name = f"original_{item.name}"
            # Avoid duplicating outputs: we've already copied original outputs to ./outputs for AzureML.
            if item.name == "outputs":
                print(
                    "Skipping original 'outputs' directory for MLflow upload to avoid duplication."
                )
                continue
            dest_path = staging_root / target_name
            try:
                if item.is_dir():
                    shutil.copytree(item, dest_path)
                else:
                    shutil.copy2(item, dest_path)
                copied_items += 1
            except Exception as e:
                print(f"WARNING: Failed to copy {item} -> {dest_path}: {e}")
                skipped_items += 1

        print(
            f"Staged artifacts for MLflow upload (copied={copied_items}, skipped={skipped_items}) into {staging_root}"  # noqa: E501
        )
        # 3. Upload under a safe prefix to avoid collisions
        try:
            print(
                "Uploading staged artifacts via MLflow (prefix=replayed_artifacts)..."
            )
            mlflow.log_artifacts(str(staging_root), artifact_path="replayed_artifacts")
            print("Staged artifacts upload completed.")
        except Exception as e:
            print(f"Failed to upload staged artifacts: {e}")
    else:
        if artifacts_dir:
            print(f"Artifacts directory not found or empty: {artifacts_dir}")

    # --- Metrics logging ---
    print("Attempting to log metrics to the current Azure ML job run.")
    try:
        if not metrics:
            print("No metrics found in the parsed data to log.")

        for key, value in metrics.items():
            try:
                metric_value = float(value)
                mlflow.log_metric(key, metric_value)
                print(f"Logged metric: {key} = {metric_value}")
            except (ValueError, TypeError) as e:
                print(
                    f"Failed to convert or log metric '{key}' with value '{value}': {e}"
                )

        # NOTE: Removed duplicate tag 'replayed_from_job' (original_job_id already logged at pipeline build level)
        # If needed in future, reintroduce behind a flag.
        mlflow.set_tag("original_job_id", job_id)
        print(f"Set tag 'original_job_id' = {job_id}")

    except Exception as e:
        print(f"An error occurred during MLflow logging: {e}")
        # raise # Optional: re-raise if logging failure is critical


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=str, required=True)
    parser.add_argument("--metrics-file", type=str, required=True)  # Expect file path
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        required=False,
        default=None,
        help="Optional local artifacts directory to upload via MLflow (legacy path).",
    )
    parser.add_argument(
        "--artifact-manifest",
        type=str,
        required=False,
        default=None,
        help="Path to artifact manifest JSON enabling in-run server-side copy into outputs/",
    )
    parser.add_argument(
        "--copy-artifacts",
        action="store_true",
        help="Perform server-side artifact copy described by the manifest into outputs/ of this run.",
        default=False,
    )
    parser.add_argument(
        "--copy-concurrency",
        type=int,
        default=8,
        help="(Reserved) Concurrency for future async copy batching (currently sequential).",
    )
    args = parser.parse_args()

    log_metrics(
        args.job_id,
        args.metrics_file,
        args.artifacts_dir,
        artifact_manifest_path=args.artifact_manifest,
        perform_server_copy=args.copy_artifacts,
        copy_concurrency=args.copy_concurrency,
    )
    print("Metrics & artifacts logging script finished.")
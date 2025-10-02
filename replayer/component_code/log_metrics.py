import argparse
import json
import os
import mlflow
import shutil
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from azure.storage.blob import BlobClient


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
        if manifest and not manifest.get("disabled") and perform_server_copy:
            print("Starting artifact download into local ./outputs ...")
            src_info = manifest.get("source", {})
            src_acct = src_info.get("account")
            src_container = src_info.get("container")
            src_prefix = (src_info.get("prefix") or "").strip("/")
            src_sas = src_info.get("sas")
            if not (src_acct and src_container and src_sas):
                print(
                    "Manifest missing required source fields (account/container/sas); skipping downloads."
                )
            else:
                src_list: List[str] = manifest.get("relative_paths", [])
                norm_list_raw = manifest.get("normalized_relative_paths")
                norm_list: Optional[List[str]] = (
                    list(norm_list_raw) if isinstance(norm_list_raw, list) else None
                )
                use_pairing = bool(norm_list and len(norm_list) == len(src_list))
                work_items: List[Tuple[str, Optional[str]]]
                if use_pairing:
                    assert norm_list is not None
                    work_items = list(zip(src_list, norm_list))
                else:
                    if norm_list:
                        print(
                            "Normalized list length mismatch; falling back to dynamic normalization."
                        )
                    work_items = [(p, None) for p in src_list]

                if not work_items:
                    print("No artifacts listed in manifest to download.")
                else:
                    print(
                        f"Planned downloads: {len(work_items)} (pairing={use_pairing})"
                    )
                    for s_rel, maybe_norm in work_items[:3]:
                        print(
                            f"  SAMPLE SRC='{s_rel}' -> DST='{maybe_norm or '<normalize>'}'"
                        )

                base_outputs = Path("outputs")
                base_outputs.mkdir(exist_ok=True)
                start_time = time.time()
                success = 0
                failures: List[Dict[str, Any]] = []
                total_bytes = 0
                for src_rel, maybe_norm in work_items:
                    rel_clean = src_rel.lstrip("/\\")
                    # destination path logic
                    if use_pairing and maybe_norm:
                        dest_rel = maybe_norm.lstrip("/\\")
                    else:
                        if rel_clean.startswith("outputs/"):
                            dest_rel = rel_clean[len("outputs/") :]
                        elif (
                            rel_clean.startswith("logs/")
                            or rel_clean.startswith("system_logs/")
                            or rel_clean.startswith("user_logs/")
                        ):
                            dest_rel = f"original_logs/{rel_clean}"
                        else:
                            dest_rel = rel_clean
                    # Basic path sanitization
                    dest_rel = dest_rel.replace("..", "__")
                    local_path = base_outputs / dest_rel
                    local_path.parent.mkdir(parents=True, exist_ok=True)

                    # Build source blob URL
                    source_url = f"https://{src_acct}.blob.core.windows.net/{src_container}/{src_prefix}/{rel_clean}".rstrip(
                        "/"
                    )
                    # Avoid double slashes except protocol
                    source_url = source_url.replace("//", "/").replace(":/", "://")
                    if "?" in source_url:
                        # Already contains query - unlikely here
                        full_url = f"{source_url}&{src_sas}"
                    else:
                        full_url = f"{source_url}?{src_sas}"
                    try:
                        blob_client = BlobClient.from_blob_url(full_url)
                        downloader = blob_client.download_blob()
                        with open(local_path, "wb") as lf:
                            for chunk in downloader.chunks():
                                lf.write(chunk)
                        size = os.path.getsize(local_path)
                        total_bytes += size
                        success += 1
                        if success % 50 == 0:
                            print(
                                f"Downloaded {success}/{len(work_items)} files (bytes={total_bytes})"
                            )
                    except Exception as e:  # noqa: BLE001
                        failures.append(
                            {
                                "source": rel_clean,
                                "dest": str(local_path),
                                "error": str(e),
                            }
                        )
                        # Remove partial file if exists
                        try:
                            if local_path.exists():
                                local_path.unlink()
                        except Exception:
                            pass

                elapsed = time.time() - start_time
                print(
                    f"Artifact download summary: total={len(work_items)} success={success} failed={len(failures)} bytes={total_bytes} time_sec={elapsed:.2f}"
                )
                if failures:
                    print("First failure:", failures[0])
                    if len(failures) < 6:
                        print("All failures:", failures)
                # Write summary file for visibility
                try:
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
                                "pairing": use_pairing,
                            },
                            sf,
                            indent=2,
                        )
                    print(
                        f"Wrote download summary to {summary_path} (will appear in Outputs + logs)."
                    )
                except Exception as se:  # noqa: BLE001
                    print(f"Failed to write summary file: {se}")
        else:
            if manifest and manifest.get("disabled"):
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

        mlflow.set_tag("replayed_from_job", job_id)
        print(f"Set tag 'replayed_from_job' = {job_id}")

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

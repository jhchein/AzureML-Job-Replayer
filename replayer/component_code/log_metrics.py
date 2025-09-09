import argparse
import json
import os
import mlflow
import shutil
from pathlib import Path


from typing import Optional


def log_metrics(
    job_id: str, metrics_filepath: str, artifacts_dir: Optional[str] = None
):  # Accept file path
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
        help="Optional artifacts directory to upload before metrics are logged.",
    )
    args = parser.parse_args()

    log_metrics(args.job_id, args.metrics_file, args.artifacts_dir)
    print("Metrics & artifacts logging script finished.")

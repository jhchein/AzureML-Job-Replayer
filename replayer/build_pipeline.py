import argparse
import json
import logging
import os
import shutil
import tempfile
import uuid
from typing import Dict, List, Optional, Tuple

from azure.ai.ml import (
    Input,
)
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    CommandJob,
    PipelineJob,
    PipelineJobSettings,
    UserIdentityConfiguration,
)
from azure.core.exceptions import HttpResponseError

from extractor.extract_jobs import JobMetadata
from replayer.dummy_components import (
    REGISTERED_ENV_ID,
    replay_metrics_component,
)
from utils.aml_clients import get_ml_client
from utils.log_setup import setup_logging

setup_logging(
    log_filename_prefix=(
        "extractor"
        if __name__ == "__main__" and "extractor" in __file__
        else "replayer"
    )
)


logger = logging.getLogger(__name__)

# --- Constants ---
REPLAY_PIPELINE_TAG = "replay_pipeline"
REPLAY_STANDALONE_TAG = "replay_standalone"
ORIGINAL_JOB_ID_TAG = "original_job_id"
REPLAY_TYPE_TAG = "replay_type"
DUMMY_REPLAY_VALUE = "dummy_metrics_replay"


def get_parent_and_children(
    jobs_in_group: List[JobMetadata],
) -> Tuple[Optional[JobMetadata], List[JobMetadata]]:
    """Separates the parent job from child jobs in a list."""
    parent = None
    children = []
    parent_job_name = jobs_in_group[
        0
    ].parent_job_name  # Assume first one tells us if it's a child group

    if (
        parent_job_name is None
    ):  # This group represents a standalone job OR a parent itself
        if len(jobs_in_group) == 1:  # Likely standalone
            return jobs_in_group[0], []
        else:  # It's a group containing the parent AND children, find the parent
            potential_parent_name = jobs_in_group[
                0
            ].name  # Key used for grouping was parent name
            for job in jobs_in_group:
                if job.name == potential_parent_name and job.parent_job_name is None:
                    parent = job
                else:
                    children.append(job)
            if parent:
                return parent, children
            else:
                # Should not happen with current grouping, but handle defensively
                print(
                    f"Warning: Could not definitively identify parent in group for key {potential_parent_name}. Treating all as children."
                )
                return None, jobs_in_group
    else:  # This group contains only children of a parent processed elsewhere
        # Or the grouping logic needs refinement if parents are included here too.
        # Assuming the grouping key ensures this list is *only* children or *only* standalone.
        # If the key is parent_job_name, this list *should* be only children.
        # Let's refine the grouping logic later if needed.
        # For now, assume this case doesn't happen if grouping is correct.
        print(
            f"Warning: Unexpected grouping for key {parent_job_name}. Expected standalone or parent+children."
        )
        return None, jobs_in_group  # Treat as children?


def build_dummy_pipeline_for_children(
    parent_job: JobMetadata,
    child_jobs: List[JobMetadata],
    temp_dir_path: str,
) -> Optional[PipelineJob]:
    if not child_jobs:
        return None

    pipeline_steps_dict = {}
    # Use a context manager for temp dir if creating many files
    # with tempfile.TemporaryDirectory() as temp_dir:
    # print(f"DEBUG: Using temp dir for metrics files: {temp_dir}")
    for child_job in child_jobs:
        metrics_to_log = child_job.mlflow_metrics or {}

        # --- Create temp JSON file ---
        metrics_json_str = json.dumps(metrics_to_log)
        # Create a unique filename
        temp_filename = f"metrics_{uuid.uuid4()}.json"
        temp_filepath = os.path.join(temp_dir_path, temp_filename)
        try:
            with open(temp_filepath, "w") as f:
                f.write(metrics_json_str)
            # print(
            #     f"DEBUG [Pipeline Step {child_job.name}]: Created temp metrics file: {temp_filepath}"
            # )
        except IOError as e:
            print(f"ERROR: Failed to write temp metrics file {temp_filepath}: {e}")
            # TODO: Consider how to handle this - skip step? Fail pipeline build?
            # For now, let's skip the step
            continue  # Skip this step if file cannot be created

        # --- Call component, passing FILE PATH via Input ---
        step = replay_metrics_component(
            original_job_id=child_job.name,
            # Pass an Input object pointing to the temp file
            metrics_file=Input(type=AssetTypes.URI_FILE, path=temp_filepath),
        )

        # step.compute = "serverless" # (compute defaults to pipeline setting)
        sanitized_step_key = "".join(
            c if c.isalnum() or c in ["-", "_"] else "_"
            for c in (child_job.display_name or child_job.name)[:50]
        )
        count = 1
        final_key = sanitized_step_key
        while final_key in pipeline_steps_dict:
            final_key = f"{sanitized_step_key}_{count}"
            count += 1
        step.name = final_key
        step.display_name = f"replay_{child_job.display_name or child_job.name}"
        step.tags = child_job.tags or {}
        step.tags[ORIGINAL_JOB_ID_TAG] = child_job.name
        step.tags["original_parent_job_id"] = parent_job.name

        pipeline_steps_dict[final_key] = step

    # --- Construct PipelineJob (outside temp dir context is fine) ---
    if not pipeline_steps_dict:  # Check if any steps were actually added
        print("Warning: No steps added to pipeline, possibly due to file errors.")
        return None

    pipeline_job_object = PipelineJob(
        jobs=pipeline_steps_dict,
        settings=PipelineJobSettings(default_compute="serverless"),
        display_name=f"Replay of {parent_job.display_name or parent_job.name}",
        description=f"Dummy replay of pipeline {parent_job.name}. Original Desc: {parent_job.description or ''}",
        tags={
            ORIGINAL_JOB_ID_TAG: parent_job.name,
            REPLAY_TYPE_TAG: DUMMY_REPLAY_VALUE,
            **(parent_job.tags or {}),
        },
        experiment_name=parent_job.experiment_name or "replayed_jobs",
        identity=UserIdentityConfiguration(),
    )
    return pipeline_job_object


def build_dummy_standalone_job(
    original_job: JobMetadata,
    metrics_file_path: str,
) -> CommandJob:
    """
    Builds a dummy CommandJob using a pre-existing metrics file path.
    """
    # Metrics file is already created by the caller

    job: CommandJob = replay_metrics_component(
        original_job_id=original_job.name,
        metrics_file=Input(type=AssetTypes.URI_FILE, path=metrics_file_path),
    )

    # Apply runtime settings
    job.display_name = f"Replay of {original_job.display_name or original_job.name}"
    job.description = f"Dummy replay of job {original_job.name}. Original Desc: {original_job.description or ''}"
    job.tags = {
        ORIGINAL_JOB_ID_TAG: original_job.name,
        REPLAY_TYPE_TAG: DUMMY_REPLAY_VALUE,
        REPLAY_STANDALONE_TAG: "true",
        **(original_job.tags or {}),
    }
    job.experiment_name = original_job.experiment_name or "replayed_jobs"
    job.environment = REGISTERED_ENV_ID
    job.identity = UserIdentityConfiguration()

    return job


# --- Main execution logic ---


def main(args):
    print("Loading job metadata...")
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            jobs_raw = json.load(f)
            all_jobs_metadata = [JobMetadata(**j) for j in jobs_raw]
        print(f"Loaded {len(all_jobs_metadata)} job metadata records.")
    except Exception as e:
        print(f"Error loading or parsing {args.input}: {e}")
        # Optional: Log exception details if logging is configured
        exit(1)

    # Group jobs by ORIGINAL parent_job_name (or own name if no parent)
    original_execution_units: Dict[str, List[JobMetadata]] = {}
    for job in all_jobs_metadata:
        grouping_key = job.parent_job_name if job.parent_job_name else job.name
        original_execution_units.setdefault(grouping_key, []).append(job)

    print(
        f"Grouped into {len(original_execution_units)} original execution units (pipelines/standalone jobs)."
    )

    # --- Connect to Client and Register Environment ---
    registered_env_id_for_jobs = None
    try:
        client = get_ml_client(args.target)
        print(f"Connected to target workspace: {client.workspace_name}")

        # Register the environment only if not doing a dry run
        if not args.dry_run:
            from replayer.dummy_components import DUMMY_ENV, REGISTERED_ENV_ID

            print(
                f"Ensuring dummy environment '{DUMMY_ENV.name}:{DUMMY_ENV.version}' exists..."
            )
            try:
                env_reg = client.environments.create_or_update(DUMMY_ENV)
                print(f" -> Environment '{env_reg.name}:{env_reg.version}' is ready.")
                registered_env_id_for_jobs = REGISTERED_ENV_ID
            except Exception as e:
                print(f"❌ Error registering/updating dummy environment: {e}")
                print("Cannot proceed without the registered environment. Exiting.")
                exit(1)
        else:
            from replayer.dummy_components import REGISTERED_ENV_ID

            registered_env_id_for_jobs = REGISTERED_ENV_ID
            print("Dry run: Skipping environment registration.")

    except Exception as e:
        print(f"Error connecting to target workspace: {e}")
        # Optional: Log exception details if logging is configured
        exit(1)

    # --- Process and Submit/Dry-Run Jobs ---
    submitted_count = 0
    failed_count = 0
    skipped_count = 0
    processed_count = 0
    job_map = {}  # original_id -> new_job_name

    # Ensure environment ID was set (should be unless dry run skipped it, which is ok for dry run)
    if not registered_env_id_for_jobs and not args.dry_run:
        print("Error: Environment ID was not set after registration attempt. Exiting.")
        exit(1)

    for grouping_key, jobs_in_group in original_execution_units.items():
        if args.limit is not None and processed_count >= args.limit:
            print(f"\nReached processing limit ({args.limit}). Stopping.")
            break
        processed_count += 1
        print(
            f"\nProcessing original unit {processed_count}/{len(original_execution_units) if args.limit is None else args.limit}: {grouping_key} ({len(jobs_in_group)} records)"
        )

        is_standalone = (
            all(j.parent_job_name is None for j in jobs_in_group)
            and len(jobs_in_group) == 1
        )
        is_pipeline_group = any(
            j.parent_job_name == grouping_key
            for j in jobs_in_group
            if j.parent_job_name is not None
        ) or (
            len(jobs_in_group) > 1
            and any(
                j.name == grouping_key and j.parent_job_name is None
                for j in jobs_in_group
            )
        )

        job_to_submit: Optional[CommandJob | PipelineJob] = None
        original_identifier = grouping_key
        temp_metrics_filepath_standalone = None  # For standalone file path
        temp_pipeline_dir_path = None  # For pipeline

        try:
            try:
                if is_standalone:
                    original_job_metadata = jobs_in_group[0]
                    original_identifier = original_job_metadata.name
                    print(
                        f" -> Identified as Standalone Job: {original_job_metadata.name}"
                    )

                    # --- Create temp file BEFORE build ---
                    metrics_to_log = original_job_metadata.mlflow_metrics or {}
                    metrics_json_str = json.dumps(metrics_to_log)
                    try:
                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=".json", delete=False, encoding="utf-8"
                        ) as temp_f:
                            temp_metrics_filepath_standalone = temp_f.name
                            temp_f.write(metrics_json_str)
                    except IOError as e:
                        raise IOError(
                            f"Failed to write temp metrics file for standalone job {original_identifier}: {e}"
                        )  # Re-raise

                    # --- Pass file path to build function ---
                    job_to_submit = build_dummy_standalone_job(
                        original_job_metadata, temp_metrics_filepath_standalone
                    )

                elif is_pipeline_group:
                    parent_meta, children_meta = get_parent_and_children(jobs_in_group)
                    if parent_meta and children_meta:
                        original_identifier = parent_meta.name
                        print(
                            f" -> Identified as Pipeline Job: {original_identifier} ({len(children_meta)} children)"
                        )

                        # --- Create persistent temp directory BEFORE build ---
                        try:
                            temp_pipeline_dir_path = tempfile.mkdtemp()
                            # logger.debug(f"Pipeline {original_identifier}: Created temp directory: {temp_pipeline_dir_path}")
                        except OSError as e:
                            raise OSError(
                                f"Failed to create temp directory for pipeline {original_identifier}: {e}"
                            )  # Re-raise

                        # --- Pass directory path to builder ---
                        job_to_submit = build_dummy_pipeline_for_children(
                            parent_meta,
                            children_meta,
                            temp_pipeline_dir_path,  # Pass dir path
                        )
                    elif parent_meta and not children_meta:
                        # Treat as standalone (uses temp file, not dir)
                        original_identifier = parent_meta.name
                        print(
                            f" -> Identified as Pipeline Job with NO children: {original_identifier}. Replaying as standalone."
                        )
                        metrics_to_log = parent_meta.mlflow_metrics or {}
                        metrics_json_str = json.dumps(metrics_to_log)
                        try:
                            with tempfile.NamedTemporaryFile(
                                mode="w", suffix=".json", delete=False, encoding="utf-8"
                            ) as temp_f:
                                temp_metrics_filepath_standalone = temp_f.name
                                temp_f.write(metrics_json_str)
                            # logger.debug(f"Standalone Pipeline {original_identifier}: Created temp metrics file: {temp_metrics_filepath_standalone}")
                        except IOError as e:
                            raise IOError(
                                f"Failed to write temp metrics file for pipeline-as-standalone job {original_identifier}: {e}"
                            )
                        job_to_submit = build_dummy_standalone_job(
                            parent_meta, temp_metrics_filepath_standalone
                        )
                    else:
                        print(
                            f" -> Warning: Could not determine parent/child structure for pipeline group {grouping_key}. Skipping."
                        )
                        skipped_count += 1
                        continue
                else:
                    print(
                        f" -> Warning: Ambiguous structure or invalid group for key {grouping_key}. Skipping."
                    )
                    skipped_count += 1
                    continue

            except Exception as build_error:
                print(
                    f"   ❌ Error building job object for original {original_identifier}: {build_error}"
                )
                failed_count += 1
                # Note: No submission attempt, so jump straight to finally for cleanup
                raise  # Re-raise to trigger the outer finally block for cleanup

            # Submit the job or perform dry run, THEN clean up
            if job_to_submit:
                if args.dry_run:
                    print(f"\n--- DRY RUN for Original Unit: {original_identifier} ---")
                    local_path_info = "N/A"
                    if temp_metrics_filepath_standalone:
                        local_path_info = temp_metrics_filepath_standalone
                    elif temp_pipeline_dir_path:
                        local_path_info = f"(Files in {temp_pipeline_dir_path})"
                    print(f"  Metrics File Input Path(s) (LOCAL): {local_path_info}")
                    print(f"  Would submit Job Type: {type(job_to_submit).__name__}")
                    print(f"  Display Name: {job_to_submit.display_name}")
                    print(f"  Experiment Name: {job_to_submit.experiment_name}")
                    print(f"  Tags: {job_to_submit.tags}")
                    # Add environment check for dry run
                    env_id = None
                    if isinstance(job_to_submit, CommandJob):
                        env_id = job_to_submit.environment
                    elif isinstance(job_to_submit, PipelineJob) and job_to_submit.jobs:
                        # Check env on first step as representative
                        first_step_key = next(iter(job_to_submit.jobs))
                        env_id = job_to_submit.jobs[first_step_key].environment
                    print(f"  Environment: {env_id}")

                    if isinstance(job_to_submit, PipelineJob):
                        print(f"  Pipeline Steps ({len(job_to_submit.jobs)}):")
                        for step_name, step_job in job_to_submit.jobs.items():
                            comp_name = "N/A"
                            if hasattr(step_job, "component"):
                                if isinstance(step_job.component, str):
                                    comp_name = step_job.component
                                elif hasattr(step_job.component, "name"):
                                    comp_name = step_job.component.name
                            print(
                                f"    - Step: {step_name} (Display: {step_job.display_name}, Comp: {comp_name}, Compute: {step_job.compute}, Tags: {step_job.tags})"
                            )
                    elif isinstance(job_to_submit, CommandJob):
                        print(f"  Command: {job_to_submit.command}")
                        print(f"  Compute: {job_to_submit.compute}")
                    print("--- END DRY RUN ---")
                    submitted_count += 1
                    job_map[original_identifier] = (
                        f"<Dry Run - {type(job_to_submit).__name__}>"
                    )
                else:
                    # --- Actual Submission ---
                    try:
                        print("   Submitting replay job/pipeline...")
                        created_job = client.jobs.create_or_update(job_to_submit)
                        print(
                            f"   ✔ Submitted: {created_job.name} (Type: {created_job.type}) for original: {original_identifier}"
                        )
                        submitted_count += 1
                        job_map[original_identifier] = created_job.name
                    except HttpResponseError as http_err:
                        print(
                            f"\n   ❌ HTTP Error submitting job for original {original_identifier}: Status Code {http_err.status_code}"
                        )
                        print(f"      Reason: {http_err.reason}")
                        # Attempt to extract more details from the error response body
                        error_details = "No additional details found in error object."
                        if http_err.error:
                            # Attempt to access common error structures
                            code = getattr(http_err.error, "code", "N/A")
                            message = getattr(
                                http_err.error, "message", str(http_err)
                            )  # Fallback to basic message
                            error_details = f"Service Error Code: {code}\n      Service Message: {message}"
                        elif hasattr(http_err, "message"):
                            # Sometimes the message attribute has the core info
                            error_details = f"Error Message: {http_err.message}"

                        print(f"      {error_details}")
                        # Log the full error object to the debug log file for deeper inspection
                        # logger.debug(f"Full HttpResponseError object for {original_identifier}:", exc_info=True) # exc_info adds traceback
                        # Or just log the raw response if available
                        if hasattr(http_err, "response") and hasattr(
                            http_err.response, "text"
                        ):
                            print(
                                "      (Check log file for full response body if DEBUG logging is enabled)"
                            )
                            # logger.debug(f"Raw error response body for {original_identifier}: {http_err.response.text}")

                        print("      Recommendations:")
                        print(
                            "        - Check the Azure Portal Activity Log for the target resource group around this time."
                        )
                        print(
                            "        - Verify Azure Policies assigned to the target scope (workspace, RG, subscription)."
                        )
                        print(
                            "        - Double-check for any Deny Assignments in IAM for the target scope."
                        )
                        print(
                            "        - Ensure network connectivity if target workspace has network restrictions."
                        )

                        failed_count += 1
                    except Exception as submit_error:
                        print(
                            f"\n   ❌ Unexpected Error submitting job for original {original_identifier}: {submit_error}"
                        )
                        failed_count += 1
            else:
                print(
                    f"   No valid job object was generated for {original_identifier}. Skipping submission."
                )

        finally:
            # --- Clean up temporary files/directories AFTER attempt ---
            if temp_metrics_filepath_standalone and os.path.exists(
                temp_metrics_filepath_standalone
            ):
                try:
                    os.remove(temp_metrics_filepath_standalone)
                    # logger.debug(f"Cleaned up temp file: {temp_metrics_filepath_standalone}")
                except OSError as e:
                    print(
                        f"Warning: Failed to delete temp file {temp_metrics_filepath_standalone}: {e}"
                    )
            if temp_pipeline_dir_path and os.path.isdir(
                temp_pipeline_dir_path
            ):  # Check if it's a directory
                try:
                    shutil.rmtree(
                        temp_pipeline_dir_path
                    )  # Use shutil.rmtree for directories
                    # logger.debug(f"Cleaned up temp directory: {temp_pipeline_dir_path}")
                except OSError as e:
                    print(
                        f"Warning: Failed to delete temp directory {temp_pipeline_dir_path}: {e}"
                    )

    # --- Final Summary ---
    print("\n--- Replay Summary ---")
    print(f"Total original units found: {len(original_execution_units)}")
    print(f"Units processed (due to limit or completion): {processed_count}")
    if args.dry_run:
        print(f"Successfully processed (Dry Run): {submitted_count}")
    else:
        print(f"Successfully submitted replay jobs/pipelines: {submitted_count}")
    print(f"Failed submissions or builds: {failed_count}")
    print(f"Skipped due to structure issues: {skipped_count}")
    print("Original ID -> Replay Job Name/Status Mapping:")
    if job_map:
        for original, replay in job_map.items():
            print(f" - {original} -> {replay}")
    elif submitted_count > 0 and args.dry_run:
        print(" (See dry run output above for details)")
    else:
        print(" (No jobs submitted or processed)")
    print("----------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and submit replay jobs/pipelines"
    )
    parser.add_argument("--target", required=True, help="Path to target config JSON")
    parser.add_argument("--input", required=True, help="Path to extracted jobs.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",  # Makes it a flag, True if present
        help="Perform parsing and build job objects, but do not submit to Azure ML.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,  # Default is no limit
        help="Limit the number of original execution units (pipelines/standalone jobs) to process for testing.",
    )
    args = parser.parse_args()

    main(args)

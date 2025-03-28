# %load_ext autoreload
# %autoreload 2
import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

import mlflow
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Job
from azure.core.exceptions import ResourceNotFoundError
from dateutil import parser as date_parser
from mlflow.entities import Run as MlflowRun
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from tqdm import tqdm

from utils.aml_clients import get_ml_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---


def safe_isoformat(dt_obj: Any) -> Optional[str]:
    """Safely convert a datetime object or timestamp string/ms to ISO format string."""
    if dt_obj is None:
        return None
    try:
        if isinstance(dt_obj, (int, float)):  # Assume milliseconds if large number
            if dt_obj > 1e12:  # Simple heuristic for milliseconds vs seconds
                dt_obj = datetime.fromtimestamp(dt_obj / 1000, tz=timezone.utc)
            else:  # Assume seconds
                dt_obj = datetime.fromtimestamp(dt_obj, tz=timezone.utc)
        elif isinstance(dt_obj, str):
            dt_obj = date_parser.parse(dt_obj)

        if isinstance(dt_obj, datetime):
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(
                    tzinfo=timezone.utc
                )  # Assume UTC if no timezone
            return dt_obj.isoformat()
        else:
            logger.warning(
                f"Cannot convert non-datetime object to ISO format: {dt_obj} (Type: {type(dt_obj)})"
            )
            return str(dt_obj)  # Fallback to string representation
    except (ValueError, TypeError, OverflowError) as e:
        logger.warning(f"Failed to convert timestamp {dt_obj} to ISO format: {e}")
        return str(dt_obj)  # Fallback


def safe_duration(start: Any, end: Any) -> Optional[float]:
    """Safely calculate duration in seconds between two timestamp representations."""
    if start is None or end is None:
        return None
    try:
        start_dt = None
        end_dt = None

        # Convert start time
        if isinstance(start, (int, float)):
            start_dt = datetime.fromtimestamp(
                start / 1000 if start > 1e12 else start, tz=timezone.utc
            )
        elif isinstance(start, str):
            start_dt = date_parser.parse(start)
        elif isinstance(start, datetime):
            start_dt = start

        # Convert end time
        if isinstance(end, (int, float)):
            end_dt = datetime.fromtimestamp(
                end / 1000 if end > 1e12 else end, tz=timezone.utc
            )
        elif isinstance(end, str):
            end_dt = date_parser.parse(end)
        elif isinstance(end, datetime):
            end_dt = end

        if start_dt and end_dt:
            # Ensure timezone awareness for comparison
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            # Ensure end time is after start time
            if end_dt >= start_dt:
                return (end_dt - start_dt).total_seconds()
            else:
                logger.warning(
                    f"End time {end_dt} is before start time {start_dt}. Cannot calculate duration."
                )
                return None
        else:
            logger.warning(
                f"Could not parse start ({start}) or end ({end}) time for duration calculation."
            )
            return None

    except (ValueError, TypeError, OverflowError) as e:
        logger.warning(f"Failed to calculate duration between {start} and {end}: {e}")
        return None


def convert_complex_dict(obj: Any) -> Any:
    """Recursively convert complex objects (like SDK entities) within dicts/lists to basic types."""
    if isinstance(obj, dict):
        return {k: convert_complex_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_complex_dict(item) for item in obj]
    elif hasattr(obj, "_to_dict"):  # Handle Azure SDK entities with _to_dict method
        try:
            return convert_complex_dict(obj._to_dict())
        except Exception as e:
            logger.debug(
                f"Could not convert object with _to_dict using method: {e}. Falling back."
            )
            pass  # Fall through to other methods
    elif hasattr(obj, "asdict"):  # Handle dataclasses
        try:
            return convert_complex_dict(asdict(obj))
        except Exception as e:
            logger.debug(
                f"Could not convert object with asdict using method: {e}. Falling back."
            )
            pass  # Fall through to other methods
    elif hasattr(obj, "__dict__"):  # Generic objects
        try:
            # Filter out private/protected attributes and methods
            public_attrs = {
                k: v
                for k, v in vars(obj).items()
                if not k.startswith("_") and not callable(v)
            }
            return convert_complex_dict(public_attrs)
        except Exception as e:
            logger.debug(f"Could not convert object using __dict__: {e}. Falling back.")
            pass
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Fallback for other types (timedelta, etc.)
        return str(obj)


# --- Dataclass Definition ---


@dataclass
class JobMetadata:
    # Core Identifiers
    name: str  # From job.name (also run_id in many cases)
    id: Optional[str] = None  # From job.id (full ARM ID)
    display_name: Optional[str] = None  # From job.display_name
    job_type: Optional[str] = None  # From job.type
    status: Optional[str] = None  # From job.status

    # Hierarchy and Context
    experiment_name: Optional[str] = None  # From job.experiment_name
    parent_job_name: Optional[str] = None  # From job.parent_job_name

    # Timestamps and Duration
    created_at: Optional[str] = (
        None  # From job.creation_context.created_at (ISO format)
    )
    start_time: Optional[str] = (
        None  # From MLflow run.info or job.properties (ISO format)
    )
    end_time: Optional[str] = (
        None  # From MLflow run.info or job.properties (ISO format)
    )
    duration_seconds: Optional[float] = None  # Calculated
    # compute_duration_seconds: Optional[float] = None # Hard to get reliably

    # Creator Info
    created_by: Optional[str] = None  # From job.creation_context.created_by
    created_by_type: Optional[str] = None  # From job.creation_context.created_by_type

    # Modification Info
    last_modified_at: Optional[str] = None  # From job.creation_context.last_modified_at
    last_modified_by: Optional[str] = None  # From job.creation_context.last_modified_by
    last_modified_by_type: Optional[str] = (
        None  # From job.creation_context.last_modified_by_type
    )

    # Execution Details
    command: Optional[str] = None  # From job.command (for CommandJobs)
    script_name: Optional[str] = None  # From MLflow tags (mlflow.source.name)
    environment_name: Optional[str] = None  # From job.environment (name or string)
    environment_id: Optional[str] = None  # From job.environment (ARM ID if specified)
    environment_variables: Optional[Dict[str, str]] = (
        None  # From job.environment_variables
    )
    code_id: Optional[str] = None  # From job.code (ARM ID or path string)
    arguments: Optional[Dict[str, Any]] = (
        None  # From job.inputs (Literal inputs might represent args) - Alias? Let's keep command separate.
    )
    # job.parameters might be relevant for CommandJobs
    job_parameters: Optional[Dict[str, Any]] = (
        None  # From job.parameters (specific to CommandJob in REST)
    )

    # Compute Details
    compute_target: Optional[str] = None  # From job.compute (name)
    compute_id: Optional[str] = (
        None  # From job.computeId (ARM ID in REST) - SDK uses job.compute? Check. -> job.compute gives name. Need properties?
    )
    # REST properties.computeId exists. Let's try getattr(job, 'computeId', None)
    compute_type: Optional[str] = (
        None  # From job.properties['_azureml.ComputeTargetType']
    )
    instance_count: Optional[int] = None  # From job.resources.instance_count
    instance_type: Optional[str] = None  # From job.resources.instance_type

    # Job Specific Configuration
    distribution: Optional[Dict[str, Any]] = None  # From job.distribution
    job_limits: Optional[Dict[str, Any]] = None  # From job.limits
    job_inputs: Optional[Dict[str, Any]] = None  # From job.inputs
    job_outputs: Optional[Dict[str, Any]] = None  # From job.outputs
    identity_type: Optional[str] = None  # From job.identity.type
    services: Optional[Dict[str, Any]] = None  # From job.services
    job_properties: Optional[Dict[str, Any]] = (
        None  # From job.properties (generic key-value)
    )

    # AutoML/Sweep Specific
    task_details: Optional[Dict[str, Any]] = None  # From job.task_details (AutoML)
    objective: Optional[Dict[str, Any]] = None  # From job.objective (Sweep)
    search_space: Optional[Dict[str, Any]] = None  # From job.search_space (Sweep)
    sampling_algorithm: Optional[Dict[str, Any]] = (
        None  # From job.sampling_algorithm (Sweep)
    )
    early_termination: Optional[Dict[str, Any]] = (
        None  # From job.early_termination (Sweep)
    )
    trial_component: Optional[Dict[str, Any]] = None  # From job.trial (Sweep)

    # Pipeline Specific
    pipeline_settings: Optional[Dict[str, Any]] = None  # From job.settings (Pipeline)
    pipeline_sub_jobs: Optional[Dict[str, Any]] = None  # From job.jobs (Pipeline)

    # Metadata
    description: Optional[str] = None  # From job.description
    tags: Optional[Dict[str, str]] = None  # From job.tags

    # MLflow Specific Data
    mlflow_run_id: Optional[str] = None  # From run.info.run_id
    mlflow_run_name: Optional[str] = None  # From run.info.run_name
    mlflow_experiment_id: Optional[str] = None  # From run.info.experiment_id
    mlflow_user_id: Optional[str] = None  # From run.info.user_id (Often empty)
    mlflow_artifact_uri: Optional[str] = None  # From run.info.artifact_uri
    mlflow_metrics: Optional[Dict[str, float]] = None  # From run.data.metrics
    mlflow_params: Optional[Dict[str, str]] = None  # From run.data.params
    mlflow_tags: Optional[Dict[str, str]] = None  # From run.data.tags
    mlflow_dataset_inputs: Optional[List[Any]] = None  # From run.inputs.dataset_inputs

    def to_dict(self):
        """Convert dataclass to dictionary, handling complex types."""
        d = {}
        for f in fields(self):
            value = getattr(self, f.name)
            d[f.name] = convert_complex_dict(value)
        return d


# --- Extraction Logic ---


def extract_all_jobs(client: MLClient) -> Iterator[JobMetadata]:
    """Extract all jobs from the AzureML workspace and yield them one by one."""
    mlflow_tracking_uri = client.workspaces.get(
        client.workspace_name
    ).mlflow_tracking_uri

    logger.info(f"Using MLflow tracking URI: {mlflow_tracking_uri}")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow_client = MlflowClient()

    all_job_summaries = list(client.jobs.list())  # Get all summaries first for tqdm
    logger.info(f"Found {len(all_job_summaries)} total job summaries.")

    for job_summary in tqdm(all_job_summaries, desc="Processing Jobs"):
        job_name = job_summary.name
        # logger.info(f"Processing job: {job_name}") # Reduce verbosity with tqdm

        try:
            # Use the specific job name (which is often the run_id) to get the job
            job: Job = client.jobs.get(name=job_name)
            # logger.debug(f"Successfully retrieved Job object for {job_name}")
        except ResourceNotFoundError:
            logger.warning(f"Job {job_name} not found via client.jobs.get(). Skipping.")
            continue
        except Exception as e:
            logger.error(
                f"Failed to retrieve job {job_name} via client.jobs.get(): {e}"
            )
            continue

        # --- MLflow Data Extraction ---
        mlflow_run: Optional[MlflowRun] = None
        mlflow_metrics = {}
        mlflow_params = {}
        mlflow_tags_dict = {}
        mlflow_dataset_inputs = None
        mlflow_run_id = None
        mlflow_run_name = None
        mlflow_experiment_id = None
        mlflow_user_id = None
        mlflow_artifact_uri = None
        mlflow_start_time_ms = None
        mlflow_end_time_ms = None
        script_name = None

        try:
            # Use job_name which often corresponds to run_id
            mlflow_run = mlflow_client.get_run(job_name)
            if mlflow_run:
                # logger.debug(f"Found MLflow run for job {job_name}")
                if mlflow_run.data:
                    mlflow_metrics = mlflow_run.data.metrics or {}
                    mlflow_params = mlflow_run.data.params or {}
                    mlflow_tags_dict = mlflow_run.data.tags or {}
                    script_name = mlflow_tags_dict.get("mlflow.source.name")
                    # logger.info(f"Retrieved {len(mlflow_metrics)} metrics for job {job_name}")
                if mlflow_run.info:
                    mlflow_run_id = mlflow_run.info.run_id
                    mlflow_run_name = mlflow_run.info.run_name
                    mlflow_experiment_id = mlflow_run.info.experiment_id
                    mlflow_user_id = mlflow_run.info.user_id
                    mlflow_artifact_uri = mlflow_run.info.artifact_uri
                    mlflow_start_time_ms = mlflow_run.info.start_time
                    mlflow_end_time_ms = mlflow_run.info.end_time
                if hasattr(mlflow_run, "inputs") and hasattr(
                    mlflow_run.inputs, "dataset_inputs"
                ):
                    mlflow_dataset_inputs = mlflow_run.inputs.dataset_inputs

        except MlflowException as e:
            # This is common if the job wasn't logged to MLflow (e.g., some system jobs)
            if (
                "RESOURCE_DOES_NOT_EXIST" in str(e)
                or "Run with id=" in str(e)
                and "not found" in str(e)
            ):
                logger.debug(
                    f"No MLflow run found for job {job_name}. This might be expected."
                )
            else:
                logger.warning(f"MLflow Error retrieving run for job {job_name}: {e}")
        except Exception as e:
            logger.warning(
                f"Unexpected Error retrieving MLflow run for job {job_name}: {e}"
            )

        # --- Job Object Data Extraction ---
        job_id = getattr(job, "id", None)
        display_name = getattr(job, "display_name", job.name)  # Fallback to name
        job_type = getattr(job, "type", None)
        status = getattr(job, "status", None)
        experiment_name = getattr(job, "experiment_name", None)
        parent_job_name = getattr(job, "parent_job_name", None)
        description = getattr(job, "description", None)
        tags = getattr(job, "tags", {})
        command = getattr(job, "command", None)  # Specific to CommandJob primarily
        job_parameters = getattr(job, "parameters", None)  # Specific to CommandJob
        environment_variables = getattr(job, "environment_variables", None)
        code_obj = getattr(job, "code", None)
        code_id = (
            str(code_obj) if code_obj else None
        )  # Store as string (can be path or ARM ID)
        compute_target_name = getattr(job, "compute", None)
        job_inputs = getattr(job, "inputs", None)
        job_outputs = getattr(job, "outputs", None)
        identity_obj = getattr(job, "identity", None)
        identity_type = getattr(identity_obj, "type", None) if identity_obj else None
        services = getattr(job, "services", None)
        job_properties = getattr(job, "properties", None)  # Generic properties dict

        # Specific Job Type Fields
        distribution = getattr(job, "distribution", None)
        job_limits = getattr(job, "limits", None)
        task_details = getattr(job, "task_details", None)  # AutoML
        objective = getattr(job, "objective", None)  # Sweep
        search_space = getattr(job, "search_space", None)  # Sweep
        sampling_algorithm = getattr(job, "sampling_algorithm", None)  # Sweep
        early_termination = getattr(job, "early_termination", None)  # Sweep
        trial_component = getattr(job, "trial", None)  # Sweep
        pipeline_settings = getattr(job, "settings", None)  # Pipeline
        pipeline_sub_jobs = getattr(job, "jobs", None)  # Pipeline

        # Environment (Name vs ID)
        environment_name = None
        environment_id = None
        environment_obj = getattr(job, "environment", None)
        if environment_obj:
            if isinstance(environment_obj, str):
                # Assume it's an ARM ID or shorthand like azureml:name@version
                environment_id = environment_obj
                # Try to parse name from ID string if possible (basic parsing)
                if ":" in environment_obj:
                    parts = environment_obj.split("/")
                    if len(parts) > 1 and "@" in parts[-1]:
                        environment_name = parts[-1]
                    elif len(parts) > 1:
                        environment_name = parts[-1]  # Fallback to last segment
                else:
                    environment_name = environment_obj  # If just 'name@version'
            elif hasattr(environment_obj, "name"):
                environment_name = environment_obj.name
                # Check if environment_obj itself might be the ID string? Less likely now.
                if hasattr(environment_obj, "id"):
                    environment_id = environment_obj.id
                elif isinstance(
                    environment_obj, str
                ):  # Double check, SDK might be inconsistent
                    environment_id = str(environment_obj)

        # Compute (Name vs ID vs Type)
        compute_id_from_props = None
        compute_type = None
        if job_properties:
            compute_type = job_properties.get("_azureml.ComputeTargetType")
            # REST schema shows properties.computeId, let's check if it's in job.properties
            compute_id_from_props = job_properties.get(
                "computeId"
            )  # Or maybe job.properties['computeId']? Unsure

        # Resources (Instance Count/Type)
        instance_count = None
        instance_type = None
        if hasattr(job, "resources"):
            instance_count = getattr(job.resources, "instance_count", None)
            instance_type = getattr(job.resources, "instance_type", None)

        # Creation / Modification Context
        created_at = None
        created_by = None
        created_by_type = None
        last_modified_at = None
        last_modified_by = None
        last_modified_by_type = None
        if hasattr(job, "creation_context"):
            created_at = safe_isoformat(
                getattr(job.creation_context, "created_at", None)
            )
            created_by = getattr(job.creation_context, "created_by", None)
            created_by_type = getattr(job.creation_context, "created_by_type", None)
            last_modified_at = safe_isoformat(
                getattr(job.creation_context, "last_modified_at", None)
            )
            last_modified_by = getattr(job.creation_context, "last_modified_by", None)
            last_modified_by_type = getattr(
                job.creation_context, "last_modified_by_type", None
            )

        # Start/End Time & Duration
        start_time = None
        end_time = None
        duration_seconds = None

        if mlflow_start_time_ms is not None:
            start_time = safe_isoformat(mlflow_start_time_ms)
        elif job_properties and "StartTimeUtc" in job_properties:
            start_time = safe_isoformat(job_properties["StartTimeUtc"])
        # else: start_time remains None

        if mlflow_end_time_ms is not None:
            end_time = safe_isoformat(mlflow_end_time_ms)
        elif job_properties and "EndTimeUtc" in job_properties:
            end_time = safe_isoformat(job_properties["EndTimeUtc"])
        # else: end_time remains None

        # Calculate duration using the determined start/end times (could be ms, iso str, or datetime)
        start_source = (
            mlflow_start_time_ms
            if mlflow_start_time_ms
            else (job_properties.get("StartTimeUtc") if job_properties else None)
        )
        end_source = (
            mlflow_end_time_ms
            if mlflow_end_time_ms
            else (job_properties.get("EndTimeUtc") if job_properties else None)
        )
        duration_seconds = safe_duration(start_source, end_source)

        # --- Assemble JobMetadata ---
        jm = JobMetadata(
            # Core Identifiers
            name=job_name,  # Use the name we used to retrieve the job
            id=job_id,
            display_name=display_name,
            job_type=job_type,
            status=status,
            # Hierarchy and Context
            experiment_name=experiment_name,
            parent_job_name=parent_job_name,
            # Timestamps and Duration
            created_at=created_at,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration_seconds,
            # Creator Info
            created_by=created_by,
            created_by_type=created_by_type,
            # Modification Info
            last_modified_at=last_modified_at,
            last_modified_by=last_modified_by,
            last_modified_by_type=last_modified_by_type,
            # Execution Details
            command=command,
            script_name=script_name,  # From MLflow
            environment_name=environment_name,
            environment_id=environment_id,
            environment_variables=environment_variables,
            code_id=code_id,
            arguments=None,  # Keeping separate from command/job_parameters for now
            job_parameters=job_parameters,
            # Compute Details
            compute_target=compute_target_name,
            compute_id=compute_id_from_props,  # Prefer value from properties if available
            compute_type=compute_type,
            instance_count=instance_count,
            instance_type=instance_type,
            # Job Specific Configuration
            distribution=distribution,
            job_limits=job_limits,
            job_inputs=job_inputs,
            job_outputs=job_outputs,
            identity_type=identity_type,
            services=services,
            job_properties=job_properties,
            # AutoML/Sweep Specific
            task_details=task_details,
            objective=objective,
            search_space=search_space,
            sampling_algorithm=sampling_algorithm,
            early_termination=early_termination,
            trial_component=trial_component,
            # Pipeline Specific
            pipeline_settings=pipeline_settings,
            pipeline_sub_jobs=pipeline_sub_jobs,
            # Metadata
            description=description,
            tags=tags,
            # MLflow Specific Data
            mlflow_run_id=mlflow_run_id,
            mlflow_run_name=mlflow_run_name,
            mlflow_experiment_id=mlflow_experiment_id,
            mlflow_user_id=mlflow_user_id,
            mlflow_artifact_uri=mlflow_artifact_uri,
            mlflow_metrics=mlflow_metrics,
            mlflow_params=mlflow_params,
            mlflow_tags=mlflow_tags_dict,
            mlflow_dataset_inputs=mlflow_dataset_inputs,
        )

        yield jm


# --- Main Execution ---


def main(source_config: str, output_path: str):
    """Main function to extract job metadata and save to JSON."""
    try:
        source_client = get_ml_client(source_config)
        logger.info(f"Connected to workspace: {source_client.workspace_name}")
    except Exception as e:
        logger.error(
            f"Failed to connect to source workspace using config {source_config}: {e}"
        )
        return

    output_dir = os.path.dirname(output_path)
    if (
        output_dir
    ):  # Ensure directory exists only if output_path includes a directory part
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory ensured: {output_dir}")

    job_count = 0
    first_item = True
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("[\n")

            for job_metadata in extract_all_jobs(source_client):
                try:
                    # Use the custom to_dict method for better serialization control
                    job_dict = job_metadata.to_dict()
                    job_json = json.dumps(job_dict, indent=2)

                    if not first_item:
                        f.write(",\n")
                    else:
                        first_item = False

                    f.write(job_json)
                    job_count += 1

                    # Optional: Provide less frequent updates with tqdm
                    # if job_count % 50 == 0:
                    #     logger.info(f"Processed {job_count} jobs so far...")

                except TypeError as e:
                    logger.error(
                        f"Serialization Error for job {job_metadata.name}: {e}. Skipping this job."
                    )
                    logger.debug(
                        f"Problematic JobMetadata: {job_metadata}"
                    )  # Log full object for debugging
                except Exception as e:
                    logger.error(
                        f"Unexpected error processing/writing job {job_metadata.name}: {e}. Skipping this job."
                    )
                    logger.debug(f"Problematic JobMetadata: {job_metadata}")

            f.write("\n]")

        logger.info(f"Successfully extracted {job_count} jobs to {output_path}")

    except IOError as e:
        logger.error(f"Failed to write to output file {output_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during extraction: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract AzureML job metadata to JSON")
    parser.add_argument(
        "--source", required=True, help="Path to source workspace config JSON"
    )
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()
    main(args.source, args.output)

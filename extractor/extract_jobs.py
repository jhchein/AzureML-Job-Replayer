# %load_ext autoreload
# %autoreload 2
import argparse
import json
import logging
import os
import sys
import random
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Set

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

# --- Logging Setup ---

# Get our specific logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Capture INFO level messages for the file
logger.propagate = False  # Prevent propagation to root logger

# Also configure the azure SDK loggers if desired
azure_logger = logging.getLogger("azure")
azure_logger.setLevel(logging.INFO)  # Capture SDK INFO messages for the file
azure_logger.propagate = False


def setup_logging():
    """Configures file and console logging handlers."""
    # Create logs directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamped log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"extract_jobs_{timestamp}.log")

    # --- File Handler (INFO and above) ---
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # --- Console Handler (WARNING and above) ---
    # Use sys.stderr because tqdm typically writes to stderr
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter("%(levelname)s: %(name)s: %(message)s")
    console_handler.setFormatter(console_formatter)

    # --- Add Handlers ---
    # Add handlers to our specific logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Add handlers to the azure SDK logger
    azure_logger.addHandler(file_handler)
    azure_logger.addHandler(console_handler)

    print(f"Detailed logs will be written to: {log_filename}")  # Info to stdout


# --- Helper Functions (Keep as before) ---


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
            # Handle potential timezone info like 'Z' or +00:00
            dt_obj = (
                date_parser.isoparse(dt_obj)
                if ("Z" in dt_obj or "+" in dt_obj)
                else date_parser.parse(dt_obj)
            )

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
    except (ValueError, TypeError, OverflowError, date_parser.ParserError) as e:
        logger.warning(f"Failed to convert timestamp {dt_obj} to ISO format: {e}")
        return str(dt_obj)  # Fallback


def safe_duration(start: Any, end: Any) -> Optional[float]:
    """Safely calculate duration in seconds between two timestamp representations."""
    if start is None or end is None:
        return None
    try:
        start_dt = None
        end_dt = None

        # Function to parse time input
        def parse_time(time_input):
            if isinstance(time_input, (int, float)):
                dt = datetime.fromtimestamp(
                    time_input / 1000 if time_input > 1e12 else time_input,
                    tz=timezone.utc,
                )
            elif isinstance(time_input, str):
                # Use isoparse for better ISO 8601 handling, fallback to parse
                try:
                    dt = date_parser.isoparse(time_input)
                except ValueError:
                    dt = date_parser.parse(time_input)

            elif isinstance(time_input, datetime):
                dt = time_input
            else:
                return None

            # Ensure timezone awareness for comparison
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)  # Convert aware datetimes to UTC

        start_dt = parse_time(start)
        end_dt = parse_time(end)

        if start_dt and end_dt:
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
                f"Could not parse start ({start}, type: {type(start)}) or end ({end}, type: {type(end)}) time for duration calculation."
            )
            return None

    except (ValueError, TypeError, OverflowError, date_parser.ParserError) as e:
        logger.warning(f"Failed to calculate duration between {start} and {end}: {e}")
        return None


def convert_complex_dict(obj: Any) -> Any:
    """Recursively convert complex objects (like SDK entities) within dicts/lists to basic types."""
    if isinstance(obj, dict):
        # Handle potential non-string keys if they are basic types
        return {
            (
                str(k) if not isinstance(k, (str, int, float, bool, type(None))) else k
            ): convert_complex_dict(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list) or isinstance(obj, tuple):  # Include tuples
        return [convert_complex_dict(item) for item in obj]
    elif hasattr(obj, "_to_dict"):  # Handle Azure SDK entities with _to_dict method
        try:
            return convert_complex_dict(obj._to_dict())
        except Exception as e:
            logger.debug(
                f"Could not convert object '{type(obj).__name__}' with _to_dict: {e}. Falling back."
            )
            pass  # Fall through to other methods
    elif hasattr(obj, "asdict"):  # Handle dataclasses
        try:
            return convert_complex_dict(asdict(obj))
        except Exception as e:
            logger.debug(
                f"Could not convert object '{type(obj).__name__}' with asdict: {e}. Falling back."
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
            if not public_attrs and hasattr(obj, "__slots__"):  # Handle slotted classes
                public_attrs = {
                    slot: getattr(obj, slot, None)
                    for slot in obj.__slots__
                    if not slot.startswith("_")
                }
            return (
                convert_complex_dict(public_attrs) if public_attrs else str(obj)
            )  # Fallback if no public attrs
        except Exception as e:
            logger.debug(
                f"Could not convert object '{type(obj).__name__}' using __dict__/__slots__: {e}. Falling back."
            )
            pass
    # Handle basic types directly
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    # Handle specific types that need string conversion
    elif isinstance(obj, (datetime, bytes)):
        return (
            safe_isoformat(obj)
            if isinstance(obj, datetime)
            else obj.decode("utf-8", errors="replace")
        )
    # Fallback for other types (timedelta, non-serializable objects)
    else:
        try:
            # Attempt a generic string conversion as last resort
            s = str(obj)
            # Avoid overly long or complex string representations in JSON
            if len(s) > 200:
                s = s[:197] + "..."
            return s
        except Exception:
            return f"<Unserializable object: {type(obj).__name__}>"


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

    # Add a field to store child run names/ids if needed directly on parent. Optional.
    child_run_ids: Optional[List[str]] = None

    def to_dict(self):
        """Convert dataclass to dictionary, handling complex types."""
        d = {}
        for f in fields(self):
            value = getattr(self, f.name)
            # Ensure keys are strings for JSON compatibility
            if isinstance(value, dict):
                value = {str(k): v for k, v in value.items()}
            d[f.name] = convert_complex_dict(value)
        return d


# --- Extraction Logic ---


def _process_job_object(job: Job, mlflow_client: MlflowClient) -> Optional[JobMetadata]:
    """
    Processes a single Azure ML Job object (parent or child)
    and returns its JobMetadata.
    Returns None if the job object itself is invalid or causes critical errors.
    """
    job_name = job.name  # Use the name from the passed Job object
    logger.info(f"Processing job object: {job_name} (Type: {job.type})")

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
        mlflow_run = mlflow_client.get_run(job_name)  # Use job_name which IS the run_id
        if mlflow_run:
            logger.debug(f"Found MLflow run for job {job_name}")
            if mlflow_run.data:
                mlflow_metrics = mlflow_run.data.metrics or {}
                mlflow_params = mlflow_run.data.params or {}
                mlflow_tags_dict = mlflow_run.data.tags or {}
                script_name = mlflow_tags_dict.get("mlflow.source.name")
                logger.info(
                    f"Retrieved {len(mlflow_metrics)} metrics, {len(mlflow_params)} params for MLflow run {job_name}"
                )
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
                mlflow_dataset_inputs = convert_complex_dict(
                    mlflow_run.inputs.dataset_inputs
                )

    except MlflowException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e) or (
            "Run with id=" in str(e) and "not found" in str(e)
        ):
            logger.info(
                f"No MLflow run found for job {job_name}. This might be expected."
            )
        else:
            logger.warning(f"MLflow Error retrieving run for job {job_name}: {e}")
    except Exception as e:
        logger.warning(
            f"Unexpected Error retrieving MLflow run for job {job_name}: {e}"
        )

    # --- Job Object Data Extraction ---
    # (Use the SAME getattr logic as before, applied to the 'job' object passed in)
    # ... (all the getattr calls: job_id, display_name, job_type, status, etc.) ...
    job_id = getattr(job, "id", None)
    display_name = getattr(job, "display_name", job.name)
    job_type = getattr(job, "type", None)
    status = getattr(job, "status", None)
    experiment_name = getattr(job, "experiment_name", None)
    # Crucially, get parent_job_name from the job object itself
    parent_job_name = getattr(job, "parent_job_name", None)
    description = getattr(job, "description", None)
    tags = getattr(job, "tags", {})
    command = getattr(job, "command", None)
    job_parameters = getattr(job, "parameters", None)
    environment_variables = getattr(job, "environment_variables", None)
    code_obj = getattr(job, "code", None)
    code_id = str(code_obj) if code_obj else None
    compute_target_name = getattr(job, "compute", None)
    job_inputs = getattr(job, "inputs", None)
    job_outputs = getattr(job, "outputs", None)
    identity_obj = getattr(job, "identity", None)
    identity_type = getattr(identity_obj, "type", None) if identity_obj else None
    services = getattr(job, "services", None)
    job_properties = getattr(job, "properties", None)

    # Specific Job Type Fields
    distribution = getattr(job, "distribution", None)
    job_limits = getattr(job, "limits", None)
    task_details = getattr(job, "task_details", None)
    objective = getattr(job, "objective", None)
    search_space = getattr(job, "search_space", None)
    sampling_algorithm = getattr(job, "sampling_algorithm", None)
    early_termination = getattr(job, "early_termination", None)
    trial_component = getattr(job, "trial", None)
    pipeline_settings = getattr(job, "settings", None)
    pipeline_sub_jobs = getattr(
        job, "jobs", None
    )  # Note: This usually just has keys, not full objects

    # Environment
    environment_name = None
    environment_id = None
    environment_obj = getattr(job, "environment", None)
    if environment_obj:
        if isinstance(environment_obj, str):
            environment_id = environment_obj
            parts = environment_obj.split("/")
            name_part = parts[-1]
            if ":" in name_part and "@" in name_part:
                environment_name = name_part.split("@")[0]
            elif ":" in name_part:
                environment_name = name_part
            else:
                environment_name = name_part
        elif hasattr(environment_obj, "name"):
            environment_name = environment_obj.name
            if hasattr(environment_obj, "id"):
                environment_id = environment_obj.id
            elif isinstance(environment_obj, str):
                environment_id = str(environment_obj)

    # Compute
    compute_id_from_props = None
    compute_type = None
    if job_properties:
        compute_type = job_properties.get("_azureml.ComputeTargetType")
        compute_id_from_props = job_properties.get("computeId")

    # Resources
    instance_count = None
    instance_type = None
    if hasattr(job, "resources"):
        instance_count = getattr(job.resources, "instance_count", None)
        instance_type = getattr(job.resources, "instance_type", None)

    # Creation / Modification Context
    created_at_dt = (
        getattr(job.creation_context, "created_at", None)
        if hasattr(job, "creation_context")
        else None
    )
    created_at = safe_isoformat(created_at_dt)
    created_by = (
        getattr(job.creation_context, "created_by", None)
        if hasattr(job, "creation_context")
        else None
    )
    created_by_type = (
        getattr(job.creation_context, "created_by_type", None)
        if hasattr(job, "creation_context")
        else None
    )
    last_modified_at_dt = (
        getattr(job.creation_context, "last_modified_at", None)
        if hasattr(job, "creation_context")
        else None
    )
    last_modified_at = safe_isoformat(last_modified_at_dt)
    last_modified_by = (
        getattr(job.creation_context, "last_modified_by", None)
        if hasattr(job, "creation_context")
        else None
    )
    last_modified_by_type = (
        getattr(job.creation_context, "last_modified_by_type", None)
        if hasattr(job, "creation_context")
        else None
    )

    # Start/End Time & Duration
    prop_start = job_properties.get("StartTimeUtc") if job_properties else None
    prop_end = job_properties.get("EndTimeUtc") if job_properties else None
    start_time_raw = (
        mlflow_start_time_ms if mlflow_start_time_ms is not None else prop_start
    )
    end_time_raw = mlflow_end_time_ms if mlflow_end_time_ms is not None else prop_end
    start_time = safe_isoformat(start_time_raw)
    end_time = safe_isoformat(end_time_raw)
    duration_seconds = safe_duration(start_time_raw, end_time_raw)

    # --- Assemble JobMetadata ---
    jm = JobMetadata(
        # Core Identifiers
        name=job_name,  # Use the name from the Job object
        id=job_id,
        display_name=display_name,
        job_type=job_type,
        status=status,
        # Hierarchy and Context
        experiment_name=experiment_name,
        parent_job_name=parent_job_name,  # Get directly from job object
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
        script_name=script_name,
        environment_name=environment_name,
        environment_id=environment_id,
        environment_variables=environment_variables,
        code_id=code_id,
        arguments=None,
        job_parameters=job_parameters,
        # Compute Details
        compute_target=compute_target_name,
        compute_id=compute_id_from_props,
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
        pipeline_sub_jobs=pipeline_sub_jobs,  # Still just keys here
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
    return jm


def extract_all_jobs(
    client: MLClient,
    limit: Optional[int] = None,
    filter: Optional[
        list[str]
    ] = None,  # (kept for backward compatibility / future use)
    include_names: Optional[Set[str]] = None,
    automl_max_trials: Optional[int] = None,
    automl_top_metric: Optional[str] = None,
    automl_trial_sampling: str = "best",  # best|first|random
    automl_include_trials: bool = True,
) -> Iterator[JobMetadata]:
    """
    Extract selected top-level jobs (optionally restricted by include_names)
    AND all their recursively nested child jobs (for pipelines),
    yielding JobMetadata objects.

    include_names:
        If provided, only top-level jobs whose name is in this set are processed.
        Their descendants are still fully traversed.
    """
    mlflow_tracking_uri = client.workspaces.get(
        client.workspace_name
    ).mlflow_tracking_uri
    logger.info(f"Using MLflow tracking URI: {mlflow_tracking_uri}")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow_client = MlflowClient()

    try:
        all_job_summaries = list(client.jobs.list())
    except Exception as e:
        logger.exception(f"Failed to list initial job summaries: {e}")
        return

    total_jobs = len(all_job_summaries)

    # Apply include_names filter (top-level only)
    if include_names:
        name_to_summary = {s.name: s for s in all_job_summaries}
        missing = [n for n in include_names if n not in name_to_summary]
        if missing:
            logger.warning(
                f"{len(missing)} requested top-level job name(s) not found and will be skipped: {missing}"
            )
        filtered_summaries = [
            name_to_summary[n] for n in include_names if n in name_to_summary
        ]
        print(
            f"Top-level include filter active: {len(filtered_summaries)}/{total_jobs} jobs selected before limit."
        )
    else:
        filtered_summaries = all_job_summaries
        print(
            f"Found {total_jobs} total top-level job summaries. Starting extraction including children..."
        )

    # Apply limit AFTER filtering (cap the number of selected roots)
    if limit is not None and limit < len(filtered_summaries):
        print(
            f"Limiting selected top-level jobs to {limit} (from {len(filtered_summaries)} after filtering)."
        )
        job_summaries_to_process = filtered_summaries[:limit]
    else:
        job_summaries_to_process = filtered_summaries

    visited: Set[str] = set()

    for job_summary in tqdm(
        job_summaries_to_process, desc="Processing Top-Level Jobs", unit="job"
    ):
        root_job_name = job_summary.name
        if root_job_name in visited:
            continue
        try:
            root_job_object: Job = client.jobs.get(name=root_job_name)
        except ResourceNotFoundError:
            logger.warning(
                f"Top-level job {root_job_name} summary found but GET failed (not found). Skipping."
            )
            continue
        except Exception as e:
            logger.error(
                f"Failed to retrieve full job details for top-level job {root_job_name}: {e}. Skipping."
            )
            continue

        root_metadata = _process_job_object(root_job_object, mlflow_client)
        if not root_metadata:
            logger.warning(f"Failed to process top-level job object {root_job_name}.")
            continue

        visited.add(root_job_name)
        yield root_metadata

        # Always traverse descendants (not only pipelines) to capture AutoML / Sweep / nested structures
        yield from _traverse_descendants(
            client=client,
            mlflow_client=mlflow_client,
            root_name=root_job_name,
            visited=visited,
            automl_max_trials=automl_max_trials,
            automl_top_metric=automl_top_metric,
            automl_trial_sampling=automl_trial_sampling,
            automl_include_trials=automl_include_trials,
        )


def _traverse_descendants(
    client: MLClient,
    mlflow_client: MlflowClient,
    root_name: str,
    visited: Set[str],
    automl_max_trials: Optional[int],
    automl_top_metric: Optional[str],
    automl_trial_sampling: str,
    automl_include_trials: bool,
) -> Iterator[JobMetadata]:
    """Generic depth-first traversal of all descendant jobs (pipelines, AutoML, sweeps, etc.)."""
    stack: List[str] = [root_name]

    while stack:
        current_parent = stack.pop()

        # Fetch parent to detect type & properties
        try:
            parent_job_obj: Job = client.jobs.get(name=current_parent)
            parent_type_raw = getattr(parent_job_obj, "type", "") or ""
            parent_type = parent_type_raw.lower()
            parent_properties = getattr(parent_job_obj, "properties", {}) or {}
            parent_tags = getattr(parent_job_obj, "tags", {}) or {}
        except Exception:
            parent_type = ""
            parent_properties = {}
            parent_tags = {}

        try:
            child_job_summaries = list(client.jobs.list(parent_job_name=current_parent))
        except Exception as e:
            logger.error(f"Failed to list child jobs for parent {current_parent}: {e}")
            continue

        if not child_job_summaries:
            continue

        logger.info(
            f"Found {len(child_job_summaries)} child jobs for parent '{current_parent}'."
        )

        processed_children: List[JobMetadata] = []
        for child_summary in child_job_summaries:
            child_name = child_summary.name
            if child_name in visited:
                continue
            try:
                child_job_object: Job = client.jobs.get(name=child_name)
            except ResourceNotFoundError:
                logger.warning(
                    f"Child job {child_name} listed but GET failed (not found) for parent {current_parent}. Skipping."
                )
                continue
            except Exception as e:
                logger.error(
                    f"Failed to retrieve full details for child job {child_name} (parent {current_parent}): {e}"
                )
                continue
            child_metadata = _process_job_object(child_job_object, mlflow_client)
            if not child_metadata:
                logger.warning(
                    f"Failed to process child job object {child_name} (parent {current_parent})."
                )
                continue
            if child_metadata.parent_job_name != current_parent:
                child_metadata.parent_job_name = current_parent
            processed_children.append(child_metadata)

        # Detect AutoML parent (can present as type 'automl' OR a step with runTemplate AutoML / StepType AutoMLStep)
        is_automl_parent = False
        if parent_type == "automl":
            is_automl_parent = True
        else:
            run_template = parent_properties.get("runTemplate")
            step_type = parent_properties.get("StepType")
            pipeline_component = parent_properties.get("azureml.pipelineComponent", "")
            if (
                run_template == "AutoML"
                or step_type == "AutoMLStep"
                or pipeline_component.startswith("masterautoml")
                or "automl_best_child_run_id" in parent_tags
            ):
                is_automl_parent = True

        if is_automl_parent:
            if not automl_include_trials:
                logger.info(
                    f"AutoML trials excluded for parent {current_parent}; skipping its child trials."
                )
                # Still yield the AutoML parent itself (already yielded earlier) – skip its children
                continue
            if processed_children:
                primary_metric = automl_top_metric
                if not primary_metric:
                    for cm in processed_children:
                        if cm.mlflow_metrics:
                            for k, v in cm.mlflow_metrics.items():
                                if isinstance(v, (int, float)):
                                    primary_metric = k
                                    break
                        if primary_metric:
                            break
                scored = []
                unscored = []
                for cm in processed_children:
                    val = None
                    if primary_metric and cm.mlflow_metrics:
                        metric_val = cm.mlflow_metrics.get(primary_metric)
                        if isinstance(metric_val, (int, float)):
                            val = float(metric_val)
                    if val is not None:
                        scored.append((val, cm))
                    else:
                        unscored.append(cm)
                if automl_trial_sampling == "first":
                    selected_children = processed_children
                elif automl_trial_sampling == "random":
                    random.shuffle(processed_children)
                    selected_children = processed_children
                else:  # best
                    if scored:
                        scored.sort(key=lambda x: x[0], reverse=True)
                        ordered = [c for _, c in scored] + unscored
                        selected_children = ordered
                    else:
                        selected_children = processed_children
                if (
                    automl_max_trials is not None
                    and len(selected_children) > automl_max_trials
                ):
                    logger.info(
                        f"AutoML trial cap: keeping top {automl_max_trials} of {len(selected_children)} for parent {current_parent}."
                    )
                    selected_children = selected_children[:automl_max_trials]
                processed_children = selected_children

        # Yield children and push each onto stack for further traversal (generic approach)
        for child_metadata in processed_children:
            child_name = child_metadata.name
            if child_name in visited:
                continue
            visited.add(child_name)
            yield child_metadata
            # Always attempt further traversal (cheap if no children)
            stack.append(child_name)


# --- Main Execution ---


def main(
    source_config: str,
    output_path: str,
    limit: Optional[int] = None,
    include_names: Optional[Set[str]] = None,
    automl_max_trials: Optional[int] = None,
    automl_top_metric: Optional[str] = None,
    automl_trial_sampling: str = "best",
    automl_include_trials: bool = True,
):
    setup_logging()
    try:
        source_client = get_ml_client(source_config)
        print(f"Connected to workspace: {source_client.workspace_name}")
    except Exception as e:
        logger.error(
            f"Failed to connect to source workspace using config {source_config}: {e}"
        )
        return

    output_dir = os.path.dirname(output_path)
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            return

    job_count = 0
    first_item = True
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("[")
            for job_metadata in extract_all_jobs(
                source_client,
                limit=limit,
                include_names=include_names,
                automl_max_trials=automl_max_trials,
                automl_top_metric=automl_top_metric,
                automl_trial_sampling=automl_trial_sampling,
                automl_include_trials=automl_include_trials,
            ):
                if not first_item:
                    f.write(",\n")
                else:
                    first_item = False
                f.write(json.dumps(job_metadata.to_dict(), indent=2))
                job_count += 1
            f.write("]")

        print(
            f"\nSuccessfully extracted {job_count} total jobs (including descendants) to {output_path}"
        )

    except IOError as e:
        logger.error(f"Failed to write to output file {output_path}: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during extraction: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract AzureML job metadata (including (nested) pipeline children) to JSON"
    )
    parser.add_argument(
        "--source", required=True, help="Path to source workspace config JSON"
    )
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of TOP-LEVEL jobs to process (after filtering).",
    )
    parser.add_argument(
        "--include",
        help="Comma-separated list of top-level job names to include (exact match).",
    )
    parser.add_argument(
        "--include-file",
        help="Path to a text file with one top-level job name per line to include.",
    )
    parser.add_argument(
        "--automl-max-trials",
        type=int,
        default=None,
        help="Limit number of AutoML trial child jobs per AutoML parent (after ranking).",
    )
    parser.add_argument(
        "--automl-top-metric",
        type=str,
        default=None,
        help="Primary metric name to rank AutoML trials; if omitted, inferred from first numeric metric encountered.",
    )
    parser.add_argument(
        "--automl-trial-sampling",
        type=str,
        choices=["best", "first", "random"],
        default="best",
        help="Sampling strategy for selecting AutoML trials when limiting: best (by metric), first (original order), random (shuffle).",
    )
    parser.add_argument(
        "--no-automl-trials",
        action="store_true",
        help="Exclude AutoML trial child jobs entirely; only export parent AutoML job metadata.",
    )
    args = parser.parse_args()

    include_names: Set[str] = set()
    if args.include:
        include_names.update(n.strip() for n in args.include.split(",") if n.strip())
    if args.include_file:
        try:
            with open(args.include_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    name = line.strip()
                    if name:
                        include_names.add(name)
        except OSError as e:
            print(f"WARNING: Failed to read include file '{args.include_file}': {e}")

    if not include_names:
        include_names = None  # Pass None instead of empty set for clarity

    main(
        args.source,
        args.output,
        args.limit,
        include_names=include_names,
        automl_max_trials=args.automl_max_trials,
        automl_top_metric=args.automl_top_metric,
        automl_trial_sampling=args.automl_trial_sampling,
        automl_include_trials=not args.no_automl_trials,
    )

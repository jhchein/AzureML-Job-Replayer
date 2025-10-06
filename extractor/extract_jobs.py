import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, Future
from collections import deque
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple
import random as _rand  # jitter for retries

import mlflow
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Job
from azure.core.exceptions import ResourceNotFoundError
from dateutil import parser as date_parser
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from utils.aml_clients import get_ml_client
from utils.log_setup import setup_logging

# --- Logging Setup ---

logger = logging.getLogger(__name__)


def safe_isoformat(dt_obj: Any) -> Optional[str]:
    if dt_obj is None:
        return None
    try:
        if isinstance(dt_obj, (int, float)):
            if dt_obj > 1e12:
                dt_obj = datetime.fromtimestamp(dt_obj / 1000, tz=timezone.utc)
            else:
                dt_obj = datetime.fromtimestamp(dt_obj, tz=timezone.utc)
        elif isinstance(dt_obj, str):
            dt_obj = (
                date_parser.isoparse(dt_obj)
                if ("Z" in dt_obj or "+" in dt_obj)
                else date_parser.parse(dt_obj)
            )
        if isinstance(dt_obj, datetime):
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            return dt_obj.isoformat()
        logger.warning(
            f"Cannot convert non-datetime object to ISO format: {dt_obj} (Type: {type(dt_obj)})"
        )
        return str(dt_obj)
    except (ValueError, TypeError, OverflowError, date_parser.ParserError) as e:  # type: ignore[attr-defined]
        logger.warning(f"Failed to convert timestamp {dt_obj} to ISO format: {e}")
        return str(dt_obj)


def safe_duration(start: Any, end: Any) -> Optional[float]:
    if start is None or end is None:
        return None
    try:

        def parse_time(time_input):
            if isinstance(time_input, (int, float)):
                dt = datetime.fromtimestamp(
                    time_input / 1000 if time_input > 1e12 else time_input,
                    tz=timezone.utc,
                )
            elif isinstance(time_input, str):
                try:
                    dt = date_parser.isoparse(time_input)
                except ValueError:
                    dt = date_parser.parse(time_input)
            elif isinstance(time_input, datetime):
                dt = time_input
            else:
                return None
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        start_dt = parse_time(start)
        end_dt = parse_time(end)

        if start_dt and end_dt:
            if end_dt >= start_dt:
                return (end_dt - start_dt).total_seconds()
            logger.warning(
                f"End time {end_dt} is before start time {start_dt}. Cannot calculate duration."
            )
            return None
        logger.warning(
            f"Could not parse start ({start}, type: {type(start)}) or end ({end}, type: {type(end)}) time for duration calculation."
        )
        return None
    except (ValueError, TypeError, OverflowError, date_parser.ParserError) as e:  # type: ignore[attr-defined]
        logger.warning(f"Failed to calculate duration between {start} and {end}: {e}")
        return None


def convert_complex_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            (
                str(k) if not isinstance(k, (str, int, float, bool, type(None))) else k
            ): convert_complex_dict(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [convert_complex_dict(item) for item in obj]
    elif hasattr(obj, "_to_dict"):
        try:
            return convert_complex_dict(obj._to_dict())
        except Exception:  # noqa: BLE001
            pass
    elif hasattr(obj, "asdict"):
        try:
            return convert_complex_dict(asdict(obj))
        except Exception:  # noqa: BLE001
            pass
    elif hasattr(obj, "__dict__"):
        try:
            public_attrs = {
                k: v
                for k, v in vars(obj).items()
                if not k.startswith("_") and not callable(v)
            }
            if not public_attrs and hasattr(obj, "__slots__"):
                public_attrs = {
                    slot: getattr(obj, slot, None)
                    for slot in obj.__slots__  # type: ignore[attr-defined]
                    if not slot.startswith("_")
                }
            return convert_complex_dict(public_attrs) if public_attrs else str(obj)
        except Exception:  # noqa: BLE001
            pass
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, datetime):
        return safe_isoformat(obj)
    elif isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    try:
        s = str(obj)
        if len(s) > 200:
            s = s[:197] + "..."
        return s
    except Exception:  # noqa: BLE001
        return f"<Unserializable object: {type(obj).__name__}>"


def _get_default_artifact_folders() -> List[str]:
    """Return the standard MLflow artifact folder names without making REST calls.

    These are the default top-level directories created by AzureML jobs.
    Using this static list eliminates the need for recursive artifact listing,
    which can cause thousands of REST calls for deeply nested artifact trees.

    Returns folder prefixes that will be used for recursive blob copy operations:
    - "outputs/": All job outputs (files and subdirectories copied recursively)
    - "system_logs/": System-generated logs (copied recursively to outputs/original_logs/)
    - "logs/": General logs (copied recursively to outputs/original_logs/)
    - "user_logs/": User-generated logs (copied recursively to outputs/original_logs/)
    """
    return ["outputs/", "system_logs/", "logs/", "user_logs/"]


@dataclass
class JobMetadata:
    name: str
    id: Optional[str] = None
    display_name: Optional[str] = None
    job_type: Optional[str] = None
    status: Optional[str] = None
    experiment_name: Optional[str] = None
    parent_job_name: Optional[str] = None
    created_at: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    created_by: Optional[str] = None
    created_by_type: Optional[str] = None
    last_modified_at: Optional[str] = None
    last_modified_by: Optional[str] = None
    last_modified_by_type: Optional[str] = None
    command: Optional[str] = None
    script_name: Optional[str] = None
    environment_name: Optional[str] = None
    environment_id: Optional[str] = None
    environment_variables: Optional[Dict[str, str]] = None
    code_id: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    job_parameters: Optional[Dict[str, Any]] = None
    compute_target: Optional[str] = None
    compute_id: Optional[str] = None
    compute_type: Optional[str] = None
    instance_count: Optional[int] = None
    instance_type: Optional[str] = None
    distribution: Optional[Dict[str, Any]] = None
    job_limits: Optional[Dict[str, Any]] = None
    job_inputs: Optional[Dict[str, Any]] = None
    job_outputs: Optional[Dict[str, Any]] = None
    identity_type: Optional[str] = None
    services: Optional[Dict[str, Any]] = None
    job_properties: Optional[Dict[str, Any]] = None
    task_details: Optional[Dict[str, Any]] = None
    objective: Optional[Dict[str, Any]] = None
    search_space: Optional[Dict[str, Any]] = None
    sampling_algorithm: Optional[Dict[str, Any]] = None
    early_termination: Optional[Dict[str, Any]] = None
    trial_component: Optional[Dict[str, Any]] = None
    pipeline_settings: Optional[Dict[str, Any]] = None
    pipeline_sub_jobs: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    mlflow_run_id: Optional[str] = None
    mlflow_run_name: Optional[str] = None
    mlflow_experiment_id: Optional[str] = None
    mlflow_user_id: Optional[str] = None
    mlflow_artifact_uri: Optional[str] = None
    mlflow_metrics: Optional[Dict[str, float]] = None
    mlflow_params: Optional[Dict[str, str]] = None
    mlflow_tags: Optional[Dict[str, str]] = None
    mlflow_dataset_inputs: Optional[List[Any]] = None
    child_run_ids: Optional[List[str]] = None
    mlflow_artifact_paths: Optional[List[str]] = None  # manifest paths only

    def to_dict(self):
        d = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, dict):
                value = {str(k): v for k, v in value.items()}
            d[f.name] = convert_complex_dict(value)
            # convert_complex_dict handles lists recursively
        return d


def _process_job_object(job: Job, mlflow_client: MlflowClient) -> Optional[JobMetadata]:
    if job.name is None:
        logger.warning("Job object without a name encountered; skipping.")
        return None
    job_name: str = job.name
    logger.info(f"Processing job object: {job_name} (Type: {job.type})")

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
        mlflow_run = mlflow_client.get_run(job_name)
        if mlflow_run and mlflow_run.data:
            mlflow_metrics = mlflow_run.data.metrics or {}
            mlflow_params = mlflow_run.data.params or {}
            mlflow_tags_dict = mlflow_run.data.tags or {}
            script_name = mlflow_tags_dict.get("mlflow.source.name")
            logger.info(
                f"Retrieved {len(mlflow_metrics)} metrics, {len(mlflow_params)} params for MLflow run {job_name}"
            )
        if mlflow_run and mlflow_run.info:
            mlflow_run_id = mlflow_run.info.run_id
            mlflow_run_name = mlflow_run.info.run_name
            mlflow_experiment_id = mlflow_run.info.experiment_id
            mlflow_user_id = mlflow_run.info.user_id
            mlflow_artifact_uri = mlflow_run.info.artifact_uri
            mlflow_start_time_ms = mlflow_run.info.start_time
            mlflow_end_time_ms = mlflow_run.info.end_time
        if (
            mlflow_run
            and hasattr(mlflow_run, "inputs")
            and hasattr(mlflow_run.inputs, "dataset_inputs")
        ):
            mlflow_dataset_inputs = convert_complex_dict(
                mlflow_run.inputs.dataset_inputs
            )
    except MlflowException as e:  # noqa: BLE001
        if "RESOURCE_DOES_NOT_EXIST" in str(e) or (
            "Run with id=" in str(e) and "not found" in str(e)
        ):
            logger.info(
                f"No MLflow run found for job {job_name}. This might be expected."
            )
        else:
            logger.warning(f"MLflow Error retrieving run for job {job_name}: {e}")
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"Unexpected Error retrieving MLflow run for job {job_name}: {e}"
        )

    job_id = getattr(job, "id", None)
    display_name = getattr(job, "display_name", job.name)
    job_type = getattr(job, "type", None)
    status = getattr(job, "status", None)
    experiment_name = getattr(job, "experiment_name", None)
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

    distribution = getattr(job, "distribution", None)
    job_limits = getattr(job, "limits", None)
    task_details = getattr(job, "task_details", None)
    objective = getattr(job, "objective", None)
    search_space = getattr(job, "search_space", None)
    sampling_algorithm = getattr(job, "sampling_algorithm", None)
    early_termination = getattr(job, "early_termination", None)
    trial_component = getattr(job, "trial", None)
    pipeline_settings = getattr(job, "settings", None)
    pipeline_sub_jobs = getattr(job, "jobs", None)

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

    compute_id_from_props = None
    compute_type = None
    if job_properties:
        compute_type = job_properties.get("_azureml.ComputeTargetType")
        compute_id_from_props = job_properties.get("computeId")

    instance_count = None
    instance_type = None
    resources_obj = (
        getattr(job, "resources", None) if hasattr(job, "resources") else None
    )
    if resources_obj is not None:
        instance_count = getattr(resources_obj, "instance_count", None)
        instance_type = getattr(resources_obj, "instance_type", None)

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

    prop_start = job_properties.get("StartTimeUtc") if job_properties else None
    prop_end = job_properties.get("EndTimeUtc") if job_properties else None
    start_time_raw = (
        mlflow_start_time_ms if mlflow_start_time_ms is not None else prop_start
    )
    end_time_raw = mlflow_end_time_ms if mlflow_end_time_ms is not None else prop_end
    start_time = safe_isoformat(start_time_raw)
    end_time = safe_isoformat(end_time_raw)
    duration_seconds = safe_duration(start_time_raw, end_time_raw)

    jm = JobMetadata(
        name=job_name,
        id=job_id,
        display_name=display_name,
        job_type=job_type,
        status=status,
        experiment_name=experiment_name,
        parent_job_name=parent_job_name,
        created_at=created_at,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=duration_seconds,
        created_by=created_by,
        created_by_type=created_by_type,
        last_modified_at=last_modified_at,
        last_modified_by=last_modified_by,
        last_modified_by_type=last_modified_by_type,
        command=command,
        script_name=script_name,
        environment_name=environment_name,
        environment_id=environment_id,
        environment_variables=environment_variables,
        code_id=code_id,
        arguments=None,
        job_parameters=job_parameters,
        compute_target=compute_target_name,
        compute_id=compute_id_from_props,
        compute_type=compute_type,
        instance_count=instance_count,
        instance_type=instance_type,
        distribution=distribution,
        job_limits=job_limits,
        job_inputs=job_inputs,
        job_outputs=job_outputs,
        identity_type=identity_type,
        services=services,
        job_properties=job_properties,
        task_details=task_details,
        objective=objective,
        search_space=search_space,
        sampling_algorithm=sampling_algorithm,
        early_termination=early_termination,
        trial_component=trial_component,
        pipeline_settings=pipeline_settings,
        pipeline_sub_jobs=pipeline_sub_jobs,
        description=description,
        tags=tags,
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


# -------------------------------
# Retry helpers
# -------------------------------


def _is_transient(ex: Exception) -> bool:
    msg = str(ex).lower()
    transient_terms = [
        "429",
        "timeout",
        "temporarily",
        "rate limit",
        "unavailable",
        "503",
        "502",
        "connection reset",
        "timed out",
    ]
    return any(t in msg for t in transient_terms)


def _with_retries(fn, *args, **kwargs):
    attempts = kwargs.pop("_attempts", 5)
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            if attempt == attempts - 1 or not _is_transient(e):
                raise
            backoff_base = 0.5
            sleep_time = min(backoff_base * (2**attempt), 8.0)
            # jitter
            sleep_time *= 0.7 + 0.6 * _rand.random()
            logger.warning(
                f"Transient error calling {fn.__name__}: {e} (retry {attempt+1}/{attempts} in {sleep_time:.2f}s)"
            )
            time.sleep(sleep_time)


# -------------------------------
# Concurrency job processor
# -------------------------------


def _process_single_job(
    client: MLClient,
    mlflow_client: MlflowClient,
    job_name: str,
    collect_artifacts: bool,
    visited: Set[str],
    visited_lock: Optional[Lock],
) -> Tuple[Optional[JobMetadata], List[str]]:
    try:
        job_obj: Job = _with_retries(client.jobs.get, name=job_name)  # type: ignore[arg-type]
    except ResourceNotFoundError:
        logger.warning(f"Job {job_name} not found during processing.")
        return None, []
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed job GET for {job_name}: {e}")
        return None, []

    meta = _process_job_object(job_obj, mlflow_client)
    if not meta:
        return None, []

    if collect_artifacts and meta.mlflow_run_id:
        # Use static list of default artifact folders instead of recursive REST calls
        meta.mlflow_artifact_paths = _get_default_artifact_folders()
        logger.debug(
            f"Assigned default artifact folders for run {meta.mlflow_run_id}: {meta.mlflow_artifact_paths}"
        )

    # Children
    try:
        child_summaries = _with_retries(client.jobs.list, parent_job_name=job_name)
        child_summaries = child_summaries or []  # type: ignore[assignment]
        child_names = [
            c.name  # type: ignore[attr-defined]
            for c in child_summaries
            if getattr(c, "name", None) is not None
        ]
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to list children for {job_name}: {e}")
        child_names = []

    new_children: List[str] = []
    if visited_lock:
        with visited_lock:
            for cn in child_names:
                if cn is None:
                    continue
                if cn not in visited:
                    visited.add(cn)
                    new_children.append(cn)
    else:
        for cn in child_names:
            if cn is None:
                continue
            if cn not in visited:
                visited.add(cn)
                new_children.append(cn)
    return meta, new_children


def _init_mlflow(client: MLClient) -> MlflowClient:
    ws_obj = client.workspaces.get(client.workspace_name)
    mlflow_tracking_uri = getattr(ws_obj, "mlflow_tracking_uri", None)
    if mlflow_tracking_uri:
        logger.info(f"Using MLflow tracking URI: {mlflow_tracking_uri}")
        mlflow.set_tracking_uri(mlflow_tracking_uri)  # type: ignore[arg-type]
    else:
        logger.warning(
            "Workspace has no mlflow_tracking_uri; proceeding without MLflow tracking URI override."
        )
    return MlflowClient()


def main(
    source_config: str,
    output_path: str,
    limit: Optional[int] = None,
    include_names: Optional[Set[str]] = None,
    parallel: int = 1,
    include_artifacts: bool = True,
):
    setup_logging(log_filename_prefix="extract_jobs")
    try:
        client = get_ml_client(source_config)
        print(f"Connected to workspace: {client.workspace_name}")
    except Exception as e:  # noqa: BLE001
        logger.error(
            f"Failed to connect to source workspace using config {source_config}: {e}"
        )
        return

    out_dir = os.path.dirname(output_path)
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {out_dir}: {e}")
            return

    mlflow_client = _init_mlflow(client)
    visited: Set[str] = set()
    missing_requested: List[str] = []

    # Determine root job names
    if include_names:
        root_names = list(include_names)
        print(
            f"Extracting {len(root_names)} specified top-level job(s) and descendants..."
        )
    else:
        try:
            summaries_iter = _with_retries(client.jobs.list)
            summaries = list(summaries_iter) if summaries_iter else []
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Failed to list top-level jobs: {e}")
            return
        total = len(summaries)
        if limit is not None and limit < total:
            summaries = summaries[:limit]
            print(f"Found {total} top-level jobs. Limiting to first {limit}.")
        else:
            print(f"Found {total} top-level jobs.")
        root_names = [s.name for s in summaries if getattr(s, "name", None)]
        print("Extracting all selected top-level jobs and their descendants...")

    job_count = 0
    results: List[JobMetadata] = []

    # Seed visited with roots
    for rn in root_names:
        if rn:
            visited.add(rn)

    start_time = time.time()

    if parallel <= 1:
        queue = deque(root_names)
        # Progress bar for sequential processing
        pbar = tqdm(
            total=len(root_names),
            desc="Processing jobs",
            unit="job",
            dynamic_ncols=True,
            file=sys.stdout,
        )
        while queue:
            name = queue.popleft()
            if not name:
                continue
            meta, children = _process_single_job(
                client,
                mlflow_client,
                name,
                include_artifacts,
                visited,
                visited_lock=None,
            )
            if meta:
                results.append(meta)
                job_count += 1
                pbar.set_postfix({"completed": job_count, "queued": len(queue)})
            if children:
                pbar.total += len(children)
                pbar.refresh()
            for c in children:
                queue.append(c)
            pbar.update(1)
        pbar.close()
    else:
        parallel = max(1, parallel)
        visited_lock = Lock()
        executor = ThreadPoolExecutor(max_workers=parallel)
        futures: Set[Future] = set()

        # Progress bar for parallel processing with thread-safe updates
        pbar = tqdm(
            total=len(root_names),
            desc="Processing jobs",
            unit="job",
            dynamic_ncols=True,
            file=sys.stdout,
        )
        pbar_lock = Lock()

        def submit(name: str):
            futures.add(
                executor.submit(
                    _process_single_job,
                    client,
                    mlflow_client,
                    name,
                    include_artifacts,
                    visited,
                    visited_lock,
                )
            )

        for rn in root_names:
            if rn:
                submit(rn)

        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                try:
                    meta, children = fut.result()
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Worker failed: {e}")
                    with pbar_lock:
                        pbar.update(1)
                    continue
                if meta:
                    results.append(meta)
                    job_count += 1
                if children:
                    with pbar_lock:
                        pbar.total += len(children)
                        pbar.refresh()
                    for child in children:
                        submit(child)
                with pbar_lock:
                    pbar.set_postfix(
                        {
                            "completed": job_count,
                            "active": len(futures),
                            "workers": parallel,
                        }
                    )
                    pbar.update(1)
        pbar.close()
        executor.shutdown(wait=True)

    elapsed = time.time() - start_time

    # Persist results
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([m.to_dict() for m in results], f, indent=2)
    except IOError as e:
        logger.error(f"Failed to write to output file {output_path}: {e}")
        return

    if include_names:
        missing_requested_list_raw = [
            n for n in root_names if all(m.name != n for m in results)
        ]
        missing_requested_list: List[str] = [n for n in missing_requested_list_raw if n]
        missing_requested = missing_requested_list
        if missing_requested:
            logger.warning(
                f"{len(missing_requested)} requested top-level job name(s) not found and were skipped: {missing_requested}"
            )

    print(
        f"\nSuccessfully extracted {job_count} total jobs (including descendants) to {output_path} in {elapsed:.2f}s (parallel={parallel})"
    )


def _parse_include_args(args) -> Optional[Set[str]]:
    include_names: Optional[Set[str]] = set()
    if args.include:
        inc_clean = args.include.strip()
        if (
            inc_clean
            and os.path.isfile(inc_clean)
            and not args.include_file
            and inc_clean.lower().endswith((".txt", ".list", ".csv"))
        ):
            logger.warning(
                f"--include looks like a file path '{inc_clean}'. Did you mean to use --include-file {inc_clean}?"
            )
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
        return None
    return include_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract AzureML job metadata (including nested pipeline children) to JSON"
    )
    parser.add_argument(
        "--source", required=True, help="Path to source workspace config JSON"
    )
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of TOP-LEVEL jobs to process.",
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
        "--parallel",
        type=int,
        default=1,
        help="Max number of concurrent job processing workers (default 1 = sequential).",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Skip including artifact folder paths in the output (default: include standard artifact folders).",
    )

    args = parser.parse_args()
    include_names = _parse_include_args(args)
    main(
        source_config=args.source,
        output_path=args.output,
        limit=args.limit,
        include_names=include_names,
        parallel=max(1, args.parallel),
        include_artifacts=not args.no_artifacts,
    )

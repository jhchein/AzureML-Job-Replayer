import argparse
import json
import logging
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from azure.ai.ml import (
    Input,
)
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    PipelineJob,
    PipelineJobSettings,
    UserIdentityConfiguration,
)
from azure.core.exceptions import HttpResponseError
from azure.identity import AzureCliCredential
from azure.storage.blob import (
    BlobServiceClient,
    ContainerSasPermissions,
    generate_container_sas,
)

from extractor.extract_jobs import JobMetadata
from replayer.dummy_components import (
    REGISTERED_ENV_ID,
    replay_metrics_component,
)
from utils.aml_clients import get_ml_client, load_workspace_config
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
ORIGINAL_PARENT_PIPELINE_ID_TAG = "original_parent_pipeline_id"
PIPELINE_DEPTH_TAG = "original_pipeline_depth"


def _index_jobs(jobs: List[JobMetadata]):
    """Build indexes for replay construction.

    Returns:
        jobs_by_name: name -> metadata
        children_map: parent_name -> list[child metadata]
        pipeline_names: set of names where job_type == 'pipeline'
    """
    jobs_by_name: Dict[str, JobMetadata] = {}
    children_map: Dict[str, List[JobMetadata]] = {}
    pipeline_names: Set[str] = set()

    for jm in jobs:
        jobs_by_name[jm.name] = jm
        if jm.job_type and jm.job_type.lower() == "pipeline":
            pipeline_names.add(jm.name)

    for jm in jobs:
        if jm.parent_job_name and jm.parent_job_name in jobs_by_name:
            children_map.setdefault(jm.parent_job_name, []).append(jm)

    return jobs_by_name, children_map, pipeline_names


def _compute_depths(jobs_by_name: Dict[str, JobMetadata]) -> Dict[str, int]:
    """Compute depth (0-based) for each job based on parent chain."""
    depth_cache: Dict[str, int] = {}

    def depth(name: str, seen: Set[str]) -> int:
        if name in depth_cache:
            return depth_cache[name]
        jm = jobs_by_name.get(name)
        if not jm or not jm.parent_job_name:
            depth_cache[name] = 0
            return 0
        if jm.parent_job_name in seen:  # cycle guard
            depth_cache[name] = 0
            return 0
        d = 1 + depth(jm.parent_job_name, seen | {jm.parent_job_name})
        depth_cache[name] = d
        return d

    for n in jobs_by_name:
        depth(n, {n})
    return depth_cache


def build_dummy_pipeline_for_children(
    parent_job: JobMetadata,
    child_jobs: List[JobMetadata],
    temp_dir_path: str,
    *,
    children_map: Dict[str, List[JobMetadata]],
    manifest_paths: Optional[Dict[str, str]] = None,
    disable_automl_expansion: bool = True,
) -> Optional[PipelineJob]:
    """Build a pipeline job containing ONLY direct non-pipeline child steps (leaf command jobs) of parent_job.

    Nested pipeline children are not flattened; they will be replayed as their own PipelineJobs separately
    and linked via tags. This preserves the original multi-level hierarchy.
    """
    if not child_jobs:
        return None

    # Filter only leaf (non-pipeline) children for inclusion as steps.
    effective_children: List[Dict[str, Any]] = []
    for child in child_jobs:
        jt = (child.job_type or "").lower()
        if jt == "pipeline":
            # Defer to separate replay; skip flattening
            continue
        # AutoML parents treated as normal single steps when disable_automl_expansion True
        effective_children.append({"meta": child, "role": "normal"})

    pipeline_steps_dict = {}
    for step_info in effective_children:
        step_job_meta: JobMetadata = step_info["meta"]
        metrics_to_log = step_job_meta.mlflow_metrics or {}
        metrics_json_str = json.dumps(metrics_to_log)
        temp_filename = f"metrics_{uuid.uuid4()}.json"
        temp_filepath = os.path.join(temp_dir_path, temp_filename)
        try:
            with open(temp_filepath, "w") as f:
                f.write(metrics_json_str)
        except IOError as e:
            print(f"ERROR: Failed to write temp metrics file {temp_filepath}: {e}")
            continue

        manifest_path = None
        if manifest_paths:
            manifest_path = manifest_paths.get(step_job_meta.name)
        if not manifest_path:
            # Create disabled manifest if absent
            fd, manifest_path = tempfile.mkstemp(suffix="_disabled_manifest.json")
            with os.fdopen(fd, "w", encoding="utf-8") as mf:
                json.dump(
                    {
                        "schema_version": 1,
                        "disabled": True,
                        "original_run_id": step_job_meta.name,
                        "relative_paths": [],
                    },
                    mf,
                )

        step_inputs = dict(
            original_job_id=step_job_meta.name,
            metrics_file=Input(type=AssetTypes.URI_FILE, path=temp_filepath),
            artifact_manifest=Input(type=AssetTypes.URI_FILE, path=manifest_path),
        )
        # Attach artifacts if enabled
        # (artifacts_dir temporarily disabled in component for debugging quoting issue)
        step = replay_metrics_component(**step_inputs)

        # Naming strategy
        base_display = step_job_meta.display_name or step_job_meta.name
        base_name_core = base_display

        sanitized_step_key = "".join(
            c if c.isalnum() or c in ["-", "_"] else "_" for c in base_name_core[:60]
        )
        count = 1
        final_key = sanitized_step_key
        while final_key in pipeline_steps_dict:
            final_key = f"{sanitized_step_key}_{count}"
            count += 1
        step.name = final_key
        step.display_name = f"replay_{base_display}"

        step.tags = step_job_meta.tags.copy() if step_job_meta.tags else {}
        step.tags[ORIGINAL_JOB_ID_TAG] = step_job_meta.name
        step.tags["original_parent_job_id"] = parent_job.name

        pipeline_steps_dict[final_key] = step

    if not pipeline_steps_dict:
        print("Warning: No steps added to pipeline, possibly due to file errors.")
        return None

    base_tags = {
        ORIGINAL_JOB_ID_TAG: parent_job.name,
        REPLAY_TYPE_TAG: DUMMY_REPLAY_VALUE,
        **(parent_job.tags or {}),
    }
    pipeline_job_object = PipelineJob(
        jobs=pipeline_steps_dict,
        settings=PipelineJobSettings(default_compute="serverless"),
        display_name=f"Replay of {parent_job.display_name or parent_job.name}",
        description=f"Dummy replay of pipeline {parent_job.name}. Original Desc: {parent_job.description or ''}",
        tags=base_tags,
        experiment_name=parent_job.experiment_name or "replayed_jobs",
        identity=UserIdentityConfiguration(),
    )
    return pipeline_job_object


def build_dummy_standalone_job(
    original_job: JobMetadata,
    metrics_file_path: str,
    # *,
    # artifacts_dir: Optional[str] = None,
    # upload_artifacts: bool = True,
    *,
    artifact_manifest_path: Optional[str] = None,
):
    """
    Builds a dummy CommandJob using a pre-existing metrics file path.
    """
    # Metrics file is already created by the caller

    if not artifact_manifest_path:
        fd, artifact_manifest_path = tempfile.mkstemp(suffix="_disabled_manifest.json")
        with os.fdopen(fd, "w", encoding="utf-8") as mf:
            json.dump(
                {
                    "schema_version": 1,
                    "disabled": True,
                    "original_run_id": original_job.name,
                    "relative_paths": [],
                },
                mf,
            )

    inputs = dict(
        original_job_id=original_job.name,
        metrics_file=Input(type=AssetTypes.URI_FILE, path=metrics_file_path),
        artifact_manifest=Input(type=AssetTypes.URI_FILE, path=artifact_manifest_path),
    )
    # (artifacts_dir temporarily disabled in component for debugging quoting issue)
    job = replay_metrics_component(**inputs)

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


def build_container_sas(service: BlobServiceClient, container: str, hours: int = 4):
    start = datetime.now(timezone.utc) - timedelta(minutes=5)
    expiry = datetime.now(timezone.utc) + timedelta(hours=hours)
    try:
        udk = service.get_user_delegation_key(start, expiry)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Failed to get user delegation key (need Storage Blob Data Delegator OR Data Owner on the storage account). "
            f"Underlying error: {e}"
        ) from e
    # For write scenarios we may request additional permissions later; default read/list
    perms = ContainerSasPermissions(read=True, list=True)
    sas = generate_container_sas(
        account_name=str(service.account_name),
        container_name=container,
        user_delegation_key=udk,
        permission=perms,
        expiry=expiry,
    )
    return sas


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
        exit(1)

    # --- Build indexes for nested pipeline aware replay ---
    jobs_by_name, children_map, pipeline_names = _index_jobs(all_jobs_metadata)
    depths = _compute_depths(jobs_by_name)

    # --- Infer missing parent relationships (some extracts have parent_job_name=null) ---
    inferred_links = 0
    for jm in all_jobs_metadata:
        try:
            if jm.parent_job_name:  # already set
                continue
            props = jm.job_properties or {}
            # Prefer explicit pipelinerun id, fall back to 'azureml.pipeline'
            candidate_parent = props.get("azureml.pipelinerunid") or props.get(
                "azureml.pipeline"
            )
            if (
                candidate_parent
                and candidate_parent in jobs_by_name
                and candidate_parent in pipeline_names
                and candidate_parent != jm.name
            ):
                # Set inferred parent
                jm.parent_job_name = candidate_parent  # type: ignore[attr-defined]
                inferred_links += 1
        except Exception:
            # Non-fatal; continue trying others
            continue
    if inferred_links:
        print(
            f"Inferred {inferred_links} pipeline child relationship(s) from provenance fields (azureml.pipelinerunid / azureml.pipeline)."
        )
        # Rebuild indexes & depths with updated parent assignments
        jobs_by_name, children_map, pipeline_names = _index_jobs(all_jobs_metadata)
        depths = _compute_depths(jobs_by_name)

    manifests_by_job: Dict[str, str] = {}
    source_account_name = None
    target_account_name = None
    source_container_name = None
    target_container_name = None
    source_sas = None
    target_sas = None
    if args.copy_artifacts and args.source:
        try:
            # TODO: Add tenant_id support!
            source_client = get_ml_client(args.source)
            source_datastore = source_client.datastores.get("workspaceblobstore")
            target_client_for_manifest = get_ml_client(args.target)
            target_datastore = target_client_for_manifest.datastores.get(
                "workspaceblobstore"
            )

            src_config = load_workspace_config(args.source)
            target_config = load_workspace_config(args.target)

            source_tenant_id = src_config["tenant_id"]
            target_tenant_id = target_config["tenant_id"]

            source_credential = AzureCliCredential(tenant_id=source_tenant_id)
            target_credential = AzureCliCredential(tenant_id=target_tenant_id)

            source_account_name = getattr(source_datastore, "account_name", None)
            target_account_name = getattr(target_datastore, "account_name", None)
            source_container_name = "azureml"
            target_container_name = "azureml"
            src_blob_service = BlobServiceClient(
                f"https://{source_account_name}.blob.core.windows.net",
                credential=source_credential,
            )
            tgt_blob_service = BlobServiceClient(
                f"https://{target_account_name}.blob.core.windows.net",
                credential=target_credential,
            )
            # Build SAS: source read, target write
            source_sas = build_container_sas(
                src_blob_service, source_container_name, hours=6
            )
            # Write-enabled target SAS
            start = datetime.now(timezone.utc) - timedelta(minutes=5)
            expiry = datetime.now(timezone.utc) + timedelta(hours=6)
            try:
                udk_tgt = tgt_blob_service.get_user_delegation_key(start, expiry)
                perms_tgt = ContainerSasPermissions(
                    read=True, list=True, create=True, write=True, add=True
                )
                target_sas = generate_container_sas(
                    account_name=str(tgt_blob_service.account_name),
                    container_name=target_container_name,
                    user_delegation_key=udk_tgt,
                    permission=perms_tgt,
                    expiry=expiry,
                )
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: Failed generating write-enabled SAS for target: {e}")
                target_sas = None
            print("Prepared storage context for in-run server-side artifact copy.")
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: Could not prepare storage context for manifests: {e}")
            source_account_name = None

    for jm in jobs_raw:  # jobs_raw contains dicts
        meta = JobMetadata(**jm)
        rel_paths = getattr(meta, "mlflow_artifact_paths", None) or []
        if not (
            args.copy_artifacts and source_account_name and rel_paths
        ):  # Todo: remove copy_artifacts
            fd, manifest_path = tempfile.mkstemp(
                suffix=f"_{meta.name}_manifest_disabled.json"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as mf:
                json.dump(
                    {
                        "schema_version": 1,
                        "disabled": True,
                        "original_run_id": meta.name,
                        "relative_paths": [],
                    },
                    mf,
                )
            manifests_by_job[meta.name] = manifest_path
            continue

        # rel_paths now contains the static list of default folders: ["outputs/", "system_logs/", "logs/", "user_logs/"]
        # These represent folder prefixes to copy recursively (all files and subdirectories under each)
        selected_paths = rel_paths  # Use the static folder list directly

        # Map target paths: outputs/ stays as-is (root of artifacts), logs go to outputs/original_logs/
        normalized_selected_paths = []
        for p in selected_paths:
            if p.startswith("outputs/"):
                # outputs/ content goes to root of target artifacts (strip the outputs/ prefix)
                # But we copy the entire outputs/ folder, so keep it as-is for blob copy
                normalized_selected_paths.append(p)
            elif (
                p.startswith("logs/")
                or p.startswith("system_logs/")
                or p.startswith("user_logs/")
            ):
                # All logs go under outputs/original_logs/ in target
                normalized_selected_paths.append(f"outputs/original_logs/{p}")
            else:
                normalized_selected_paths.append(p)
        fd, manifest_path = tempfile.mkstemp(
            suffix=f"_{meta.name}_artifact_manifest.json"
        )
        manifest = {
            "schema_version": 1,
            "disabled": False,
            "original_run_id": meta.name,
            "source": {
                "account": source_account_name,
                "container": source_container_name,
                "prefix": f"ExperimentRun/dcid.{meta.name}",
                "sas": source_sas,
            },
            "target": {
                "account": target_account_name,
                "container": target_container_name,
                "sas": target_sas,
            },
            "relative_paths": selected_paths,
            "normalized_relative_paths": normalized_selected_paths,
            "comment": "Static folder list: all contents under each folder copied recursively. Logs remapped to outputs/original_logs/",
        }
        with os.fdopen(fd, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf)
        manifests_by_job[meta.name] = manifest_path
    if manifests_by_job:
        print(f"Prepared {len(manifests_by_job)} artifact manifest file(s).")

    # --- Hierarchical classification ---
    # Build set of pipeline nodes (job_type==pipeline) and map their direct children.
    pipeline_children: Dict[str, List[str]] = {name: [] for name in pipeline_names}
    leaf_command_jobs: Set[str] = set()
    for jm in all_jobs_metadata:
        if jm.parent_job_name and jm.parent_job_name in pipeline_children:
            # classify as child of a pipeline
            pipeline_children[jm.parent_job_name].append(jm.name)
        jt = (jm.job_type or "").lower()
        if jt != "pipeline":
            leaf_command_jobs.add(jm.name)

    # Determine root pipelines (no parent pipeline) and nested pipelines (have parent pipeline)
    parent_pipeline_of: Dict[str, Optional[str]] = {}
    for p in pipeline_names:
        parent_name = jobs_by_name[p].parent_job_name
        if parent_name and parent_name in pipeline_names:
            parent_pipeline_of[p] = parent_name
        else:
            parent_pipeline_of[p] = None
    root_pipelines = [p for p, par in parent_pipeline_of.items() if par is None]
    # nested_pipelines list not needed directly; retained via parent_pipeline_of

    # Attach depth information for pipelines
    pipeline_depths: Dict[str, int] = {}

    def compute_pipeline_depth(p: str) -> int:
        if p in pipeline_depths:
            return pipeline_depths[p]
        parent = parent_pipeline_of.get(p)
        if not parent:
            pipeline_depths[p] = 0
            return 0
        d = 1 + compute_pipeline_depth(parent)
        pipeline_depths[p] = d
        return d

    for p in pipeline_names:
        compute_pipeline_depth(p)

    # Build replay units: process pipelines in increasing depth, then standalone command roots (those without parent and not pipelines)
    pipeline_order = sorted(
        pipeline_names, key=lambda n: (pipeline_depths.get(n, 0), n)
    )
    standalone_root_commands = [
        jm.name
        for jm in all_jobs_metadata
        if (jm.name not in pipeline_names) and not jm.parent_job_name
    ]
    standalone_root_commands = sorted(standalone_root_commands)
    replay_units: List[Tuple[str, str]] = [("pipeline", n) for n in pipeline_order] + [
        ("standalone", n) for n in standalone_root_commands
    ]
    print(
        f"Prepared {len(replay_units)} replay units (pipelines={len(pipeline_names)}, standalone_roots={len(standalone_root_commands)})."
    )
    if args.debug_hierarchy:
        print("\nHierarchy Debug Tree (pipelines only):")

        def print_tree(node: str, indent: str = ""):
            print(f"{indent}- {node} (depth={pipeline_depths[node]})")
            for ch in sorted(pipeline_children.get(node, [])):
                if ch in pipeline_names:
                    print_tree(ch, indent + "  ")

        for rp in sorted(root_pipelines):
            print_tree(rp)
        print("End Hierarchy Debug Tree\n")

    # --- Connect to Client and Register Environment ---
    try:
        client = get_ml_client(args.target)
        print(f"Connected to target workspace: {client.workspace_name}")

        # Register the environment only if not doing a dry run
        if not args.dry_run:
            from replayer.dummy_components import DUMMY_ENV

            print(
                f"Ensuring dummy environment '{DUMMY_ENV.name}:{DUMMY_ENV.version}' exists..."
            )
            try:
                env_reg = client.environments.create_or_update(DUMMY_ENV)
                print(f" -> Environment '{env_reg.name}:{env_reg.version}' is ready.")
            except Exception as e:
                print(f"❌ Error registering/updating dummy environment: {e}")
                print("Cannot proceed without the registered environment. Exiting.")
                exit(1)
    except Exception as e:
        print(f"Error connecting to target workspace: {e}")
        exit(1)

    # --- Process and Submit/Dry-Run Jobs ---
    submitted_count = 0
    failed_count = 0
    skipped_count = 0
    processed_count = 0
    job_map = {}  # original_id -> new_job_name / description

    total_units = len(replay_units)

    # We also optionally prepare for artifact copy: we only need mapping original->replay job names.
    for unit_type, job_name in replay_units:
        if args.limit is not None and processed_count >= args.limit:
            print(f"\nReached processing limit ({args.limit}). Stopping.")
            break
        processed_count += 1

        jm = jobs_by_name[job_name]
        print(
            f"\nProcessing replay unit {processed_count}/{total_units if args.limit is None else min(total_units, args.limit)}: {job_name} (type={unit_type}, depth={depths.get(job_name, 0)})"
        )

        job_to_submit: Optional[object] = None
        original_identifier = job_name
        temp_metrics_filepath_standalone = None
        temp_pipeline_dir_path = None

        # Build job object
        try:
            if unit_type == "standalone":
                original_job_metadata = jm
                print(f" -> Standalone root job: {original_job_metadata.name}")
                metrics_to_log = original_job_metadata.mlflow_metrics or {}
                metrics_json_str = json.dumps(metrics_to_log)
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                ) as temp_f:
                    temp_metrics_filepath_standalone = temp_f.name
                    temp_f.write(metrics_json_str)
                job_to_submit = build_dummy_standalone_job(
                    original_job_metadata,
                    temp_metrics_filepath_standalone,
                    artifact_manifest_path=manifests_by_job.get(
                        original_job_metadata.name
                    ),
                )
            elif unit_type == "pipeline":
                parent_meta = jm
                direct_children = [
                    jobs_by_name[n] for n in pipeline_children.get(parent_meta.name, [])
                ]
                leaf_children = [
                    c
                    for c in direct_children
                    if (c.job_type or "").lower() != "pipeline"
                ]
                print(
                    f" -> Pipeline job depth={pipeline_depths.get(parent_meta.name, 0)} with {len(leaf_children)} direct leaf step(s) and {sum(1 for c in direct_children if (c.job_type or '').lower()=='pipeline')} nested pipeline child(ren)."
                )
                if leaf_children:
                    temp_pipeline_dir_path = tempfile.mkdtemp()
                    job_to_submit = build_dummy_pipeline_for_children(
                        parent_meta,
                        leaf_children,
                        temp_pipeline_dir_path,
                        children_map=children_map,
                        manifest_paths=manifests_by_job,
                    )
                if job_to_submit is None:
                    # Fallback to standalone representation (no leaf steps)
                    metrics_to_log = parent_meta.mlflow_metrics or {}
                    metrics_json_str = json.dumps(metrics_to_log)
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".json", delete=False, encoding="utf-8"
                    ) as temp_f:
                        temp_metrics_filepath_standalone = temp_f.name
                        temp_f.write(metrics_json_str)
                    job_to_submit = build_dummy_standalone_job(
                        parent_meta,
                        temp_metrics_filepath_standalone,
                        artifact_manifest_path=manifests_by_job.get(parent_meta.name),
                    )
                # Add hierarchy tags
                if job_to_submit is not None:
                    # Ensure tags dict exists
                    if getattr(job_to_submit, "tags", None) is None:
                        try:
                            job_to_submit.tags = {}
                        except Exception:
                            pass
                    tags_dict = getattr(job_to_submit, "tags", None)
                    if tags_dict is not None and isinstance(tags_dict, dict):
                        depth_val = pipeline_depths.get(parent_meta.name, 0)
                        tags_dict[PIPELINE_DEPTH_TAG] = str(depth_val)
                        parent_pipeline = parent_pipeline_of.get(parent_meta.name)
                        if parent_pipeline:
                            tags_dict[ORIGINAL_PARENT_PIPELINE_ID_TAG] = parent_pipeline
            else:
                print(f" -> Unknown unit type '{unit_type}'. Skipping.")
                skipped_count += 1
                continue
        except Exception as build_error:
            print(
                f"   ❌ Error building job object for original {original_identifier}: {build_error}"
            )
            failed_count += 1
            job_to_submit = None

        # Submit or dry run
        if job_to_submit is None:
            print(
                f"   No valid job object was generated for {original_identifier}. Skipping submission."
            )
        else:
            if args.dry_run:
                print(f"\n--- DRY RUN for Original Unit: {original_identifier} ---")
                local_path_info = (
                    temp_metrics_filepath_standalone
                    if temp_metrics_filepath_standalone
                    else (
                        f"(Files in {temp_pipeline_dir_path})"
                        if temp_pipeline_dir_path
                        else "N/A"
                    )
                )
                print(f"  Metrics File Input Path(s) (LOCAL): {local_path_info}")
                print(f"  Would submit Job Type: {type(job_to_submit).__name__}")
                print(f"  Display Name: {job_to_submit.display_name}")
                print(f"  Experiment Name: {job_to_submit.experiment_name}")
                print(f"  Tags: {job_to_submit.tags}")
                env_id = None
                from azure.ai.ml.entities import CommandJob as _CommandJob

                if isinstance(job_to_submit, _CommandJob):
                    env_id = job_to_submit.environment
                elif isinstance(job_to_submit, PipelineJob) and job_to_submit.jobs:
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
                elif isinstance(job_to_submit, _CommandJob):
                    print(f"  Command: {job_to_submit.command}")
                    print(f"  Compute: {job_to_submit.compute}")
                print("--- END DRY RUN ---")
                submitted_count += 1
                job_map[original_identifier] = (
                    f"<Dry Run - {type(job_to_submit).__name__}>"
                )
            else:
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
                    failed_count += 1
                except Exception as submit_error:
                    print(
                        f"\n   ❌ Unexpected Error submitting job for original {original_identifier}: {submit_error}"
                    )
                    failed_count += 1

        # Cleanup
        if temp_metrics_filepath_standalone and os.path.exists(
            temp_metrics_filepath_standalone
        ):
            try:
                os.remove(temp_metrics_filepath_standalone)
            except OSError as e:
                print(
                    f"Warning: Failed to delete temp file {temp_metrics_filepath_standalone}: {e}"
                )
        if temp_pipeline_dir_path and os.path.isdir(temp_pipeline_dir_path):
            try:
                shutil.rmtree(temp_pipeline_dir_path)
            except OSError as e:
                print(
                    f"Warning: Failed to delete temp directory {temp_pipeline_dir_path}: {e}"
                )

    # --- Final Summary ---
    print("\n--- Replay Summary ---")
    print(f"Total replay units found: {total_units}")
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

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and submit replay jobs/pipelines"
    )
    parser.add_argument("--target", required=True, help="Path to target config JSON")
    parser.add_argument("--input", required=True, help="Path to extracted jobs.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform parsing and build job objects, but do not submit to Azure ML.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of original execution units (pipelines/standalone jobs) to process for testing.",
    )
    # AutoML expansion disabled; flags removed for clarity
    parser.add_argument(
        "--debug-hierarchy",
        action="store_true",
        help="Print a debug tree of the reconstructed pipeline hierarchy before submission.",
        default=False,
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to source workspace config JSON (required for --copy-artifacts to resolve original artifact locations).",
    )
    parser.add_argument(
        "--copy-artifacts",
        action="store_true",
        help="After replay submission, perform server-side copy of original artifacts into a dedicated migration prefix (no filtering).",
        default=True,
    )

    args = parser.parse_args()

    main(args)

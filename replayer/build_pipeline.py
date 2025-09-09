import argparse
import json
import logging
import os
import shutil
import tempfile
import uuid
from typing import Dict, List, Optional, Tuple, Set, Any
import random

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
    jobs_by_name: Dict[str, JobMetadata],
    expand_automl_trials: bool = False,
    replay_automl_max_trials: Optional[int] = None,
    replay_automl_top_metric: Optional[str] = None,
    replay_automl_trial_sampling: str = "best",  # best|first|random
    artifacts_dir: Optional[str] = None,
    upload_artifacts: bool = True,
    include_trial_artifacts: bool = False,
) -> Optional[PipelineJob]:
    if not child_jobs:
        return None

    def is_automl_parent(jm: JobMetadata) -> bool:
        if not jm:
            return False
        props = jm.job_properties or {}
        tags = jm.tags or {}
        jt = (jm.job_type or "").lower()
        if jt == "automl":
            return True
        if props.get("runTemplate") == "AutoML":
            return True
        if props.get("StepType") == "AutoMLStep":
            return True
        if props.get("azureml.pipelineComponent", "").startswith("masterautoml"):
            return True
        if "automl_best_child_run_id" in tags:
            return True
        return False

    def select_trials(trials: List[JobMetadata]) -> List[JobMetadata]:
        if not trials:
            return []
        primary_metric = replay_automl_top_metric
        # Infer metric if not provided
        if not primary_metric:
            for t in trials:
                if t.mlflow_metrics:
                    for k, v in t.mlflow_metrics.items():
                        if isinstance(v, (int, float)):
                            primary_metric = k
                            break
                if primary_metric:
                    break
        if replay_automl_trial_sampling == "first" or not primary_metric:
            ordered = trials
        elif replay_automl_trial_sampling == "random":
            ordered = trials[:]
            random.shuffle(ordered)
        else:  # best
            scored: List[Tuple[float, JobMetadata]] = []
            unscored: List[JobMetadata] = []
            for t in trials:
                mv = (
                    t.mlflow_metrics.get(primary_metric)
                    if t.mlflow_metrics and primary_metric in t.mlflow_metrics
                    else None
                )
                if isinstance(mv, (int, float)):
                    scored.append((float(mv), t))
                else:
                    unscored.append(t)
            if scored:
                # Assume higher is better (could parse goal later)
                scored.sort(key=lambda x: x[0], reverse=True)
                ordered = [t for _, t in scored] + unscored
            else:
                ordered = trials
        if (
            replay_automl_max_trials is not None
            and len(ordered) > replay_automl_max_trials
        ):
            print(
                f" -> AutoML trial cap: trimming from {len(ordered)} to {replay_automl_max_trials} trials."
            )
            ordered = ordered[:replay_automl_max_trials]
        return ordered

    def gather_all_descendants(root_name: str) -> List[JobMetadata]:
        """Return all descendant jobs (recursive) under a given job name."""
        collected: List[JobMetadata] = []
        stack = list(children_map.get(root_name, []))
        while stack:
            node = stack.pop()
            collected.append(node)
            stack.extend(children_map.get(node.name, []))
        return collected

    # Build structured list of effective steps with roles & metadata.
    # Each element: { 'meta': JobMetadata, 'role': 'normal'|'automl_parent'|'automl_trial',
    #                 'parent_name': <str or None>, 'rank': <int or None>,
    #                 'total_trials': <int or None>, 'selected_trials': <int or None>,
    #                 'primary_metric': <str or None>, 'is_best': <bool or None> }
    effective_children: List[Dict[str, Any]] = []
    for child in child_jobs:
        if expand_automl_trials and is_automl_parent(child):
            direct_candidates = children_map.get(child.name, [])
            deep_desc = gather_all_descendants(child.name)
            leaf_desc = [d for d in deep_desc if not children_map.get(d.name)]
            # Prefer leaves; if none (unlikely) fall back to direct children list
            trials = leaf_desc if leaf_desc else direct_candidates
            if trials:
                print(
                    f" -> Expanding AutoML parent '{child.display_name or child.name}': {len(trials)} candidate trial(s) (direct={len(direct_candidates)}, leaves={len(leaf_desc)})."
                )
                selected = select_trials(trials)
                print(
                    f"    -> Selected {len(selected)} trial(s) after sampling/limits."
                )
                total_trials = len(trials)
                selected_count = len(selected)
                # Add parent first (always retained now)
                effective_children.append(
                    {
                        "meta": child,
                        "role": "automl_parent",
                        "parent_name": parent_job.name,
                        "rank": None,
                        "total_trials": total_trials,
                        "selected_trials": selected_count,
                        "primary_metric": replay_automl_top_metric,
                        "is_best": False,
                    }
                )
                # Trials ordered as returned by select_trials (deterministic under 'best')
                for idx, tmeta in enumerate(selected, start=1):
                    effective_children.append(
                        {
                            "meta": tmeta,
                            "role": "automl_trial",
                            "parent_name": child.name,
                            "rank": idx,
                            "total_trials": total_trials,
                            "selected_trials": selected_count,
                            "primary_metric": replay_automl_top_metric,
                            "is_best": idx == 1,
                        }
                    )
                continue  # skip default append
        # Non-AutoML or not expanding
        effective_children.append(
            {
                "meta": child,
                "role": "normal",
                "parent_name": parent_job.name,
                "rank": None,
                "total_trials": None,
                "selected_trials": None,
                "primary_metric": None,
                "is_best": False,
            }
        )

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

        step_inputs = dict(
            original_job_id=step_job_meta.name,
            metrics_file=Input(type=AssetTypes.URI_FILE, path=temp_filepath),
        )
        # Attach artifacts if enabled
        if upload_artifacts and artifacts_dir:
            # Determine if this is an automl trial and whether it's included
            is_trial = step_info["role"] == "automl_trial"
            if (not is_trial) or (is_trial and include_trial_artifacts):
                run_artifacts_path = os.path.join(artifacts_dir, step_job_meta.name)
                if os.path.isdir(run_artifacts_path):
                    step_inputs["artifacts_dir"] = Input(
                        type=AssetTypes.URI_FOLDER, path=run_artifacts_path
                    )
                elif is_trial and not include_trial_artifacts:
                    pass
        step = replay_metrics_component(**step_inputs)

        # Naming strategy
        base_display = step_job_meta.display_name or step_job_meta.name
        if step_info["role"] == "automl_parent":
            base_name_core = f"automl_parent_{step_job_meta.name[:8]}"
        elif step_info["role"] == "automl_trial":
            rank = step_info.get("rank") or 0
            base_name_core = f"automl_trial_{rank:03d}_{step_job_meta.name[:8]}"
        else:
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
        if step_info["role"] == "automl_parent":
            step.display_name = f"replay_automl_parent_{base_display}"
        elif step_info["role"] == "automl_trial":
            rank = step_info.get("rank") or 0
            step.display_name = f"replay_automl_trial_{rank:03d}_{base_display}"
        else:
            step.display_name = f"replay_{base_display}"

        step.tags = step_job_meta.tags.copy() if step_job_meta.tags else {}
        step.tags[ORIGINAL_JOB_ID_TAG] = step_job_meta.name
        step.tags["original_parent_job_id"] = parent_job.name
        if step_info["role"].startswith("automl"):
            step.tags["expanded_automl_trial"] = "true"
            step.tags["automl_role"] = step_info["role"]
            if step_info["role"] == "automl_trial":
                if step_info.get("rank") is not None:
                    step.tags["automl_trial_rank"] = str(step_info["rank"])
                if step_info.get("is_best"):
                    step.tags["automl_best_trial"] = "true"
                step.tags["automl_parent_id"] = (
                    step_info.get("parent_name") or parent_job.name
                )
            elif step_info["role"] == "automl_parent":
                if step_info.get("total_trials") is not None:
                    step.tags["automl_total_trials"] = str(step_info["total_trials"])
                if step_info.get("selected_trials") is not None:
                    step.tags["automl_expanded_trials_count"] = str(
                        step_info["selected_trials"]
                    )
                if step_info.get("primary_metric"):
                    step.tags["automl_metric_primary"] = step_info["primary_metric"]

        pipeline_steps_dict[final_key] = step

    if not pipeline_steps_dict:
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
    *,
    artifacts_dir: Optional[str] = None,
    upload_artifacts: bool = True,
):
    """
    Builds a dummy CommandJob using a pre-existing metrics file path.
    """
    # Metrics file is already created by the caller

    inputs = dict(
        original_job_id=original_job.name,
        metrics_file=Input(type=AssetTypes.URI_FILE, path=metrics_file_path),
    )
    if upload_artifacts and artifacts_dir:
        run_dir = os.path.join(artifacts_dir, original_job.name)
        if os.path.isdir(run_dir):
            inputs["artifacts_dir"] = Input(type=AssetTypes.URI_FOLDER, path=run_dir)
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

    # --- Build indexes for nested pipeline aware replay ---
    jobs_by_name, children_map, pipeline_names = _index_jobs(all_jobs_metadata)
    depths = _compute_depths(jobs_by_name)

    # Define replay "units": each pipeline job (even if nested) + each standalone root command job
    standalone_root_names = [
        jm.name
        for jm in all_jobs_metadata
        if jm.name not in pipeline_names and not jm.parent_job_name
    ]

    # Promote standalone AutoML jobs to pipeline units if we want to expand trials
    if args.expand_automl_trials:
        promoted = []
        for name in standalone_root_names[:]:
            meta = jobs_by_name[name]
            jt = (meta.job_type or "").lower()
            if (
                jt == "automl"
                or (meta.job_properties or {}).get("runTemplate") == "AutoML"
            ):
                pipeline_names.add(name)
                standalone_root_names.remove(name)
                promoted.append(name)
        if promoted:
            print(
                f"Promoted {len(promoted)} standalone AutoML root job(s) to pipeline units for trial expansion: {promoted}"
            )
    # Order units by depth then name for deterministic processing (roots first)
    pipeline_unit_names = sorted(pipeline_names, key=lambda n: (depths.get(n, 0), n))
    standalone_root_names = sorted(
        standalone_root_names, key=lambda n: (depths.get(n, 0), n)
    )
    replay_units: List[Tuple[str, str]] = [
        ("pipeline", n) for n in pipeline_unit_names
    ] + [("standalone", n) for n in standalone_root_names]

    print(
        f"Prepared {len(replay_units)} replay units: {len(pipeline_unit_names)} pipeline(s) + {len(standalone_root_names)} standalone root job(s)."
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
    job_map = {}  # original_id -> new_job_name / description

    # Ensure environment ID was set (should be unless dry run skipped it, which is ok for dry run)
    if not registered_env_id_for_jobs and not args.dry_run:
        print("Error: Environment ID was not set after registration attempt. Exiting.")
        exit(1)

    total_units = len(replay_units)

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
                    artifacts_dir=args.artifacts_dir,
                    upload_artifacts=not args.no_artifacts,
                )
            elif unit_type == "pipeline":
                parent_meta = jm
                children_meta = children_map.get(parent_meta.name, [])
                if children_meta:
                    print(
                        f" -> Pipeline job with {len(children_meta)} direct child job(s)."
                    )
                    temp_pipeline_dir_path = tempfile.mkdtemp()
                    job_to_submit = build_dummy_pipeline_for_children(
                        parent_meta,
                        children_meta,
                        temp_pipeline_dir_path,
                        children_map=children_map,
                        jobs_by_name=jobs_by_name,
                        expand_automl_trials=args.expand_automl_trials,
                        replay_automl_max_trials=args.replay_automl_max_trials,
                        replay_automl_top_metric=args.replay_automl_top_metric,
                        replay_automl_trial_sampling=args.replay_automl_trial_sampling,
                        artifacts_dir=args.artifacts_dir,
                        upload_artifacts=not args.no_artifacts,
                        include_trial_artifacts=args.include_trial_artifacts,
                    )
                    if job_to_submit is None:
                        print(
                            " -> No steps generated (possibly no metrics). Falling back to standalone replay for pipeline parent."
                        )
                if not children_meta or job_to_submit is None:
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
                        artifacts_dir=args.artifacts_dir,
                        upload_artifacts=not args.no_artifacts,
                    )
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
    parser.add_argument(
        "--expand-automl-trials",
        action="store_true",
        help="Expand AutoML parent steps into individual trial steps in replay pipelines.",
    )
    parser.add_argument(
        "--replay-automl-max-trials",
        type=int,
        default=None,
        help="Cap number of AutoML trials per expanded AutoML parent during replay.",
    )
    parser.add_argument(
        "--replay-automl-top-metric",
        type=str,
        default=None,
        help="Primary metric name for ordering AutoML trials when sampling (replay). If omitted, inferred from first numeric metric.",
    )
    parser.add_argument(
        "--replay-automl-trial-sampling",
        type=str,
        choices=["best", "first", "random"],
        default="best",
        help="Sampling strategy when selecting AutoML trials for replay expansion.",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Do not upload/log artifacts for runs (artifacts are uploaded by default).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts_export",
        help="Directory root containing per-run artifact folders from extraction phase.",
    )
    parser.add_argument(
        "--include-trial-artifacts",
        action="store_true",
        help="Upload artifacts for AutoML trial runs when trials are expanded (default: only parent/non-trial).",
    )
    # Removed deprecated --keep-automl-parent-step: parent always retained when expanding trials.
    args = parser.parse_args()

    main(args)

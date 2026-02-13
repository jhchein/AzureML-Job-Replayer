"""Shared fixtures for the AzureML Job Replayer test suite."""

import json
from pathlib import Path

import pytest


@pytest.fixture()
def sample_job_metadata_dict() -> dict:
    """A dict matching JobMetadata.to_dict() output with representative fields populated."""
    return {
        "name": "happy_mango_abc123",
        "id": (
            "/subscriptions/00000000-0000-0000-0000-000000000000"
            "/resourceGroups/rg/providers/Microsoft.MachineLearningServices"
            "/workspaces/ws/jobs/happy_mango_abc123"
        ),
        "display_name": "happy_mango_abc123",
        "job_type": "command",
        "status": "Completed",
        "experiment_name": "my-experiment",
        "parent_job_name": None,
        "created_at": "2025-01-15T10:00:00+00:00",
        "start_time": "2025-01-15T10:01:00+00:00",
        "end_time": "2025-01-15T10:05:00+00:00",
        "duration_seconds": 240.0,
        "created_by": "user@example.com",
        "created_by_type": "User",
        "last_modified_at": "2025-01-15T10:05:00+00:00",
        "last_modified_by": None,
        "last_modified_by_type": None,
        "command": "python train.py",
        "script_name": "train.py",
        "environment_name": "AzureML-sklearn-1.0",
        "environment_id": "azureml:AzureML-sklearn-1.0:1",
        "environment_variables": None,
        "code_id": None,
        "arguments": None,
        "job_parameters": None,
        "compute_target": "cpu-cluster",
        "compute_id": None,
        "compute_type": None,
        "instance_count": None,
        "instance_type": None,
        "distribution": None,
        "job_limits": None,
        "job_inputs": None,
        "job_outputs": None,
        "identity_type": None,
        "services": None,
        "job_properties": None,
        "task_details": None,
        "objective": None,
        "search_space": None,
        "sampling_algorithm": None,
        "early_termination": None,
        "trial_component": None,
        "pipeline_settings": None,
        "pipeline_sub_jobs": None,
        "description": "A test job",
        "tags": {"env": "test"},
        "mlflow_run_id": "abc123",
        "mlflow_run_name": "happy_mango_abc123",
        "mlflow_experiment_id": "1",
        "mlflow_user_id": "user@example.com",
        "mlflow_artifact_uri": "azureml://artifacts/abc123",
        "mlflow_metrics": {"accuracy": 0.95, "loss": 0.05},
        "mlflow_params": {"learning_rate": "0.01"},
        "mlflow_tags": {"mlflow.source.name": "train.py"},
        "mlflow_dataset_inputs": None,
        "child_run_ids": None,
        "mlflow_artifact_paths": ["outputs/", "logs/"],
    }


@pytest.fixture()
def sample_jobs_list(sample_job_metadata_dict: dict) -> list[dict]:
    """3 jobs: one standalone, one pipeline parent, one child."""
    standalone = sample_job_metadata_dict.copy()

    parent = sample_job_metadata_dict.copy()
    parent["name"] = "pipeline_parent_001"
    parent["display_name"] = "pipeline_parent_001"
    parent["job_type"] = "pipeline"
    parent["parent_job_name"] = None

    child = sample_job_metadata_dict.copy()
    child["name"] = "child_step_001"
    child["display_name"] = "child_step_001"
    child["job_type"] = "command"
    child["parent_job_name"] = "pipeline_parent_001"

    return [standalone, parent, child]


@pytest.fixture()
def source_config_path(tmp_path: Path) -> Path:
    """Minimal valid workspace config JSON written to tmp_path."""
    cfg = {
        "subscription_id": "00000000-0000-0000-0000-000000000000",
        "resource_group": "rg-test",
        "workspace_name": "ws-test",
        "tenant_id": "11111111-1111-1111-1111-111111111111",
    }
    p = tmp_path / "source_config.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture()
def target_config_path(tmp_path: Path) -> Path:
    """Minimal valid target workspace config JSON written to tmp_path."""
    cfg = {
        "subscription_id": "22222222-2222-2222-2222-222222222222",
        "resource_group": "rg-target",
        "workspace_name": "ws-target",
    }
    p = tmp_path / "target_config.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture()
def selection_yaml_path(tmp_path: Path) -> Path:
    """Minimal valid selection YAML written to tmp_path."""
    content = """\
batch_id: test-batch-001
mltables:
  - name: my-table
    versions:
      - "1"
      - "2"
  - simple-table
source:
  subscription_id: "00000000-0000-0000-0000-000000000000"
  resource_group: rg-source
  workspace_name: ws-source
target:
  subscription_id: "11111111-1111-1111-1111-111111111111"
  resource_group: rg-target
  workspace_name: ws-target
"""
    p = tmp_path / "selection.yaml"
    p.write_text(content)
    return p

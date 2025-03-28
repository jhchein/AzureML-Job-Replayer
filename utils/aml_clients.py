import json
from pathlib import Path
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


def load_workspace_config(config_path: str) -> dict:
    """
    Load Azure ML workspace configuration from a JSON file.

    Expected JSON structure:
    {
        "subscription_id": "<Azure subscription ID>",
        "resource_group": "<Resource group name>",
        "workspace_name": "<Azure ML workspace name>"
    }
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Workspace config not found at {config_path}")
    with path.open() as f:
        return json.load(f)


def get_ml_client(config_path: str) -> MLClient:
    """
    Instantiate an MLClient for the workspace defined in the config JSON.
    """
    cfg = load_workspace_config(config_path)
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=cfg["subscription_id"],
        resource_group_name=cfg["resource_group"],
        workspace_name=cfg["workspace_name"],
    )


def get_clients(source_config: str, target_config: str) -> tuple[MLClient, MLClient]:
    """
    Return a tuple of MLClient instances (source, target) based on provided config paths.
    """
    src_client = get_ml_client(source_config)
    tgt_client = get_ml_client(target_config)
    return src_client, tgt_client

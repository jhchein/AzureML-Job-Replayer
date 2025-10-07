import json
from pathlib import Path
from threading import Lock
from typing import Optional

from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential


def load_workspace_config(config_path: str) -> dict:
    """
    Load Azure ML workspace configuration from a JSON file.

    Expected JSON structure:
    {
        "subscription_id": "<Azure subscription ID>",
        "resource_group": "<Resource group name>",
        "workspace_name": "<Azure ML workspace name>",
        "tenant_id": "<Azure tenant ID (optional)>"
    }
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Workspace config not found at {config_path}")
    with path.open() as f:
        return json.load(f)


class AzureCliCredentialPool:
    """Thread-safe cache of AzureCliCredential instances keyed by tenant."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._cache: dict[str, AzureCliCredential] = {}

    def get(self, tenant_id: Optional[str] = None) -> AzureCliCredential:
        key = tenant_id or "__default__"
        with self._lock:
            cred = self._cache.get(key)
            if cred is None:
                cred = (
                    AzureCliCredential(tenant_id=tenant_id)
                    if tenant_id
                    else AzureCliCredential()
                )
                self._cache[key] = cred
        return cred


_DEFAULT_POOL = AzureCliCredentialPool()


def get_cli_credential(
    tenant_id: Optional[str] = None,
    pool: Optional[AzureCliCredentialPool] = None,
) -> AzureCliCredential:
    """Obtain (and cache) an AzureCliCredential for the given tenant."""

    target_pool = pool or _DEFAULT_POOL
    return target_pool.get(tenant_id)


def create_ml_client(
    *,
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    tenant_id: Optional[str] = None,
    credential: Optional[AzureCliCredential] = None,
    credential_pool: Optional[AzureCliCredentialPool] = None,
) -> MLClient:
    """Create an MLClient using Azure CLI authentication with credential pooling."""

    cred = credential or get_cli_credential(tenant_id, credential_pool)
    return MLClient(
        credential=cred,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )


def get_ml_client(config_path: str) -> MLClient:
    """
    Instantiate an MLClient for the workspace defined in the config JSON.
    """
    cfg = load_workspace_config(config_path)

    tenant_id = cfg.get("tenant_id")
    if not tenant_id:
        raise ValueError(
            f"Workspace config '{config_path}' is missing 'tenant_id'. Provide the tenant so cross-tenant auth can succeed."
        )

    return create_ml_client(
        subscription_id=cfg["subscription_id"],
        resource_group=cfg["resource_group"],
        workspace_name=cfg["workspace_name"],
        tenant_id=tenant_id,
    )


def get_clients(source_config: str, target_config: str) -> tuple[MLClient, MLClient]:
    """
    Return a tuple of MLClient instances (source, target) based on provided config paths.
    """
    src_client = get_ml_client(source_config)
    tgt_client = get_ml_client(target_config)
    return src_client, tgt_client

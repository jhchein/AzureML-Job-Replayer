from azure.ai.ml import command, Input
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment

# Define a minimal environment (can be shared across all dummy steps)
# Assuming conda.yaml is correctly located relative to the project root
# when running with 'python -m'
DUMMY_ENV_NAME = "dummy-env"
DUMMY_ENV_VERSION = "1"
DUMMY_ENV_CONDA_PATH = "replayer/conda.yaml"  # Path relative to project root

DUMMY_ENV = Environment(
    name=DUMMY_ENV_NAME,
    version=DUMMY_ENV_VERSION,
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    conda_file=DUMMY_ENV_CONDA_PATH,
    description="Lightweight environment for replaying metrics.",
)

REGISTERED_ENV_ID = f"{DUMMY_ENV_NAME}:{DUMMY_ENV_VERSION}"

replay_metrics_component = command(
    name="replay_metrics",
    display_name="Replay Metrics from Original Job",
    description="Logs historical metrics into this dummy step",
    inputs={
        "original_job_id": "string",
        "metrics_file": Input(type=AssetTypes.URI_FILE),
    },
    code="./replayer/component_code",
    command='python log_metrics.py --job-id "${{inputs.original_job_id}}" --metrics-file ${{inputs.metrics_file}}',
    environment=REGISTERED_ENV_ID,
)

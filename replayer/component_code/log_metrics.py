import argparse
import json
import mlflow
import os

# Import the specific client needed
from mlflow.tracking import MlflowClient


def log_metrics(job_id: str, metrics_json: str):
    print(f"Replaying metrics for job: {job_id}")

    try:
        metrics = json.loads(metrics_json)
    except json.JSONDecodeError as e:
        print(f"Failed to parse metrics JSON: {e}")
        print(f"Received JSON string: {metrics_json}")  # Log the problematic string
        return  # Exit if JSON is invalid

    # --- Use explicit client and existing run ---
    # Azure ML jobs automatically start a run context.
    # We should log to the *current* run context provided by the job environment.
    # Do NOT call mlflow.start_run() as it creates a *nested* run which is usually not intended here.

    print("Attempting to log metrics to the current Azure ML job run.")
    try:
        # Log metrics directly using the fluent API, which uses the current active run
        for key, value in metrics.items():
            try:
                # Ensure value is floatable
                metric_value = float(value)
                mlflow.log_metric(key, metric_value)
                print(f"Logged metric: {key} = {metric_value}")
            except (ValueError, TypeError) as e:
                print(
                    f"Failed to convert or log metric '{key}' with value '{value}': {e}"
                )

        # Log the tag to the current run
        mlflow.set_tag("replayed_from_job", job_id)
        print(f"Set tag 'replayed_from_job' = {job_id}")

    except Exception as e:
        print(f"An error occurred during MLflow logging: {e}")
        # Potentially raise the error or exit differently if logging failure is critical
        # raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=str, required=True)
    parser.add_argument("--metrics-json", type=str, required=True)
    args = parser.parse_args()

    # --- Verify environment variables (optional debugging) ---
    # print("--- MLflow Environment Variables ---")
    # for var in ["MLFLOW_TRACKING_URI", "MLFLOW_RUN_ID", "MLFLOW_EXPERIMENT_ID", "MLFLOW_ARTIFACT_URI"]:
    #     print(f"{var}={os.environ.get(var)}")
    # print("---------------------------------")

    log_metrics(args.job_id, args.metrics_json)
    print("Metrics logging script finished.")

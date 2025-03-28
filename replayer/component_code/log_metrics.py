import argparse
import json
import mlflow


def log_metrics(job_id: str, metrics_filepath: str):  # Accept file path
    print(f"Replaying metrics for job: {job_id}")
    print(f"Reading metrics from file: {metrics_filepath}")

    metrics = {}
    try:
        # --- READ FROM FILE ---
        with open(metrics_filepath, "r") as f:
            metrics = json.load(f)  # Use json.load for file streams
        print(
            f"Successfully parsed metrics JSON from file. Found {len(metrics)} metrics."
        )
    except json.JSONDecodeError as e:
        print(f"Failed to parse metrics JSON from file '{metrics_filepath}': {e}")
        # Attempt to read content for debugging, handle potential read errors
        try:
            with open(metrics_filepath, "r") as f_err:
                content = f_err.read()
            print(
                f"File content received: {content[:500]}{'...' if len(content) > 500 else ''}"
            )
        except Exception as read_err:
            print(f"Could not read file content for debugging: {read_err}")
        return  # Exit if JSON is invalid
    except FileNotFoundError:
        print(f"ERROR: Metrics file not found at path: {metrics_filepath}")
        return
    except Exception as file_err:  # Catch other potential file errors
        print(f"ERROR: Could not read metrics file '{metrics_filepath}': {file_err}")
        return

    # --- Logging logic (no change needed, uses fluent API) ---
    print("Attempting to log metrics to the current Azure ML job run.")
    try:
        if not metrics:
            print("No metrics found in the parsed data to log.")

        for key, value in metrics.items():
            try:
                metric_value = float(value)
                mlflow.log_metric(key, metric_value)
                print(f"Logged metric: {key} = {metric_value}")
            except (ValueError, TypeError) as e:
                print(
                    f"Failed to convert or log metric '{key}' with value '{value}': {e}"
                )

        mlflow.set_tag("replayed_from_job", job_id)
        print(f"Set tag 'replayed_from_job' = {job_id}")

    except Exception as e:
        print(f"An error occurred during MLflow logging: {e}")
        # raise # Optional: re-raise if logging failure is critical


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=str, required=True)
    # --- UPDATE ARGUMENT NAME ---
    parser.add_argument("--metrics-file", type=str, required=True)  # Expect file path
    args = parser.parse_args()

    log_metrics(args.job_id, args.metrics_file)  # Pass file path
    print("Metrics logging script finished.")

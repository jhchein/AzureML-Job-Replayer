import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, Optional

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Job

from utils.aml_clients import get_ml_client

# TODO: Consider parallelizing or running asynchronously to speed up extraction
# TODO: Add logging and error handling
# TODO: Add TQDM for progress tracking
# TODO: Missing fields: Duration, Compute duration, script name, Created by, environment, arguments, Job YAML, compute target, compute type, instance count, metrics!!! (there must be an error in the code), description, output logs (to be stored elsewhere, contains logs, outputs, system_logs and user_logs directories), code (to be stored elsewhere)


@dataclass
class JobMetadata:
    name: str
    job_type: str
    experiment_name: Optional[str]
    parent_job_name: Optional[str]
    status: str
    created_time: Optional[str]
    metrics: Dict[str, float]
    tags: Dict[str, str]


def extract_all_jobs(client: MLClient) -> Iterator[JobMetadata]:
    """Extract all jobs from the AzureML workspace and yield them one by one."""
    for job_summary in client.jobs.list():
        job_name = job_summary.name
        job: Job = client.jobs.get(name=job_name)

        # Safely extract metrics
        metrics = {}
        try:
            metrics = job.properties.metrics or {}
        except AttributeError:
            pass

        # Safely extract creation time
        created_time = None
        try:
            if job.creation_context and hasattr(job.creation_context, "created_at"):
                time_attr = job.creation_context.created_at
                if time_attr is not None:
                    created_time = (
                        time_attr.isoformat()
                        if hasattr(time_attr, "isoformat")
                        else str(time_attr)
                    )
        except AttributeError:
            pass

        jm = JobMetadata(
            name=job.name,
            job_type=job.type,
            experiment_name=getattr(job, "experiment_name", None),
            parent_job_name=getattr(job, "parent_job_name", None),
            status=job.status,
            created_time=created_time,
            metrics=metrics,
            tags=job.tags or {},
        )
        yield jm


def main(source_config: str, output_path: str):
    client = get_ml_client(source_config)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Open file and write jobs as they're processed
    with open(output_path, "w") as f:
        f.write("[\n")

        # Track if we're on the first item to handle commas
        first_item = True
        job_count = 0

        for job in extract_all_jobs(client):
            job_count += 1
            job_json = json.dumps(asdict(job), indent=2)

            if not first_item:
                f.write(",\n")
            else:
                first_item = False

            f.write(job_json)

            # Optional: Provide progress updates
            if job_count % 10 == 0:
                print(f"Processed {job_count} jobs so far...")

        f.write("\n]")

    print(f"Extracted {job_count} jobs to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract AzureML job metadata to JSON")
    parser.add_argument(
        "--source", required=True, help="Path to source workspace config JSON"
    )
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()
    main(args.source, args.output)

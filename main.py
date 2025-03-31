import argparse
import logging
import os
import sys

from extractor.extract_jobs import main as extract_main
from replayer.build_pipeline import main as replay_main
from utils.aml_clients import get_ml_client
from utils.log_setup import setup_logging


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="AzureML Job Replayer")
    parser.add_argument(
        "--source",
        required=False,
        help="Path to source workspace config JSON",
        default="config/source_config.json",
    )
    parser.add_argument(
        "--target",
        required=False,
        help="Path to target workspace config JSON",
        default="config/target_config.json",
    )
    parser.add_argument(
        "--filter", required=False, help="Filter jobs by status or name pattern"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate extraction without submitting to target",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of original execution units to process",
    )
    parser.add_argument(
        "--output",
        help="Path to output JSON file (defaults to data/jobs.json)",
        default="data/jobs.json",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging("main")
    logger = logging.getLogger("main")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Connect to source workspace to validate config early
    try:
        source_client = get_ml_client(args.source)
        logger.info(f"Connected to source workspace: {source_client.workspace_name}")
    except Exception as e:
        logger.error(f"Failed to connect to source workspace: {e}")
        print(f"❌ Error: Failed to connect to source workspace: {e}")
        return 1

    # Connect to target workspace (even for dry run, just to validate)
    try:
        target_client = get_ml_client(args.target)
        logger.info(f"Connected to target workspace: {target_client.workspace_name}")
    except Exception as e:
        logger.error(f"Failed to connect to target workspace: {e}")
        print(f"❌ Error: Failed to connect to target workspace: {e}")
        return 1

    print(f"🔍 Source: {source_client.workspace_name}")
    print(f"🎯 Target: {target_client.workspace_name}")

    # Phase 1: Extraction
    print("\n--- EXTRACTION PHASE ---")
    # Call extraction logic, passing the source config and output path
    try:
        extract_main(args.source, args.output)
    except Exception as e:
        logger.exception(f"Extraction failed: {e}")
        print(f"❌ Error during extraction: {e}")
        return 1

    # Phase 2: Replay (if not dry run)
    if args.dry_run:
        print("\n--- DRY RUN - SKIPPING REPLAY PHASE ---")
        print(f"✅ Extracted job data saved to {args.output}")
        print("To replay these jobs, run without --dry-run flag")
    else:
        print("\n--- REPLAY PHASE ---")
        replay_args = ["--target", args.target, "--input", args.output]
        if args.limit:
            replay_args.extend(["--limit", str(args.limit)])

        # Call replay logic
        try:
            # Convert the args to the format expected by replay_main
            replay_main(replay_args)
        except Exception as e:
            logger.exception(f"Replay failed: {e}")
            print(f"❌ Error during replay: {e}")
            return 1

    print("\n✅ AzureML Job Replayer completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())

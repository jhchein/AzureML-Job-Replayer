import logging
import os
import sys
from datetime import datetime

LOG_DIR = "logs"
LOG_LEVEL_FILE = logging.INFO  # Change to DEBUG for development, INFO for production
LOG_LEVEL_CONSOLE = logging.WARNING  # Console shows only warnings and errors

# Define a flag to prevent multiple setups
_logging_configured = False


def setup_logging(log_filename_prefix="app"):
    """
    Configures file and console logging handlers for the root logger
    and the 'azure' logger. Ensures it only runs once.

    Args:
        log_filename_prefix (str): Prefix for the log file name.
    """
    global _logging_configured
    if _logging_configured:
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIR, f"{log_filename_prefix}_{timestamp}.log")

    # --- Get Root Logger ---
    root_logger = logging.getLogger()
    # Set root logger level to the lowest level needed by any handler
    root_logger.setLevel(LOG_LEVEL_FILE)  # Needs to be DEBUG to allow file handler

    # --- File Handler (DEBUG and above) ---
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(LOG_LEVEL_FILE)  # Capture DEBUG+ in file
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)-8s - %(name)-25s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # --- Console Handler (WARNING and above) ---
    console_handler = logging.StreamHandler(
        sys.stderr
    )  # <<< CHANGE HERE: Back to stderr
    console_handler.setLevel(LOG_LEVEL_CONSOLE)  # Capture WARNING+ on console
    # Use a simpler format for console
    console_formatter = logging.Formatter("%(levelname)-8s - %(name)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # --- Configure Azure Logger Specifically ---
    azure_logger = logging.getLogger("azure")
    # Set level low enough (DEBUG) so file handler can capture details
    azure_logger.setLevel(LOG_LEVEL_FILE)
    azure_logger.propagate = True  # Let messages flow to root logger's handlers

    # --- Reduce Azure HTTP logging verbosity ---
    # This suppresses the verbose HTTP request/response logs (biggest noise source)
    azure_http_logger = logging.getLogger(
        "azure.core.pipeline.policies.http_logging_policy"
    )
    azure_http_logger.setLevel(logging.WARNING)

    # --- Reduce Azure Identity logging verbosity ---
    # Suppresses repetitive "get_token succeeded" messages
    azure_identity_logger = logging.getLogger("azure.identity")
    azure_identity_logger.setLevel(logging.WARNING)

    # --- Configure MLflow Logger ---
    mlflow_logger = logging.getLogger("mlflow")
    # Keep MLflow quiet - only show warnings and errors
    mlflow_logger.setLevel(logging.WARNING)
    mlflow_logger.propagate = True

    # --- Suppress noisy third-party library DEBUG logs ---
    logging.getLogger("urllib3").setLevel(logging.INFO)
    logging.getLogger("attr_dict").setLevel(logging.INFO)
    logging.getLogger("azure.core.pipeline.policies").setLevel(logging.WARNING)
    logging.getLogger("azureml").setLevel(logging.INFO)

    # --- Final message (using print to stdout) ---
    print(
        f"Logging configured. Console level >= {logging.getLevelName(LOG_LEVEL_CONSOLE)}"
    )
    print(
        f"Detailed logs (Level >= {logging.getLevelName(LOG_LEVEL_FILE)}) in: {log_filename}"
    )

    _logging_configured = True


# --- Optional: Example Usage ---
if __name__ == "__main__":
    setup_logging("log_setup_test")
    logging.debug("This root debug message goes only to file.")
    logging.info("This root info message goes only to file.")
    logging.warning("This root warning message goes to console and file.")
    logging.getLogger("my_module").info("Info from my_module goes only to file.")
    logging.getLogger("my_module").error(
        "ERROR from my_module goes to console and file."
    )
    logging.getLogger("azure.core").debug("Azure core debug message goes only to file.")
    logging.getLogger("azure.identity").info(
        "Azure identity info message goes only to file."
    )
    logging.getLogger("azure.identity").warning(
        "Azure identity warning goes to console and file."
    )

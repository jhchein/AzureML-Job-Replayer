import json
import datetime
import sys


def log(event: str, **fields):
    """Structured JSON log line to stdout.

    Parameters
    ----------
    event: str
        Short event name
    **fields:
        Arbitrary JSON-serialisable key/value pairs
    """
    rec = {"ts": datetime.datetime.utcnow().isoformat() + "Z", "event": event, **fields}
    sys.stdout.write(json.dumps(rec) + "\n")
    sys.stdout.flush()

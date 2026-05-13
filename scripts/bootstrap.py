"""Common bootstrap for CLI scripts — call setup() before any project imports."""
import logging
import os
import sys
from pathlib import Path


def setup() -> None:
    """Add project root to sys.path, load .env, configure logging."""
    root = Path(__file__).parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")

    try:
        from dotenv import load_dotenv
        load_dotenv(root / ".env")
    except ModuleNotFoundError:
        pass

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

"""Builds a RunContext from the current environment config and request context.

Session ID is scoped to the Streamlit browser tab (persists across reruns for
the same tab). Environment metadata is loaded from config/environments/<env>.yaml
and falls back to env vars when the file is missing.
"""
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

import yaml

from .run_context import RunContext

_CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "config" / "environments"


@lru_cache(maxsize=4)
def _load_env_config(environment: str) -> Dict[str, Any]:
    config_file = _CONFIG_DIR / f"{environment}.yaml"
    if config_file.exists():
        return yaml.safe_load(config_file.read_text()) or {}
    return {}


def _streamlit_session_id() -> str:
    try:
        import streamlit as st
        if "obs_session_id" not in st.session_state:
            st.session_state.obs_session_id = str(uuid4())
        return st.session_state.obs_session_id
    except Exception:
        return os.getenv("SESSION_ID", str(uuid4()))


def make_run_context(crew_name: str = "", workflow_id: str = "") -> RunContext:
    """Create a RunContext for one crew run.

    session_id is scoped to the browser tab so all runs from one tab group in
    Langfuse. workflow_id can be set by the flows layer; defaults to run_id.
    """
    environment = os.getenv("ENVIRONMENT", "dev")
    env_cfg = _load_env_config(environment)

    session_id = _streamlit_session_id()
    user_id = os.getenv("USER_ID", "") or session_id
    run_id = str(uuid4())

    crew_versions: Dict[str, str] = env_cfg.get("crew_versions", {})

    return RunContext(
        session_id=session_id,
        run_id=run_id,
        user_id=user_id,
        environment=environment,
        app_version=env_cfg.get("app_version", os.getenv("APP_VERSION", "1.0.0")),
        crew_name=crew_name,
        deployment_sha=env_cfg.get("deployment_sha", os.getenv("DEPLOYMENT_SHA", "")),
        crew_version=crew_versions.get(crew_name, ""),
        model_version=os.getenv("MODEL_VERSION", ""),
        workflow_id=workflow_id or run_id,
    )

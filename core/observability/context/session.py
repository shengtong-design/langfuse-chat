"""Builds a RunContext from the current environment config and request context.

Session ID is scoped to the Streamlit browser tab (persists across reruns for
the same tab). Environment metadata is loaded from config/environments/<env>.yaml
and falls back to env vars when the file is missing.
"""
import os
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

import yaml

from .run_context import RunContext

_CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "config" / "environments"
_VERSION_FILE = Path(__file__).parent.parent.parent.parent / "VERSION"
_ENV_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}


def _read_app_version() -> str:
    try:
        return _VERSION_FILE.read_text().strip()
    except Exception:
        return os.getenv("APP_VERSION", "0.0.0")


def _load_env_config(environment: str) -> Dict[str, Any]:
    if environment not in _ENV_CONFIG_CACHE:
        config_file = _CONFIG_DIR / f"{environment}.yaml"
        _ENV_CONFIG_CACHE[environment] = (
            yaml.safe_load(config_file.read_text()) or {} if config_file.exists() else {}
        )
    return _ENV_CONFIG_CACHE[environment]


def _streamlit_session_id() -> str:
    try:
        import streamlit as st
        if "obs_session_id" not in st.session_state:
            st.session_state.obs_session_id = str(uuid4())
        return st.session_state.obs_session_id
    except Exception:
        return os.getenv("SESSION_ID", str(uuid4()))


def make_run_context(
    crew_name: str = "",
    crew_version: str = "",
    *,
    flow_name: str = "",
    flow_version: str = "",
) -> RunContext:
    """Create a RunContext for one crew run.

    session_id is scoped to the browser tab so all runs from one tab group in
    Langfuse. workflow_id defaults to run_id via the RunContext.workflow_id property.

    Identity is recorded on two independent axes:
      - crew_name / crew_version  → which crew recipe ran (e.g. ResearchCrew@1.0.0)
      - flow_name / flow_version  → which flow recipe orchestrated it
        (e.g. ResearchFlow@1.0.0). Same as the crew today (1:1 mapping), but
        these diverge once a flow orchestrates multiple crews, branches via
        @router, or evolves its state/post-processing independently of any crew.

    Per-run prompt resolution is recorded separately on the root span as
    agents_signature / tasks_signature.
    """
    environment = os.getenv("ENVIRONMENT", "dev")
    env_cfg = _load_env_config(environment)

    session_id = _streamlit_session_id()
    user_id = os.getenv("USER_ID", "") or session_id

    return RunContext(
        session_id=session_id,
        run_id=str(uuid4()),
        user_id=user_id,
        environment=environment,
        app_version=_read_app_version(),
        crew_name=crew_name,
        flow_name=flow_name,
        crew_version=crew_version,
        flow_version=flow_version,
        deployment_sha=env_cfg.get("deployment_sha") or os.getenv("DEPLOYMENT_SHA", ""),
        model_version=env_cfg.get("model_defaults", {}).get("default", "") or os.getenv("MODEL_VERSION", ""),
    )

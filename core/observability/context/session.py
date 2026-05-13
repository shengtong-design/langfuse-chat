"""Builds a RunContext from the current request environment.

Session ID is scoped to the Streamlit browser session (persists across reruns
for the same user tab). Falls back to a random UUID if Streamlit is unavailable.
All other fields come from environment variables so they can be set per deployment.
"""
import os
from uuid import uuid4

from .run_context import RunContext


def _streamlit_session_id() -> str:
    try:
        import streamlit as st
        if "obs_session_id" not in st.session_state:
            st.session_state.obs_session_id = str(uuid4())
        return st.session_state.obs_session_id
    except Exception:
        return os.getenv("SESSION_ID", str(uuid4()))


def make_run_context(crew_name: str = "") -> RunContext:
    """Create a RunContext for one crew run."""
    return RunContext(
        session_id=_streamlit_session_id(),
        run_id=str(uuid4()),
        user_id=os.getenv("USER_ID", ""),
        environment=os.getenv("ENVIRONMENT", "dev"),
        app_version=os.getenv("APP_VERSION", "1.0.0"),
        crew_name=crew_name,
    )

"""Map Streamlit Cloud secrets into os.environ before Settings loads."""

from __future__ import annotations

import os
from typing import Any


def _set_env(key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        os.environ.setdefault(key, "true" if value else "false")
    elif isinstance(value, (str, int, float)):
        os.environ.setdefault(key, str(value))


def apply_streamlit_secrets() -> None:
    """Load st.secrets into the environment (does not override existing env vars)."""
    try:
        import streamlit as st
    except Exception:
        return

    try:
        secrets = st.secrets
    except Exception:
        return

    _KNOWN_ALIASES = {
        ("api", "url"): "API_URL",
        ("api", "use_api"): "USE_API",
    }

    for key in secrets:
        value = secrets[key]
        if isinstance(value, dict):
            for subkey, subval in value.items():
                env_key = _KNOWN_ALIASES.get((key, subkey), f"{key}_{subkey}".upper())
                _set_env(env_key, subval)
        else:
            _set_env(str(key).upper(), value)

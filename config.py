"""Configuration lookup: environment variables, .env file, then Streamlit secrets."""

import os

from dotenv import load_dotenv

load_dotenv()


def get_setting(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        # No secrets.toml, or not running under Streamlit.
        pass
    return default


def get_int(name: str, default: int) -> int:
    try:
        return int(get_setting(name, str(default)))
    except (TypeError, ValueError):
        return default

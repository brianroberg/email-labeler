"""Configuration preprocessing utilities.

Substitutes {env.VAR_NAME} placeholders in config values with environment variables.
"""

import os
import re
from typing import Any

# Matches {env.VAR_NAME} where VAR_NAME is a valid POSIX env var name.
_ENV_PATTERN = re.compile(r"\{env\.([A-Za-z_][A-Za-z0-9_]*)\}")


def substitute_env_vars(config: dict) -> dict:
    """Recursively substitute {env.VAR_NAME} placeholders in config string values.

    Other format placeholders like {sender}, {subject}, {body} are left intact.
    Missing environment variables are replaced with empty strings.
    """
    return _walk(config)


def _walk(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _walk(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(item) for item in obj]
    if isinstance(obj, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), obj)
    return obj

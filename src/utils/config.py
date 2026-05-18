from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import re
from typing import Any, Dict

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:-|-)([^}]*))?\}")


def _expand_env_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        op = match.group(2)
        default = match.group(3)
        env_value = os.environ.get(name)
        if env_value is None or (op == ":-" and env_value == ""):
            return default if default is not None else match.group(0)
        return env_value

    expanded = _ENV_PATTERN.sub(replace, value)
    expanded = os.path.expandvars(expanded)
    return os.path.expanduser(expanded)


def expand_env_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: expand_env_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_values(v) for v in value]
    if isinstance(value, str):
        return _expand_env_string(value)
    return value


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping.")
    return expand_env_values(data)


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config_with_base(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    cfg = load_yaml(config_path)

    base_key = cfg.pop("base", None)
    if base_key is None:
        return cfg

    base_path = Path(base_key)
    if not base_path.is_absolute():
        base_path = (config_path.parent / base_key).resolve()

    base_cfg = load_config_with_base(base_path)
    return deep_merge_dict(base_cfg, cfg)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    return load_config_with_base(config_path)

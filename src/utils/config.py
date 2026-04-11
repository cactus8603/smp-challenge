from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


class ConfigNode:
    def __init__(self, data: Dict[str, Any]):
        for k, v in data.items():
            if isinstance(v, dict):
                v = ConfigNode(v)
            setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigNode):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | Path) -> ConfigNode:
    path = Path(path)
    cfg = load_yaml(path)

    if "base" in cfg:
        base_path = Path(cfg["base"])
        if not base_path.is_absolute():
            base_path = (path.parent / base_path).resolve()
        base_cfg = load_yaml(base_path)
        cfg.pop("base")
        cfg = deep_update(base_cfg, cfg)

    return ConfigNode(cfg)
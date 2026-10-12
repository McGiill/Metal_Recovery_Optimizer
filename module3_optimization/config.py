"""설정 로드 · 해시.

소유자: P4  (이 파일은 P4만 수정한다)
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """config.yaml 을 읽고, overrides 가 있으면 깊은 병합한다."""
    with open(path or DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def config_hash(cfg: dict) -> str:
    """결과 재현 추적용 짧은 해시 (meta.config_hash)."""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """'uncertainty.chance_z' 같은 점 표기 조회."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node

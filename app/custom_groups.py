import json
from pathlib import Path
from typing import Dict, List

from app.config import META_DIR

CUSTOM_GROUPS_PATH = META_DIR / "custom_groups.json"
GROUP_MASTER_PATH = META_DIR / "group_master.json"


def load_custom_groups(path: Path = CUSTOM_GROUPS_PATH) -> Dict[str, List[str]]:
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("custom_groups.json must be a dict of group name to code list")

    return {
        str(group): [str(code) for code in codes]
        for group, codes in data.items()
        if isinstance(codes, list)
    }


def save_custom_groups(groups: Dict[str, List[str]], path: Path = CUSTOM_GROUPS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        str(group): [str(code) for code in codes]
        for group, codes in groups.items()
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_group_master(path: Path = GROUP_MASTER_PATH) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("group_master.json must be a dict")

    normalized: Dict[str, Dict[str, str]] = {}
    for group_name, config in data.items():
        if not isinstance(group_name, str):
            continue
        if not isinstance(config, dict):
            config = {}
        sector_type = str(config.get("sector_type", "")).strip()
        sector_value = str(config.get("sector_value", "")).strip()
        normalized[group_name] = {
            "sector_type": sector_type,
            "sector_value": sector_value,
        }
    return normalized


def save_group_master(
    group_master: Dict[str, Dict[str, str]],
    path: Path = GROUP_MASTER_PATH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        str(group): {
            "sector_type": str(config.get("sector_type", "")).strip(),
            "sector_value": str(config.get("sector_value", "")).strip(),
        }
        for group, config in group_master.items()
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

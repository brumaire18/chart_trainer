import json
from pathlib import Path
from typing import Dict, List

from app.config import META_DIR

CUSTOM_GROUPS_PATH = META_DIR / "custom_groups.json"


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

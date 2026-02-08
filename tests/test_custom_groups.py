import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.custom_groups import load_group_master, save_group_master


class GroupMasterIoTest(unittest.TestCase):
    def test_save_and_load_group_master(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "group_master.json"
            payload = {
                "自動車": {"sector_type": "17業種", "sector_value": "輸送用機器"},
                "半導体": {"sector_type": "33業種", "sector_value": "電気機器"},
            }
            save_group_master(payload, path=path)
            loaded = load_group_master(path=path)
            self.assertEqual(loaded, payload)

    def test_load_group_master_invalid_shape_raises(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "group_master.json"
            path.write_text(json.dumps(["invalid"]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_group_master(path=path)


if __name__ == "__main__":
    unittest.main()

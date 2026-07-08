"""Tests for shared evals helpers."""

import json

from evals import atomic_write_jsonl, plural


class _Rec:
    def __init__(self, value):
        self.value = value

    def to_dict(self):
        return {"value": self.value}


class TestAtomicWriteJsonl:
    def test_writes_one_json_line_per_record(self, tmp_path):
        path = tmp_path / "out.jsonl"
        atomic_write_jsonl([_Rec(1), _Rec(2)], path)
        lines = path.read_text().strip().splitlines()
        assert [json.loads(line) for line in lines] == [{"value": 1}, {"value": 2}]

    def test_accepts_str_path(self, tmp_path):
        path = tmp_path / "out.jsonl"
        atomic_write_jsonl([_Rec(1)], str(path))  # str coerced to Path
        assert path.exists()

    def test_overwrites_existing_file(self, tmp_path):
        path = tmp_path / "out.jsonl"
        atomic_write_jsonl([_Rec(1)], path)
        atomic_write_jsonl([_Rec(2)], path)
        assert path.read_text().strip() == json.dumps({"value": 2})

    def test_leaves_no_temp_file_on_success(self, tmp_path):
        path = tmp_path / "out.jsonl"
        atomic_write_jsonl([_Rec(1)], path)
        assert [p.name for p in tmp_path.iterdir()] == ["out.jsonl"]


class TestPlural:
    def test_singular(self):
        assert plural(1, "story", "stories") == "1 story"

    def test_plural(self):
        assert plural(2, "story", "stories") == "2 stories"

    def test_zero_is_plural(self):
        assert plural(0, "error", "errors") == "0 errors"

    def test_default_plural_appends_s(self):
        assert plural(3, "row") == "3 rows"
        assert plural(1, "row") == "1 row"

"""Phase 1 tests for mltable_yaml_migrator.selection.SelectionSpec."""

from pathlib import Path

import pytest

from mltable_yaml_migrator.selection import SelectionSpec


class TestSelectionSpecLoad:
    def test_valid_yaml(self, selection_yaml_path: Path):
        spec = SelectionSpec.load(selection_yaml_path)
        assert spec.batch_id == "test-batch-001"
        assert len(spec.mltables) == 2
        # First entry: dict with name + versions
        assert spec.mltables[0].name == "my-table"
        assert spec.mltables[0].versions == ["1", "2"]
        # Second entry: bare string -> versions=None
        assert spec.mltables[1].name == "simple-table"
        assert spec.mltables[1].versions is None

    def test_source_and_target_parsed(self, selection_yaml_path: Path):
        spec = SelectionSpec.load(selection_yaml_path)
        assert spec.source is not None
        assert spec.source.subscription_id == "00000000-0000-0000-0000-000000000000"
        assert spec.source.workspace_name == "ws-source"
        assert spec.target is not None
        assert spec.target.workspace_name == "ws-target"

    def test_missing_source_target_ok(self, tmp_path: Path):
        p = tmp_path / "minimal.yaml"
        p.write_text("batch_id: b1\nmltables:\n  - name: t1\n")
        spec = SelectionSpec.load(p)
        assert spec.batch_id == "b1"
        assert spec.source is None
        assert spec.target is None

    def test_fallback_batch_id(self, tmp_path: Path):
        p = tmp_path / "no_batch_id.yaml"
        p.write_text("mltables:\n  - name: t1\n")
        spec = SelectionSpec.load(p)
        assert spec.batch_id == "batch-unknown"

    def test_empty_mltables_list(self, tmp_path: Path):
        p = tmp_path / "empty_tables.yaml"
        p.write_text("batch_id: b1\nmltables: []\n")
        spec = SelectionSpec.load(p)
        assert spec.mltables == []

    def test_invalid_mltable_entry_raises(self, tmp_path: Path):
        p = tmp_path / "bad.yaml"
        p.write_text("batch_id: b1\nmltables:\n  - 42\n")
        with pytest.raises(ValueError, match="Unsupported mltables entry"):
            SelectionSpec.load(p)

    def test_missing_workspace_keys_raises(self, tmp_path: Path):
        p = tmp_path / "bad_source.yaml"
        p.write_text("batch_id: b1\nmltables: []\nsource:\n  subscription_id: s1\n")
        with pytest.raises(ValueError, match="Missing keys in source"):
            SelectionSpec.load(p)

    def test_requested_versions(self, selection_yaml_path: Path):
        spec = SelectionSpec.load(selection_yaml_path)
        assert spec.requested_versions("my-table") == ["1", "2"]
        assert spec.requested_versions("simple-table") is None
        assert spec.requested_versions("nonexistent") is None

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            SelectionSpec.load(tmp_path / "missing.yaml")

"""Phase 1 tests for utils.aml_clients.load_workspace_config."""

import json
from pathlib import Path

import pytest

from utils.aml_clients import load_workspace_config


class TestLoadWorkspaceConfig:
    def test_valid_config(self, source_config_path: Path):
        cfg = load_workspace_config(str(source_config_path))
        assert cfg["subscription_id"] == "00000000-0000-0000-0000-000000000000"
        assert cfg["resource_group"] == "rg-test"
        assert cfg["workspace_name"] == "ws-test"
        assert cfg["tenant_id"] == "11111111-1111-1111-1111-111111111111"

    def test_config_without_tenant(self, target_config_path: Path):
        cfg = load_workspace_config(str(target_config_path))
        assert "tenant_id" not in cfg
        assert cfg["workspace_name"] == "ws-target"

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_workspace_config(str(tmp_path / "nonexistent.json"))

    def test_invalid_json(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json at all")
        with pytest.raises(json.JSONDecodeError):
            load_workspace_config(str(bad))

    def test_empty_json_object(self, tmp_path: Path):
        empty = tmp_path / "empty.json"
        empty.write_text("{}")
        cfg = load_workspace_config(str(empty))
        assert cfg == {}

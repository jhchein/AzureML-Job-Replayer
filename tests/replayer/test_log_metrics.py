"""Tests for the replay step's artifact download helpers."""

import json
from unittest.mock import MagicMock, patch

import pytest

from replayer.component_code import log_metrics as lm


class TestLocalDest:
    def test_outputs_prefix_stripped(self):
        assert lm._local_dest("outputs/model/a.pkl") == "model/a.pkl"

    @pytest.mark.parametrize("folder", ["logs", "system_logs", "user_logs"])
    def test_log_folders_remapped(self, folder):
        assert lm._local_dest(f"{folder}/x.log") == f"original_logs/{folder}/x.log"

    def test_other_paths_unchanged(self):
        assert lm._local_dest("misc/y.txt") == "misc/y.txt"


def _sas_manifest(blobs):
    return {
        "access": "sas",
        "source": {"account": "acct", "container": "c", "prefix": "ExperimentRun/dcid.j"},
        "blobs": blobs,
    }


class TestManifestBlobClients:
    def test_sas_mode_builds_one_client_per_blob(self):
        manifest = _sas_manifest(
            [{"name": "ExperimentRun/dcid.j/outputs/a.txt", "size": 1, "sas": "sig=1"}]
        )
        with patch.object(lm, "BlobClient") as blob_client:
            result = lm._manifest_blob_clients(manifest)
        assert [name for name, _ in result] == ["ExperimentRun/dcid.j/outputs/a.txt"]
        blob_client.assert_called_once_with(
            "https://acct.blob.core.windows.net",
            "c",
            "ExperimentRun/dcid.j/outputs/a.txt",
            credential="sig=1",
        )

    def test_sas_mode_without_blobs_raises(self):
        manifest = _sas_manifest([])
        del manifest["blobs"]
        with pytest.raises(RuntimeError, match="blobs"):
            lm._manifest_blob_clients(manifest)

    def test_missing_source_fields_raise(self):
        with pytest.raises(RuntimeError, match="source"):
            lm._manifest_blob_clients({"access": "sas", "source": {}})

    def test_rbac_mode_lists_folders_with_trailing_slash(self):
        container_client = MagicMock()
        container_client.list_blobs.return_value = [MagicMock()]
        container_client.list_blobs.return_value[0].name = "ExperimentRun/dcid.j/outputs/a.txt"
        manifest = {
            "access": "rbac",
            "source": {"account": "acct", "container": "c", "prefix": "ExperimentRun/dcid.j"},
            "relative_paths": ["outputs/"],
        }
        with patch.object(lm, "ContainerClient", return_value=container_client), patch.object(
            lm, "_rbac_credential"
        ):
            result = lm._manifest_blob_clients(manifest)
        container_client.list_blobs.assert_called_once_with(
            name_starts_with="ExperimentRun/dcid.j/outputs/"
        )
        assert len(result) == 1


def _client_with_content(data: bytes):
    client = MagicMock()
    client.download_blob.return_value.chunks.return_value = [data]
    return client


class TestDownloadArtifacts:
    def test_downloads_into_outputs_and_writes_summary(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        items = [
            ("ExperimentRun/dcid.j/outputs/model/a.txt", _client_with_content(b"abc")),
            ("ExperimentRun/dcid.j/logs/azureml.log", _client_with_content(b"log")),
        ]
        manifest = _sas_manifest([])
        with patch.object(lm, "_manifest_blob_clients", return_value=items):
            lm._download_artifacts(manifest)
        assert (tmp_path / "outputs/model/a.txt").read_bytes() == b"abc"
        assert (tmp_path / "outputs/original_logs/logs/azureml.log").read_bytes() == b"log"
        summary = json.loads((tmp_path / "outputs/_replay_download_summary.json").read_text())
        assert summary["success"] == 2 and summary["failed"] == 0

    def test_path_traversal_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        items = [
            ("ExperimentRun/dcid.j/outputs/../../evil.txt", _client_with_content(b"x")),
        ]
        with patch.object(lm, "_manifest_blob_clients", return_value=items):
            with pytest.raises(RuntimeError, match="downloads failed"):
                lm._download_artifacts(_sas_manifest([]))
        assert not (tmp_path / "evil.txt").exists()
        assert not list(tmp_path.rglob("evil.txt"))
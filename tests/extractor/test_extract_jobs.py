"""Tests for pure functions and helpers in extractor.extract_jobs."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from mlflow.exceptions import MlflowException

from extractor.extract_jobs import (
    JobMetadata,
    MlflowData,
    _extract_creation_context,
    _extract_environment,
    _fetch_mlflow_data,
    _get_default_artifact_folders,
    _is_transient,
    _parse_include_args,
    _resolve_timestamps,
    convert_complex_dict,
    safe_duration,
    safe_isoformat,
)

# ── safe_isoformat ──────────────────────────────────────────────────


class TestSafeIsoformat:
    def test_none_returns_none(self):
        assert safe_isoformat(None) is None

    def test_naive_datetime_gets_utc(self):
        dt = datetime(2025, 1, 15, 10, 0, 0)
        result = safe_isoformat(dt)
        assert result is not None
        assert result.endswith("+00:00")
        assert "2025-01-15T10:00:00" in result

    def test_aware_datetime_preserved(self):
        dt = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = safe_isoformat(dt)
        assert result == "2025-01-15T10:00:00+00:00"

    def test_iso_string_roundtrip(self):
        iso = "2025-01-15T10:00:00+00:00"
        result = safe_isoformat(iso)
        assert result is not None
        assert "2025-01-15" in result

    def test_iso_string_with_z(self):
        result = safe_isoformat("2025-06-01T12:00:00Z")
        assert result is not None
        assert "2025-06-01" in result

    def test_epoch_seconds(self):
        # 2025-01-15T10:00:00 UTC
        ts = 1736935200.0
        result = safe_isoformat(ts)
        assert result is not None
        assert "+00:00" in result

    def test_epoch_milliseconds(self):
        ts = 1736935200000.0  # > 1e12 -> treated as millis
        result = safe_isoformat(ts)
        assert result is not None
        assert "+00:00" in result

    def test_unparseable_returns_str(self):
        result = safe_isoformat("not-a-date")
        # Falls back to str(obj) rather than raising
        assert isinstance(result, str)

    def test_non_datetime_object_returns_str(self):
        result = safe_isoformat(object())
        assert isinstance(result, str)


# ── safe_duration ───────────────────────────────────────────────────


class TestSafeDuration:
    def test_both_none(self):
        assert safe_duration(None, None) is None

    def test_start_none(self):
        assert safe_duration(None, "2025-01-15T10:05:00+00:00") is None

    def test_end_none(self):
        assert safe_duration("2025-01-15T10:00:00+00:00", None) is None

    def test_valid_iso_pair(self):
        result = safe_duration("2025-01-15T10:00:00+00:00", "2025-01-15T10:04:00+00:00")
        assert result == pytest.approx(240.0)

    def test_end_before_start_returns_none(self):
        result = safe_duration("2025-01-15T10:05:00+00:00", "2025-01-15T10:00:00+00:00")
        assert result is None

    def test_epoch_seconds_pair(self):
        start = 1736935200.0  # some UTC epoch
        end = start + 120.0
        result = safe_duration(start, end)
        assert result == pytest.approx(120.0)

    def test_epoch_millis_pair(self):
        start = 1736935200000.0
        end = start + 60000.0  # 60 s
        result = safe_duration(start, end)
        assert result == pytest.approx(60.0)

    def test_datetime_objects(self):
        s = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        e = datetime(2025, 1, 15, 10, 2, 0, tzinfo=timezone.utc)
        assert safe_duration(s, e) == pytest.approx(120.0)

    def test_mixed_types(self):
        # ISO string start, datetime end
        s = "2025-01-15T10:00:00+00:00"
        e = datetime(2025, 1, 15, 10, 1, 0, tzinfo=timezone.utc)
        assert safe_duration(s, e) == pytest.approx(60.0)

    def test_bad_format_returns_none(self):
        assert safe_duration("not-a-date", "also-not-a-date") is None


# ── convert_complex_dict ────────────────────────────────────────────


class TestConvertComplexDict:
    def test_empty_dict(self):
        assert convert_complex_dict({}) == {}

    def test_primitives_pass_through(self):
        assert convert_complex_dict("hello") == "hello"
        assert convert_complex_dict(42) == 42
        assert convert_complex_dict(3.14) == 3.14
        assert convert_complex_dict(True) is True
        assert convert_complex_dict(None) is None

    def test_nested_dicts(self):
        inp = {"a": {"b": {"c": 1}}}
        assert convert_complex_dict(inp) == {"a": {"b": {"c": 1}}}

    def test_list_of_dicts(self):
        inp = [{"x": 1}, {"y": 2}]
        assert convert_complex_dict(inp) == [{"x": 1}, {"y": 2}]

    def test_tuple_becomes_list(self):
        inp = (1, 2, 3)
        assert convert_complex_dict(inp) == [1, 2, 3]

    def test_non_string_dict_key(self):
        # Non-string key gets str()'d
        inp = {123: "value"}
        result = convert_complex_dict(inp)
        assert result == {123: "value"}  # int keys are allowed per code

    def test_datetime_converts_to_iso(self):
        dt = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = convert_complex_dict(dt)
        assert isinstance(result, str)
        assert "2025-06-01" in result

    def test_bytes_decode(self):
        assert convert_complex_dict(b"hello") == "hello"

    def test_object_with_to_dict(self):
        class Fake:
            def _to_dict(self):
                return {"key": "val"}

        result = convert_complex_dict(Fake())
        assert result == {"key": "val"}

    def test_object_with_dict_attr(self):
        class Obj:
            def __init__(self):
                self.x = 1
                self.y = 2
                self._private = 3

        result = convert_complex_dict(Obj())
        assert result == {"x": 1, "y": 2}

    def test_unserializable_long_str_truncated(self):
        class Weird:
            # No __dict__, no _to_dict, no asdict — force str() path
            __slots__ = ()

            def __str__(self):
                return "A" * 300

        result = convert_complex_dict(Weird())
        assert len(result) <= 200
        assert result.endswith("...")


# ── _is_transient ───────────────────────────────────────────────────


class TestIsTransient:
    @pytest.mark.parametrize(
        "msg",
        [
            "429 Too Many Requests",
            "503 Service Unavailable",
            "502 Bad Gateway",
            "Request timed out",
            "Connection reset by peer",
            "temporarily unavailable",
            "rate limit exceeded",
            "timeout waiting for response",
        ],
    )
    def test_transient_errors(self, msg):
        assert _is_transient(Exception(msg)) is True

    @pytest.mark.parametrize(
        "msg",
        [
            "404 Not Found",
            "401 Unauthorized",
            "ValueError: bad input",
            "KeyError: missing",
        ],
    )
    def test_non_transient_errors(self, msg):
        assert _is_transient(Exception(msg)) is False

    def test_empty_message(self):
        assert _is_transient(Exception("")) is False


# ── _get_default_artifact_folders ───────────────────────────────────


class TestGetDefaultArtifactFolders:
    def test_returns_list(self):
        result = _get_default_artifact_folders()
        assert isinstance(result, list)

    def test_contains_standard_folders(self):
        result = _get_default_artifact_folders()
        assert "outputs/" in result
        assert "logs/" in result

    def test_all_entries_end_with_slash(self):
        for folder in _get_default_artifact_folders():
            assert folder.endswith("/"), f"Expected trailing slash: {folder}"


# ── _parse_include_args ─────────────────────────────────────────────


class TestParseIncludeArgs:
    def test_no_include_returns_none(self):
        args = SimpleNamespace(include=None, include_file=None)
        assert _parse_include_args(args) is None

    def test_empty_include_returns_none(self):
        args = SimpleNamespace(include="", include_file=None)
        assert _parse_include_args(args) is None

    def test_comma_separated_names(self):
        args = SimpleNamespace(include="job1,job2, job3 ", include_file=None)
        result = _parse_include_args(args)
        assert result == {"job1", "job2", "job3"}

    def test_single_name(self):
        args = SimpleNamespace(include="only_one", include_file=None)
        result = _parse_include_args(args)
        assert result == {"only_one"}

    def test_include_file(self, tmp_path):
        f = tmp_path / "names.txt"
        f.write_text("alpha\nbeta\n\ngamma\n")
        args = SimpleNamespace(include=None, include_file=str(f))
        result = _parse_include_args(args)
        assert result == {"alpha", "beta", "gamma"}

    def test_include_and_file_combined(self, tmp_path):
        f = tmp_path / "names.txt"
        f.write_text("file_job\n")
        args = SimpleNamespace(include="cli_job", include_file=str(f))
        result = _parse_include_args(args)
        assert result == {"cli_job", "file_job"}

    def test_missing_include_file_returns_cli_names(self, tmp_path):
        # File doesn't exist: the function prints a warning and continues
        args = SimpleNamespace(
            include="safe_job", include_file=str(tmp_path / "missing.txt")
        )
        result = _parse_include_args(args)
        assert result == {"safe_job"}

    def test_file_path_in_include_warns(self, tmp_path):
        """When --include looks like a file path, function still parses it as a name."""
        f = tmp_path / "names.txt"
        f.write_text("job1\n")
        args = SimpleNamespace(include=str(f), include_file=None)
        # It should still work (logs a warning), returns the path string as a "name"
        result = _parse_include_args(args)
        assert result is not None
        assert str(f) in result


# ── JobMetadata.to_dict ─────────────────────────────────────────────


class TestJobMetadataToDict:
    def test_all_fields_present(self):
        jm = JobMetadata(name="test_job")
        d = jm.to_dict()
        # Every dataclass field should have a key
        from dataclasses import fields as dc_fields

        for f in dc_fields(JobMetadata):
            assert f.name in d, f"Missing key: {f.name}"

    def test_none_defaults_preserved(self):
        jm = JobMetadata(name="test_job")
        d = jm.to_dict()
        assert d["name"] == "test_job"
        assert d["id"] is None
        assert d["mlflow_metrics"] is None

    def test_dict_values_serialised(self):
        jm = JobMetadata(
            name="test_job",
            mlflow_metrics={"acc": 0.9},
            tags={"env": "prod"},
        )
        d = jm.to_dict()
        assert d["mlflow_metrics"] == {"acc": 0.9}
        assert d["tags"] == {"env": "prod"}

    def test_roundtrip_from_fixture(self, sample_job_metadata_dict):
        """Construct from fixture dict, serialise back, verify key fields match."""
        jm = JobMetadata(**sample_job_metadata_dict)
        d = jm.to_dict()
        assert d["name"] == sample_job_metadata_dict["name"]
        assert d["job_type"] == sample_job_metadata_dict["job_type"]
        assert d["mlflow_metrics"] == sample_job_metadata_dict["mlflow_metrics"]

    def test_non_string_dict_keys_converted(self):
        jm = JobMetadata(name="test_job", tags={123: "val"})  # type: ignore[dict-item]
        d = jm.to_dict()
        # to_dict converts keys via str()
        assert "123" in d["tags"]


# ── _extract_environment ────────────────────────────────────────────


class TestExtractEnvironment:
    def test_no_environment(self):
        job = SimpleNamespace(environment=None)
        assert _extract_environment(job) == (None, None)

    def test_string_env_with_colon(self):
        # When the string has no '/', the whole string is treated as the name part
        job = SimpleNamespace(environment="azureml:AzureML-sklearn:1")
        name, eid = _extract_environment(job)
        # name_part = "azureml:AzureML-sklearn:1" (has ":" but no "@")
        assert name == "azureml:AzureML-sklearn:1"
        assert eid == "azureml:AzureML-sklearn:1"

    def test_string_env_with_colon_and_at(self):
        job = SimpleNamespace(
            environment="azureml://registries/azureml/environments/AzureML-sklearn:1@latest"
        )
        name, eid = _extract_environment(job)
        assert name == "AzureML-sklearn:1"
        assert eid is not None

    def test_string_env_simple_name(self):
        job = SimpleNamespace(environment="my-env")
        name, eid = _extract_environment(job)
        assert name == "my-env"
        assert eid == "my-env"

    def test_object_env_with_name_and_id(self):
        env_obj = SimpleNamespace(name="env-name", id="env-id-123")
        job = SimpleNamespace(environment=env_obj)
        name, eid = _extract_environment(job)
        assert name == "env-name"
        assert eid == "env-id-123"

    def test_object_env_with_name_only(self):
        env_obj = SimpleNamespace(name="env-name")
        job = SimpleNamespace(environment=env_obj)
        name, eid = _extract_environment(job)
        assert name == "env-name"
        assert eid is None


# ── _extract_creation_context ───────────────────────────────────────


class TestExtractCreationContext:
    def test_no_context(self):
        job = SimpleNamespace()
        result = _extract_creation_context(job)
        assert result["created_at"] is None
        assert result["created_by"] is None
        assert len(result) == 6

    def test_full_context(self):
        ctx = SimpleNamespace(
            created_at=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            created_by="user@example.com",
            created_by_type="User",
            last_modified_at=datetime(2025, 1, 15, 10, 5, 0, tzinfo=timezone.utc),
            last_modified_by="admin@example.com",
            last_modified_by_type="User",
        )
        job = SimpleNamespace(creation_context=ctx)
        result = _extract_creation_context(job)
        assert "2025-01-15" in result["created_at"]
        assert result["created_by"] == "user@example.com"
        assert result["last_modified_by"] == "admin@example.com"

    def test_partial_context(self):
        ctx = SimpleNamespace(
            created_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
            created_by="a@b.com",
        )
        job = SimpleNamespace(creation_context=ctx)
        result = _extract_creation_context(job)
        assert result["created_by"] == "a@b.com"
        # Missing attrs default to None via getattr
        assert result["created_by_type"] is None


# ── _resolve_timestamps ─────────────────────────────────────────────


class TestResolveTimestamps:
    def test_all_none(self):
        start, end, dur = _resolve_timestamps(None, None, None)
        assert start is None
        assert end is None
        assert dur is None

    def test_mlflow_ms_preferred(self):
        start, end, dur = _resolve_timestamps(
            1736935200000,
            1736935260000,
            {"StartTimeUtc": "should-not-use", "EndTimeUtc": "should-not-use"},
        )
        assert start is not None
        assert end is not None
        assert dur == pytest.approx(60.0)

    def test_fallback_to_properties(self):
        start, end, dur = _resolve_timestamps(
            None,
            None,
            {
                "StartTimeUtc": "2025-01-15T10:00:00Z",
                "EndTimeUtc": "2025-01-15T10:02:00Z",
            },
        )
        assert start is not None
        assert dur == pytest.approx(120.0)

    def test_partial_mlflow(self):
        """MLflow has start but not end; end falls back to properties."""
        start, end, dur = _resolve_timestamps(
            1736935200000,
            None,
            {"EndTimeUtc": "2025-01-15T12:00:00+00:00"},
        )
        assert start is not None
        assert end is not None


# ── _fetch_mlflow_data ──────────────────────────────────────────────


class TestFetchMlflowData:
    def _make_mock_client(self, *, has_data=True, has_info=True, has_inputs=False):
        client = MagicMock()
        run = MagicMock()
        if has_data:
            run.data.metrics = {"acc": 0.9}
            run.data.params = {"lr": "0.01"}
            run.data.tags = {"mlflow.source.name": "train.py"}
        else:
            run.data = None
        if has_info:
            run.info.run_id = "run-123"
            run.info.run_name = "happy_run"
            run.info.experiment_id = "exp-1"
            run.info.user_id = "user@example.com"
            run.info.artifact_uri = "azureml://artifacts/run-123"
            run.info.start_time = 1736935200000
            run.info.end_time = 1736935260000
        else:
            run.info = None
        if has_inputs:
            run.inputs.dataset_inputs = [{"name": "ds1"}]
        else:
            # Remove inputs attr entirely
            del run.inputs
        client.get_run.return_value = run
        return client

    def test_full_data(self):
        client = self._make_mock_client()
        result = _fetch_mlflow_data(client, "job1")
        assert isinstance(result, MlflowData)
        assert result.metrics == {"acc": 0.9}
        assert result.run_id == "run-123"
        assert result.script_name == "train.py"
        assert result.start_time_ms == 1736935200000

    def test_no_data_section(self):
        client = self._make_mock_client(has_data=False)
        result = _fetch_mlflow_data(client, "job2")
        assert result.metrics == {}
        assert result.script_name is None

    def test_resource_not_found(self):
        client = MagicMock()
        client.get_run.side_effect = MlflowException("RESOURCE_DOES_NOT_EXIST")
        result = _fetch_mlflow_data(client, "missing_job")
        assert result.metrics == {}
        assert result.run_id is None

    def test_unexpected_error(self):
        client = MagicMock()
        client.get_run.side_effect = RuntimeError("network failure")
        result = _fetch_mlflow_data(client, "fail_job")
        assert result.metrics == {}
        assert result.run_id is None

    def test_with_dataset_inputs(self):
        client = self._make_mock_client(has_inputs=True)
        result = _fetch_mlflow_data(client, "job_with_ds")
        assert result.dataset_inputs is not None

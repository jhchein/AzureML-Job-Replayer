"""Phase 1 tests for pure functions in extractor.extract_jobs."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from extractor.extract_jobs import (
    JobMetadata,
    _get_default_artifact_folders,
    _is_transient,
    _parse_include_args,
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

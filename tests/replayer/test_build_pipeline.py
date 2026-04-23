"""Tests for pure functions in replayer.build_pipeline."""

import json
from unittest.mock import MagicMock

from azure.core.exceptions import ResourceNotFoundError

from extractor.extract_jobs import JobMetadata
from replayer.build_pipeline import (
    _classify_replay_units,
    _compute_depths,
    _create_disabled_manifest,
    _index_jobs,
    _infer_parent_links,
    _load_manifest,
    _resolve_source_container,
)

# ── helpers ─────────────────────────────────────────────────────────


def _jm(
    name: str, *, job_type: str = "command", parent: str | None = None
) -> JobMetadata:
    """Shortcut to construct a minimal JobMetadata for indexing tests."""
    return JobMetadata(name=name, job_type=job_type, parent_job_name=parent)


# ── _index_jobs ─────────────────────────────────────────────────────


class TestIndexJobs:
    def test_empty_list(self):
        by_name, children, pipelines = _index_jobs([])
        assert by_name == {}
        assert children == {}
        assert pipelines == set()

    def test_single_standalone_command(self):
        jm = _jm("job1")
        by_name, children, pipelines = _index_jobs([jm])
        assert "job1" in by_name
        assert children == {}
        assert pipelines == set()

    def test_pipeline_detected(self):
        jm = _jm("pipe1", job_type="pipeline")
        _, _, pipelines = _index_jobs([jm])
        assert "pipe1" in pipelines

    def test_children_mapped(self):
        parent = _jm("parent1", job_type="pipeline")
        child1 = _jm("child1", parent="parent1")
        child2 = _jm("child2", parent="parent1")
        _, children, _ = _index_jobs([parent, child1, child2])
        assert len(children["parent1"]) == 2
        child_names = {c.name for c in children["parent1"]}
        assert child_names == {"child1", "child2"}

    def test_orphan_child_not_in_children_map(self):
        """Child referencing a parent not in the list is ignored in children_map."""
        orphan = _jm("orphan", parent="missing_parent")
        _, children, _ = _index_jobs([orphan])
        assert children == {}

    def test_last_write_wins_for_duplicate_names(self):
        jm1 = _jm("dup", job_type="command")
        jm2 = _jm("dup", job_type="pipeline")
        by_name, _, pipelines = _index_jobs([jm1, jm2])
        # Second entry overwrites first in jobs_by_name
        assert by_name["dup"].job_type == "pipeline"
        assert "dup" in pipelines

    def test_mixed_hierarchy(self, sample_jobs_list):
        """Uses the conftest fixture: standalone, pipeline parent, child."""
        jobs = [JobMetadata(**d) for d in sample_jobs_list]
        by_name, children, pipelines = _index_jobs(jobs)
        assert len(by_name) == 3
        assert "pipeline_parent_001" in pipelines
        assert len(children.get("pipeline_parent_001", [])) == 1


# ── _compute_depths ─────────────────────────────────────────────────


class TestComputeDepths:
    def test_flat_jobs(self):
        jobs = [_jm("a"), _jm("b")]
        by_name = {j.name: j for j in jobs}
        depths = _compute_depths(by_name)
        assert depths == {"a": 0, "b": 0}

    def test_single_parent_child(self):
        parent = _jm("p", job_type="pipeline")
        child = _jm("c", parent="p")
        by_name = {j.name: j for j in [parent, child]}
        depths = _compute_depths(by_name)
        assert depths["p"] == 0
        assert depths["c"] == 1

    def test_three_level_nesting(self):
        root = _jm("root", job_type="pipeline")
        mid = _jm("mid", job_type="pipeline", parent="root")
        leaf = _jm("leaf", parent="mid")
        by_name = {j.name: j for j in [root, mid, leaf]}
        depths = _compute_depths(by_name)
        assert depths["root"] == 0
        assert depths["mid"] == 1
        assert depths["leaf"] == 2

    def test_orphan_depth_is_one(self):
        """Orphan referencing a missing parent: depth('nonexistent') = 0, so orphan = 1 + 0 = 1."""
        orphan = _jm("orphan", parent="nonexistent")
        by_name = {"orphan": orphan}
        depths = _compute_depths(by_name)
        assert depths["orphan"] == 1

    def test_cycle_guard(self):
        """Circular parent references should not infinite-loop; depth defaults to 0."""
        a = _jm("a", parent="b")
        b = _jm("b", parent="a")
        by_name = {"a": a, "b": b}
        depths = _compute_depths(by_name)
        # Both should have finite depths (cycle guard returns 0)
        assert isinstance(depths["a"], int)
        assert isinstance(depths["b"], int)


# ── _create_disabled_manifest ───────────────────────────────────────


class TestCreateDisabledManifest:
    def test_returns_valid_json_file(self, tmp_path):
        path = _create_disabled_manifest("job1")
        assert path.endswith(".json")
        with open(path) as f:
            data = json.load(f)
        assert data["disabled"] is True
        assert data["original_run_id"] == "job1"

    def test_schema_version_present(self, tmp_path):
        path = _create_disabled_manifest("job2")
        with open(path) as f:
            data = json.load(f)
        assert data["schema_version"] == 1

    def test_relative_paths_empty(self, tmp_path):
        path = _create_disabled_manifest("job3")
        with open(path) as f:
            data = json.load(f)
        assert data["relative_paths"] == []

    def test_unique_paths_per_call(self):
        p1 = _create_disabled_manifest("same_job")
        p2 = _create_disabled_manifest("same_job")
        assert p1 != p2


# ── _load_manifest ──────────────────────────────────────────────────


class TestLoadManifest:
    def test_loads_valid_json(self, tmp_path):
        """Round-trip: write list of dicts, load back as (raw, JobMetadata list)."""
        data = [
            {"name": "j1", "job_type": "command"},
            {"name": "j2", "job_type": "pipeline"},
        ]
        p = tmp_path / "jobs.json"
        p.write_text(json.dumps(data))
        jobs_raw, jobs_meta = _load_manifest(str(p))
        assert len(jobs_raw) == 2
        assert len(jobs_meta) == 2
        assert jobs_meta[0].name == "j1"
        assert jobs_meta[1].job_type == "pipeline"

    def test_returns_raw_dicts_unchanged(self, tmp_path):
        """The raw list should be the original parsed dicts, not a copy."""
        data = [{"name": "j1", "description": "keep me"}]
        p = tmp_path / "jobs.json"
        p.write_text(json.dumps(data))
        jobs_raw, _ = _load_manifest(str(p))
        assert jobs_raw[0]["description"] == "keep me"

    def test_missing_file_raises(self, tmp_path):
        """Non-existent path should raise FileNotFoundError."""
        import pytest

        with pytest.raises(FileNotFoundError):
            _load_manifest(str(tmp_path / "missing.json"))

    def test_invalid_json_raises(self, tmp_path):
        """Malformed JSON should raise."""
        import pytest

        p = tmp_path / "bad.json"
        p.write_text("{not valid json")
        with pytest.raises(json.JSONDecodeError):
            _load_manifest(str(p))

    def test_empty_list(self, tmp_path):
        """Empty list should return two empty collections."""
        p = tmp_path / "empty.json"
        p.write_text("[]")
        jobs_raw, jobs_meta = _load_manifest(str(p))
        assert jobs_raw == []
        assert jobs_meta == []


# ── _infer_parent_links ────────────────────────────────────────────


class TestInferParentLinks:
    def test_no_inference_when_parent_already_set(self):
        """Jobs with existing parent_job_name should not be changed."""
        parent = _jm("pipe1", job_type="pipeline")
        child = _jm("c1", parent="pipe1")
        child.job_properties = {"azureml.pipelinerunid": "pipe1"}
        jobs = [parent, child]
        by_name, _, pipeline_names = _index_jobs(jobs)
        count = _infer_parent_links(jobs, by_name, pipeline_names)
        assert count == 0
        assert child.parent_job_name == "pipe1"

    def test_infers_from_pipelinerunid(self):
        """Should set parent_job_name from azureml.pipelinerunid."""
        parent = _jm("pipe1", job_type="pipeline")
        orphan = _jm("step1")
        orphan.job_properties = {"azureml.pipelinerunid": "pipe1"}
        jobs = [parent, orphan]
        by_name, _, pipeline_names = _index_jobs(jobs)
        count = _infer_parent_links(jobs, by_name, pipeline_names)
        assert count == 1
        assert orphan.parent_job_name == "pipe1"

    def test_infers_from_azureml_pipeline_fallback(self):
        """Should fall back to azureml.pipeline if pipelinerunid is absent."""
        parent = _jm("pipe2", job_type="pipeline")
        orphan = _jm("step2")
        orphan.job_properties = {"azureml.pipeline": "pipe2"}
        jobs = [parent, orphan]
        by_name, _, pipeline_names = _index_jobs(jobs)
        count = _infer_parent_links(jobs, by_name, pipeline_names)
        assert count == 1
        assert orphan.parent_job_name == "pipe2"

    def test_no_inference_when_candidate_not_pipeline(self):
        """Candidate parent must be in pipeline_names to be inferred."""
        cmd = _jm("cmd1")
        orphan = _jm("step3")
        orphan.job_properties = {"azureml.pipelinerunid": "cmd1"}
        jobs = [cmd, orphan]
        by_name, _, pipeline_names = _index_jobs(jobs)
        count = _infer_parent_links(jobs, by_name, pipeline_names)
        assert count == 0
        assert orphan.parent_job_name is None

    def test_no_self_reference(self):
        """A job should not infer itself as parent."""
        pipe = _jm("pipe_self", job_type="pipeline")
        pipe.job_properties = {"azureml.pipelinerunid": "pipe_self"}
        jobs = [pipe]
        by_name, _, pipeline_names = _index_jobs(jobs)
        count = _infer_parent_links(jobs, by_name, pipeline_names)
        assert count == 0

    def test_candidate_must_exist_in_manifest(self):
        """Candidate parent must exist in jobs_by_name."""
        orphan = _jm("lonely")
        orphan.job_properties = {"azureml.pipelinerunid": "nonexistent_pipe"}
        jobs = [orphan]
        by_name, _, pipeline_names = _index_jobs(jobs)
        count = _infer_parent_links(jobs, by_name, pipeline_names)
        assert count == 0


# ── _classify_replay_units ──────────────────────────────────────────


class TestClassifyReplayUnits:
    def test_single_standalone(self):
        """A lone command job produces one standalone replay unit."""
        jobs = [_jm("cmd1")]
        by_name, _, pipeline_names = _index_jobs(jobs)
        replay_units, depths, children = _classify_replay_units(
            jobs, pipeline_names, by_name
        )
        assert replay_units == [("standalone", "cmd1")]
        assert depths == {}
        assert children == {}

    def test_pipeline_with_children(self):
        """Pipeline with children: pipeline comes first as replay unit, child is not standalone."""
        parent = _jm("pipe1", job_type="pipeline")
        child = _jm("c1", parent="pipe1")
        jobs = [parent, child]
        by_name, _, pipeline_names = _index_jobs(jobs)
        replay_units, depths, children = _classify_replay_units(
            jobs, pipeline_names, by_name
        )
        assert ("pipeline", "pipe1") in replay_units
        # child is NOT a standalone root (it has a parent)
        assert ("standalone", "c1") not in replay_units
        assert children["pipe1"] == ["c1"]
        assert depths["pipe1"] == 0

    def test_nested_pipelines_ordered_by_depth(self):
        """Nested pipelines should be ordered by increasing depth."""
        root = _jm("root_pipe", job_type="pipeline")
        nested = _jm("nested_pipe", job_type="pipeline", parent="root_pipe")
        leaf = _jm("leaf", parent="nested_pipe")
        jobs = [root, nested, leaf]
        by_name, _, pipeline_names = _index_jobs(jobs)
        replay_units, depths, children = _classify_replay_units(
            jobs, pipeline_names, by_name
        )
        pipe_units = [u for u in replay_units if u[0] == "pipeline"]
        assert pipe_units[0] == ("pipeline", "root_pipe")
        assert pipe_units[1] == ("pipeline", "nested_pipe")
        assert depths["root_pipe"] == 0
        assert depths["nested_pipe"] == 1

    def test_standalone_roots_sorted_alphabetically(self):
        """Standalone root commands should be sorted by name."""
        jobs = [_jm("z_job"), _jm("a_job"), _jm("m_job")]
        by_name, _, pipeline_names = _index_jobs(jobs)
        replay_units, _, _ = _classify_replay_units(jobs, pipeline_names, by_name)
        names = [u[1] for u in replay_units]
        assert names == ["a_job", "m_job", "z_job"]

    def test_pipelines_come_before_standalones(self):
        """In replay_units, all pipelines appear before standalone roots."""
        pipe = _jm("pipe1", job_type="pipeline")
        cmd = _jm("standalone1")
        jobs = [cmd, pipe]
        by_name, _, pipeline_names = _index_jobs(jobs)
        replay_units, _, _ = _classify_replay_units(jobs, pipeline_names, by_name)
        assert replay_units[0] == ("pipeline", "pipe1")
        assert replay_units[1] == ("standalone", "standalone1")

    def test_mixed_hierarchy(self, sample_jobs_list):
        """Uses conftest fixture: standalone, pipeline parent, child."""
        jobs = [JobMetadata(**d) for d in sample_jobs_list]
        by_name, _, pipeline_names = _index_jobs(jobs)
        replay_units, depths, children = _classify_replay_units(
            jobs, pipeline_names, by_name
        )
        # pipeline_parent_001 is a replay unit
        assert ("pipeline", "pipeline_parent_001") in replay_units
        # happy_mango_abc123 is standalone root (no parent)
        assert ("standalone", "happy_mango_abc123") in replay_units
        # child_step_001 is NOT a standalone root
        assert ("standalone", "child_step_001") not in replay_units
        assert "child_step_001" in children["pipeline_parent_001"]


# ── _resolve_source_container ───────────────────────────────────────


class TestResolveSourceContainer:
    def _mock_blob_service(self, containers_with_blobs: dict[str, bool]):
        """Create a mock BlobServiceClient that returns blobs (or not) per container.

        containers_with_blobs: {"azureml-blobstore-xxx": True, "azureml": False}
        means the first container has blobs, the second doesn't.
        """
        svc = MagicMock()

        def _get_container_client(name):
            cc = MagicMock()
            if name not in containers_with_blobs:
                cc.list_blobs.side_effect = ResourceNotFoundError("ContainerNotFound")
            elif containers_with_blobs[name]:
                cc.list_blobs.return_value = iter([MagicMock()])  # one blob
            else:
                cc.list_blobs.return_value = iter([])  # empty
            return cc

        svc.get_container_client.side_effect = _get_container_client
        return svc

    def test_found_in_datastore_container(self):
        """Blobs exist in the datastore container — use it directly."""
        svc = self._mock_blob_service({"azureml-blobstore-xxx": True})
        result = _resolve_source_container(
            svc, "azureml-blobstore-xxx", ["job1", "job2"]
        )
        assert result == "azureml-blobstore-xxx"

    def test_fallback_to_azureml(self):
        """No blobs in datastore container, found in 'azureml' fallback."""
        svc = self._mock_blob_service(
            {"azureml-blobstore-xxx": False, "azureml": True}
        )
        result = _resolve_source_container(
            svc, "azureml-blobstore-xxx", ["job1", "job2"]
        )
        assert result == "azureml"

    def test_fallback_container_not_found(self):
        """Datastore container empty, fallback container doesn't exist — raises."""
        svc = self._mock_blob_service({"azureml-blobstore-xxx": False})
        import pytest

        with pytest.raises(RuntimeError, match="Could not find"):
            _resolve_source_container(
                svc, "azureml-blobstore-xxx", ["job1"]
            )

    def test_both_empty_raises(self):
        """Both containers exist but have no blobs — raises."""
        svc = self._mock_blob_service(
            {"azureml-blobstore-xxx": False, "azureml": False}
        )
        import pytest

        with pytest.raises(RuntimeError, match="Could not find"):
            _resolve_source_container(
                svc, "azureml-blobstore-xxx", ["job1"]
            )

    def test_probes_multiple_jobs(self):
        """First job has no blobs, second job does — still finds container."""
        svc = MagicMock()
        call_count = {"n": 0}

        def _get_cc(name):
            cc = MagicMock()

            def _list_blobs(**kwargs):
                prefix = kwargs.get("name_starts_with", "")
                call_count["n"] += 1
                if "job1" in prefix:
                    return iter([])
                return iter([MagicMock()])

            cc.list_blobs.side_effect = _list_blobs
            return cc

        svc.get_container_client.side_effect = _get_cc
        result = _resolve_source_container(
            svc, "azureml-blobstore-xxx", ["job1", "job2"]
        )
        assert result == "azureml-blobstore-xxx"

    def test_skips_fallback_when_same_as_datastore(self):
        """If datastore container IS 'azureml', don't probe it twice."""
        svc = self._mock_blob_service({"azureml": False})
        import pytest

        with pytest.raises(RuntimeError, match="Could not find"):
            _resolve_source_container(svc, "azureml", ["job1"])

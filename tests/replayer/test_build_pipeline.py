"""Tests for pure functions in replayer.build_pipeline."""

import json

from extractor.extract_jobs import JobMetadata
from replayer.build_pipeline import (
    _compute_depths,
    _create_disabled_manifest,
    _index_jobs,
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

from __future__ import annotations
import dataclasses as dc
from pathlib import Path
import yaml
from typing import List, Dict, Any, Optional


@dc.dataclass
class WorkspaceRef:
    subscription_id: str
    resource_group: str
    workspace_name: str
    tenant_id: Optional[str] = None


@dc.dataclass
class MLTableSelection:
    name: str
    versions: List[str] | None = None  # None => all versions


@dc.dataclass
class SelectionSpec:
    batch_id: str
    mltables: List[MLTableSelection]
    source: Optional[WorkspaceRef] = None
    target: Optional[WorkspaceRef] = None
    jobs: List[Dict[str, Any]] | None = None  # reserved for future

    @staticmethod
    def load(path: str | Path) -> "SelectionSpec":
        data = yaml.safe_load(Path(path).read_text())
        batch_id = data.get("batch_id") or data.get("id") or "batch-unknown"
        raw_tables = data.get("mltables") or []
        tables: List[MLTableSelection] = []
        for item in raw_tables:
            if isinstance(item, str):
                tables.append(MLTableSelection(name=item, versions=None))
            elif isinstance(item, dict):
                tables.append(
                    MLTableSelection(name=item["name"], versions=item.get("versions"))
                )
            else:
                raise ValueError(f"Unsupported mltables entry: {item}")

        def parse_ws(section: str) -> Optional[WorkspaceRef]:
            ws = data.get(section)
            if not ws:
                return None
            required = ["subscription_id", "resource_group", "workspace_name"]
            missing = [k for k in required if k not in ws]
            if missing:
                raise ValueError(f"Missing keys in {section}: {missing}")
            return WorkspaceRef(
                subscription_id=ws["subscription_id"],
                resource_group=ws["resource_group"],
                workspace_name=ws["workspace_name"],
                tenant_id=ws.get("tenant_id"),
            )

        return SelectionSpec(
            batch_id=batch_id,
            mltables=tables,
            source=parse_ws("source"),
            target=parse_ws("target"),
            jobs=data.get("jobs"),
        )

    def requested_versions(self, name: str) -> Optional[List[str]]:
        for t in self.mltables:
            if t.name == name:
                return t.versions
        return None

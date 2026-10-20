"""OrientDB REST adapter for durable provenance relationships.

The adapter stores provenance records in a dedicated OrientDB class. It uses
OrientDB's parameterized SQL command endpoint; values are never interpolated into
SQL text.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .types import ProvenanceRecord

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class OrientDBProvenanceStore:
    """Persist provenance as native OrientDB vertices and edges over REST."""

    _VERTEX_CLASS = "LoomEntity"
    _EDGE_CLASS = "LoomProvenance"

    def __init__(
        self,
        base_url: str,
        database: str,
        username: str,
        password: str,
        *,
        timeout: float = 10.0,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use http:// or https://")
        if not _IDENTIFIER.fullmatch(database):
            raise ValueError("database must be a simple OrientDB database name")
        if not username or not password:
            raise ValueError("OrientDB username and password are required")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.database = database
        self.username = username
        self.password = password
        self.timeout = timeout

    async def initialize(self) -> None:
        """Create provenance vertices, edges, properties, and lookup indexes."""
        await self._command(
            f"CREATE CLASS {self._VERTEX_CLASS} IF NOT EXISTS EXTENDS V"
        )
        await self._command(
            f"CREATE PROPERTY {self._VERTEX_CLASS}.entityId STRING IF NOT EXISTS"
        )
        await self._command(
            f"CREATE INDEX {self._VERTEX_CLASS}_entityId IF NOT EXISTS "
            f"ON {self._VERTEX_CLASS} (entityId) UNIQUE"
        )
        await self._command(f"CREATE CLASS {self._EDGE_CLASS} IF NOT EXISTS EXTENDS E")
        properties = {
            "recordId": "STRING",
            "relation": "STRING",
            "metadata": "EMBEDDEDMAP",
            "createdAt": "STRING",
        }
        for name, kind in properties.items():
            await self._command(
                f"CREATE PROPERTY {self._EDGE_CLASS}.{name} {kind} IF NOT EXISTS"
            )
        await self._command(
            f"CREATE INDEX {self._EDGE_CLASS}_recordId IF NOT EXISTS "
            f"ON {self._EDGE_CLASS} (recordId) UNIQUE"
        )

    async def put_provenance(self, record: ProvenanceRecord) -> None:
        created_at = record.created_at or datetime.now(timezone.utc).isoformat()
        for entity_id in (record.subject_id, record.object_id):
            await self._command(
                f"UPDATE {self._VERTEX_CLASS} SET entityId = :entityId "
                "UPSERT WHERE entityId = :entityId",
                {"entityId": entity_id},
            )

        existing = await self.get_provenance(record.id)
        parameters = {
            "recordId": record.id,
            "subjectId": record.subject_id,
            "relation": record.relation,
            "objectId": record.object_id,
            "metadata": record.metadata,
            "createdAt": created_at,
        }
        if existing is not None and (
            existing.subject_id != record.subject_id
            or existing.object_id != record.object_id
        ):
            await self.delete_provenance(record.id)
            existing = None

        if existing is not None:
            await self._command(
                f"""
                UPDATE {self._EDGE_CLASS}
                SET relation = :relation,
                    metadata = :metadata,
                    createdAt = :createdAt
                WHERE recordId = :recordId
                """,
                parameters,
            )
            return

        await self._command(
            f"""
            CREATE EDGE {self._EDGE_CLASS}
            FROM (SELECT FROM {self._VERTEX_CLASS} WHERE entityId = :subjectId)
            TO (SELECT FROM {self._VERTEX_CLASS} WHERE entityId = :objectId)
            SET recordId = :recordId,
                relation = :relation,
                metadata = :metadata,
                createdAt = :createdAt
            """,
            parameters,
        )

    async def get_provenance(self, record_id: str) -> ProvenanceRecord | None:
        response = await self._command(
            f"""
            SELECT recordId, out.entityId AS subjectId, relation,
                   in.entityId AS objectId, metadata, createdAt
            FROM {self._EDGE_CLASS}
            WHERE recordId = :recordId
            LIMIT 1
            """,
            {"recordId": record_id},
        )
        rows = _result_rows(response)
        return _record_from_row(rows[0]) if rows else None

    async def get_provenance_for_subject(
        self, subject_id: str, limit: int = 1000
    ) -> list[ProvenanceRecord]:
        if not isinstance(limit, int) or not 1 <= limit <= 10000:
            raise ValueError("limit must be between 1 and 10000")
        response = await self._command(
            f"""
            SELECT recordId, out.entityId AS subjectId, relation,
                   in.entityId AS objectId, metadata, createdAt
            FROM {self._EDGE_CLASS}
            WHERE out.entityId = :subjectId
            ORDER BY createdAt, recordId
            LIMIT {limit}
            """,
            {"subjectId": subject_id},
        )
        return [_record_from_row(row) for row in _result_rows(response)]

    async def delete_provenance(self, record_id: str) -> bool:
        response = await self._command(
            f"DELETE EDGE {self._EDGE_CLASS} WHERE recordId = :recordId",
            {"recordId": record_id},
        )
        if isinstance(response, int):
            return response > 0
        if isinstance(response, dict):
            result = response.get("result")
            if isinstance(result, int):
                return result > 0
            if isinstance(result, list):
                return bool(result)
        return False

    async def _command(
        self, command: str, parameters: dict[str, Any] | None = None
    ) -> Any:
        return await asyncio.to_thread(self._command_sync, command, parameters or {})

    def _command_sync(self, command: str, parameters: dict[str, Any]) -> Any:
        url = f"{self.base_url}/command/{quote(self.database, safe='')}/sql"
        body = json.dumps(
            {"command": command, "parameters": parameters},
            separators=(",", ":"),
        ).encode("utf-8")
        token = base64.b64encode(f"{self.username}:{self.password}".encode()).decode(
            "ascii"
        )
        request = Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Basic {token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(
                f"OrientDB command failed with HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"OrientDB is unavailable: {exc.reason}") from exc
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("OrientDB returned invalid JSON") from exc


def _result_rows(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, dict):
        rows = response.get("result", [])
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    if isinstance(response, list):
        return [row for row in response if isinstance(row, dict)]
    return []


def _record_from_row(row: dict[str, Any]) -> ProvenanceRecord:
    metadata = row.get("metadata")
    return ProvenanceRecord(
        id=str(row["recordId"]),
        subject_id=str(row["subjectId"]),
        relation=str(row["relation"]),
        object_id=str(row["objectId"]),
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
        created_at=str(row.get("createdAt", "")),
    )

"""LanceDB operations for learning storage."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import lancedb
import pyarrow as pa

logger = logging.getLogger(__name__)

_TABLE_NAME = "learnings"


def _make_schema(vector_dimensions: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), vector_dimensions)),
            pa.field("text", pa.string()),
            pa.field("category", pa.string()),
            pa.field("tags", pa.list_(pa.string())),
            pa.field("project", pa.string()),
            pa.field("session_id", pa.string()),
            pa.field("git_sha", pa.string()),
            pa.field("timestamp", pa.timestamp("us", tz="UTC")),
            pa.field("confidence", pa.float32()),
            pa.field("active_count", pa.int32()),
            pa.field("stale", pa.bool_()),
            pa.field("shared", pa.bool_()),
        ]
    )


def _get_vector_dim_from_schema(schema: pa.Schema) -> int:
    """Extract vector dimensionality from an existing table's schema."""
    vector_field = schema.field("vector")
    return vector_field.type.list_size


class LearningStore:
    """Manages the LanceDB learnings table."""

    def __init__(
        self,
        db_path: str,
        vector_dimensions: int | None = None,
        dedup_threshold: float = 0.95,
    ):
        """Open or create the LanceDB database and learnings table.

        Args:
            db_path: Path to LanceDB directory (e.g. ~/.crowd-control/db).
            vector_dimensions: Length of embedding vectors. Required when creating
                a new table. If table already exists, read from schema.
            dedup_threshold: Cosine similarity threshold for near-duplicate rejection.
        """
        self._dedup_threshold = dedup_threshold
        expanded = Path(db_path).expanduser()
        expanded.mkdir(parents=True, exist_ok=True)

        self._db = lancedb.connect(str(expanded))

        if _TABLE_NAME in self._db.list_tables().tables:
            self._table = self._db.open_table(_TABLE_NAME)
            existing_dims = _get_vector_dim_from_schema(self._table.schema)
            if vector_dimensions is not None and vector_dimensions != existing_dims:
                raise ValueError(
                    f"Embedding dimension mismatch. Table has {existing_dims}-dim vectors "
                    f"but embedder produces {vector_dimensions}-dim.\n"
                    f"This happens when switching embedding models. To fix:\n"
                    f"  1. Back up: cp -r {expanded} {expanded}.bak\n"
                    f"  2. Delete: rm -rf {expanded}\n"
                    f"  3. Re-ingest sessions with the new model."
                )
            self._vector_dimensions = existing_dims
        else:
            if vector_dimensions is None:
                raise ValueError(
                    "vector_dimensions is required when creating a new table. "
                    "Run an ingestion first to initialize the database."
                )
            self._vector_dimensions = vector_dimensions
            schema = _make_schema(vector_dimensions)
            self._table = self._db.create_table(_TABLE_NAME, schema=schema)

    def add(self, learnings: list[dict]) -> int:
        """Insert learnings into the table with deduplication.

        Returns the number of learnings actually inserted (after dedup filtering).
        Checks each learning against both existing DB rows and other learnings
        already accepted in the same batch.
        """
        if not learnings:
            return 0

        is_empty = self._table.count_rows() == 0

        to_insert = []
        seen_texts: set[str] = set()
        for learning in learnings:
            text = learning["text"]
            if text in seen_texts:
                continue
            if not is_empty:
                if self._has_exact_text(text):
                    continue
                if self._has_near_duplicate(learning["vector"]):
                    continue
            seen_texts.add(text)
            to_insert.append(learning)

        if to_insert:
            self._table.add(to_insert)

        return len(to_insert)

    def get(self, learning_id: str) -> dict | None:
        """Get a single learning by ID. Returns None if not found."""
        escaped = learning_id.replace("'", "''")
        results = self._table.search().where(f"id = '{escaped}'").limit(1).to_list()
        if not results:
            return None
        row = results[0]
        row.pop("_rowid", None)
        return row

    def _query_learnings(
        self,
        project: str | None = None,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Shared query logic for list and export. Returns raw rows."""
        conditions: list[str] = []
        if project is not None:
            escaped = project.replace("'", "''")
            conditions.append(f"project = '{escaped}'")
        if category is not None:
            escaped = category.replace("'", "''")
            conditions.append(f"category = '{escaped}'")

        query = self._table.search()
        if conditions:
            query = query.where(" AND ".join(conditions))

        effective_limit = limit if limit is not None else self._table.count_rows()
        if effective_limit == 0:
            return []

        return query.limit(effective_limit).to_list()

    def export_learnings(
        self,
        project: str | None = None,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Export learnings as a list of dicts, stripping vectors and internal keys.

        Unlike list_learnings, defaults to no limit (all matching rows).
        Timestamps are converted to ISO 8601 strings.
        """
        rows = self._query_learnings(project=project, category=category, limit=limit)

        _EXCLUDE_KEYS = {"vector", "_rowid", "_distance"}
        results = []
        for row in rows:
            record = {k: v for k, v in row.items() if k not in _EXCLUDE_KEYS}
            ts = record.get("timestamp")
            if ts is not None:
                if hasattr(ts, "isoformat"):
                    record["timestamp"] = ts.isoformat()
                elif hasattr(ts, "to_pydatetime"):
                    record["timestamp"] = ts.to_pydatetime().isoformat()
            results.append(record)

        results.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        return results

    def list_learnings(
        self,
        project: str | None = None,
        category: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List learnings with optional filtering, ordered by timestamp descending."""
        results = self._query_learnings(project=project, category=category, limit=limit)

        for row in results:
            row.pop("_rowid", None)
            row.pop("_distance", None)
            row.pop("vector", None)

        results.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        return results

    def delete(self, learning_id: str) -> bool:
        """Delete a learning by ID. Returns True if it existed."""
        existing = self.get(learning_id)
        if existing is None:
            return False
        escaped = learning_id.replace("'", "''")
        self._table.delete(f"id = '{escaped}'")
        return True

    def count(self) -> int:
        """Return the total number of learnings in the table."""
        return self._table.count_rows()

    def has_session(self, session_id: str) -> bool:
        """Check if any learnings exist for a given session ID."""
        escaped = session_id.replace("'", "''")
        results = self._table.search().where(f"session_id = '{escaped}'").limit(1).to_list()
        return len(results) > 0

    def _has_exact_text(self, text: str) -> bool:
        escaped = text.replace("'", "''")
        results = self._table.search().where(f"text = '{escaped}'").limit(1).to_list()
        return len(results) > 0

    def _has_near_duplicate(self, vector: list[float]) -> bool:
        results = self._table.search(vector).metric("cosine").limit(1).to_list()
        if not results:
            return False
        return results[0]["_distance"] < (1.0 - self._dedup_threshold)

    def increment_active_count(self, learning_ids: list[str]) -> None:
        """Increment active_count by 1 for each learning ID.

        Fetches all matching rows in a single query, then applies batch
        updates. No-ops silently for IDs that don't exist. Race conditions
        on concurrent increments are acceptable — the counter is approximate.
        """
        if not learning_ids:
            return

        escaped_ids = [lid.replace("'", "''") for lid in learning_ids]
        id_list = ", ".join(f"'{eid}'" for eid in escaped_ids)
        where_clause = f"id IN ({id_list})"

        rows = self._table.search().where(where_clause).limit(len(learning_ids)).to_list()

        for row in rows:
            new_count = row.get("active_count", 0) + 1
            escaped = row["id"].replace("'", "''")
            self._table.update(where=f"id = '{escaped}'", values={"active_count": new_count})

    def vector_search(
        self,
        query_vector: list[float],
        limit: int,
        min_similarity: float = 0.3,
        category: str | None = None,
        tags: list[str] | None = None,
        exclude_stale: bool = True,
        scope: str = "project",
        current_project: str | None = None,
    ) -> list[dict]:
        """Execute a vector search with metadata filtering.

        Returns up to `limit` results, each with a `_similarity` key.
        Results below `min_similarity` are filtered out.
        """
        if self._table.count_rows() == 0:
            return []

        conditions: list[str] = []

        if scope == "project" and current_project is not None:
            escaped = current_project.replace("'", "''")
            conditions.append(f"project = '{escaped}'")
        elif scope == "mixed" and current_project is not None:
            escaped = current_project.replace("'", "''")
            conditions.append(f"(project = '{escaped}' OR shared = true)")

        if category is not None:
            escaped_cat = category.replace("'", "''")
            conditions.append(f"category = '{escaped_cat}'")

        if tags:
            tag_clauses = []
            for tag in tags:
                escaped_tag = tag.lower().replace("'", "''")
                tag_clauses.append(f"array_contains(tags, '{escaped_tag}')")
            conditions.append(f"({' OR '.join(tag_clauses)})")

        if exclude_stale:
            conditions.append("stale = false")

        query = self._table.search(query_vector).metric("cosine").limit(limit)
        if conditions:
            query = query.where(" AND ".join(conditions))

        rows = query.to_list()

        # Internal keys injected by LanceDB that are not part of the schema
        _INTERNAL_KEYS = {"_rowid", "_distance"}

        results = []
        for row in rows:
            similarity = 1.0 - row["_distance"]
            if similarity < min_similarity:
                logger.debug(
                    "vector_search: rejected (similarity=%.3f < %.2f): %.80s",
                    similarity,
                    min_similarity,
                    row.get("text", ""),
                )
                continue

            # Build a clean result dict without mutating the source row
            result = {k: v for k, v in row.items() if k not in _INTERNAL_KEYS}

            # Normalize timestamp to Python datetime
            ts = result.get("timestamp")
            if ts is not None and not isinstance(ts, datetime):
                result["timestamp"] = ts.to_pydatetime()

            result["_similarity"] = similarity
            results.append(result)

        rejected_count = len(rows) - len(results)
        logger.debug(
            "vector_search: %d raw rows, %d passed threshold (min_similarity=%.2f), "
            "%d rejected. Lowest accepted: %.3f",
            len(rows),
            len(results),
            min_similarity,
            rejected_count,
            min(r["_similarity"] for r in results) if results else 0.0,
        )

        return results

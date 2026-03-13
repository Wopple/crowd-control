"""LanceDB operations for learning storage."""

from __future__ import annotations

import logging
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
        """
        if not learnings:
            return 0

        is_empty = self._table.count_rows() == 0

        to_insert = []
        for learning in learnings:
            if not is_empty:
                if self._has_exact_text(learning["text"]):
                    continue
                if self._has_near_duplicate(learning["vector"]):
                    continue
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

    def list_learnings(
        self,
        project: str | None = None,
        category: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List learnings with optional filtering, ordered by timestamp descending."""
        conditions = []
        if project is not None:
            escaped = project.replace("'", "''")
            conditions.append(f"project = '{escaped}'")
        if category is not None:
            escaped = category.replace("'", "''")
            conditions.append(f"category = '{escaped}'")

        query = self._table.search()
        if conditions:
            query = query.where(" AND ".join(conditions))
        results = query.limit(limit).to_list()

        for row in results:
            row.pop("_rowid", None)
            row.pop("_distance", None)

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

    def _has_exact_text(self, text: str) -> bool:
        escaped = text.replace("'", "''")
        results = self._table.search().where(f"text = '{escaped}'").limit(1).to_list()
        return len(results) > 0

    def _has_near_duplicate(self, vector: list[float]) -> bool:
        results = self._table.search(vector).metric("cosine").limit(1).to_list()
        if not results:
            return False
        return results[0]["_distance"] < (1.0 - self._dedup_threshold)

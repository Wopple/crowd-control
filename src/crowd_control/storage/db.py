"""LanceDB operations for learning storage."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import lancedb
import pyarrow as pa
import pyarrow.compute as pc

from crowd_control.storage.migration import run_migrations, stamp_initial_version

logger = logging.getLogger(__name__)


@dataclass
class PruneCandidate:
    """A learning identified as eligible for pruning."""

    id: str
    text: str
    category: str
    age_days: float
    active_count: int
    required_count: int


@dataclass
class DuplicateInfo:
    """Details about a learning rejected as a near-duplicate."""

    new_text: str
    matched_text: str
    similarity: float


@dataclass
class AddResult:
    """Result of an add() operation with deduplication details."""

    stored: int
    duplicates: list[DuplicateInfo] = field(default_factory=list)


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


def _required_active_count(age_days: float, interval_days: int) -> int:
    """Minimum retrievals required for a learning of the given age.

    Returns ceil(age_days / interval_days), minimum 1.
    """
    if interval_days <= 0:
        return 0
    return max(1, math.ceil(age_days / interval_days))


def _find_prunable(
    candidates: list[dict],
    retrieval_interval_days: int,
    now: datetime,
) -> list[PruneCandidate]:
    """Identify learnings that haven't been retrieved enough for their age.

    Pure function: takes raw rows from LanceDB, returns prunable candidates.
    """
    prunable: list[PruneCandidate] = []
    for row in candidates:
        ts = row.get("timestamp")
        if ts is None:
            continue
        if not isinstance(ts, datetime):
            ts = ts.to_pydatetime()
        age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
        required = _required_active_count(age_days, retrieval_interval_days)
        active = row.get("active_count", 0)
        if active < required:
            prunable.append(
                PruneCandidate(
                    id=row["id"],
                    text=row.get("text", ""),
                    category=row.get("category", ""),
                    age_days=age_days,
                    active_count=active,
                    required_count=required,
                )
            )
    return prunable


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _find_similar_in_batch(
    vector: list[float],
    accepted_vectors: list[list[float]],
    threshold: float,
) -> tuple[int, float] | None:
    """Find the first vector in accepted_vectors above the similarity threshold.

    Returns (index, similarity) if found, else None.
    """
    for i, accepted in enumerate(accepted_vectors):
        sim = _cosine_similarity(vector, accepted)
        if sim >= threshold:
            return i, sim
    return None


class LearningStore:
    """Manages the LanceDB learnings table."""

    def __init__(
        self,
        db_path: str,
        vector_dimensions: int | None = None,
        dedup_threshold: float = 0.90,
    ):
        """Open or create the LanceDB database and learnings table.

        Args:
            db_path: Path to LanceDB directory (e.g. ~/.crowd-control/db).
            vector_dimensions: Length of embedding vectors. Required when creating
                a new table. If table already exists, read from schema.
            dedup_threshold: Cosine similarity threshold for near-duplicate rejection.
                Default matches IngestionConfig.dedup_threshold.
        """
        self._dedup_threshold = dedup_threshold
        expanded = Path(db_path).expanduser()
        expanded.mkdir(parents=True, exist_ok=True)

        self._db = lancedb.connect(str(expanded))

        if _TABLE_NAME in self._db.list_tables().tables:
            run_migrations(self._db, _TABLE_NAME)
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
            stamp_initial_version(self._db)

    def add(self, learnings: list[dict]) -> AddResult:
        """Insert learnings into the table with deduplication.

        Returns an AddResult with the count of stored learnings and details
        about any rejected duplicates. Checks each learning against existing
        DB rows, and also performs intra-batch dedup (both exact text and
        vector similarity) so duplicates within a single batch are caught.
        """
        if not learnings:
            return AddResult(stored=0)

        is_empty = self._table.count_rows() == 0

        to_insert: list[dict] = []
        duplicates: list[DuplicateInfo] = []
        seen_texts: set[str] = set()
        accepted_vectors: list[list[float]] = []

        for learning in learnings:
            text = learning["text"]
            vector = learning["vector"]

            # Exact text dedup: within batch
            if text in seen_texts:
                continue

            if not is_empty:
                # Exact text dedup: against existing DB rows
                if self._has_exact_text(text):
                    continue
                # Near-duplicate dedup: against existing DB rows
                match = self._find_near_duplicate(vector)
                if match is not None:
                    matched_text, similarity = match
                    duplicates.append(DuplicateInfo(text, matched_text, similarity))
                    logger.debug(
                        "dedup: rejected (sim=%.3f): %.80s",
                        similarity,
                        text,
                    )
                    continue

            # Near-duplicate dedup: within batch (covers empty table case too)
            batch_match = _find_similar_in_batch(vector, accepted_vectors, self._dedup_threshold)
            if batch_match is not None:
                matched_idx, similarity = batch_match
                duplicates.append(DuplicateInfo(text, to_insert[matched_idx]["text"], similarity))
                logger.debug(
                    "dedup: rejected within batch (sim=%.3f): %.80s",
                    similarity,
                    text,
                )
                continue

            seen_texts.add(text)
            accepted_vectors.append(vector)
            to_insert.append(learning)

        if to_insert:
            self._table.add(to_insert)

        return AddResult(stored=len(to_insert), duplicates=duplicates)

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

    def find_by_prefix(self, prefix: str) -> list[dict]:
        """Find learnings whose ID starts with the given prefix.

        Returns matching rows without vector data. Used for resolving
        short ID prefixes to full IDs. Filters in Python rather than SQL
        because LanceDB's search-mode WHERE clause is a post-filter on
        KNN results, not a full-table scan.
        """
        total = self._table.count_rows()
        if total == 0:
            return []

        rows = self._table.search().limit(total).to_list()
        results = []
        for row in rows:
            if row["id"].startswith(prefix):
                row.pop("_rowid", None)
                row.pop("_distance", None)
                row.pop("vector", None)
                results.append(row)
        return results

    def delete(self, learning_id: str) -> bool:
        """Delete a learning by ID. Returns True if it existed."""
        existing = self.get(learning_id)
        if existing is None:
            return False
        escaped = learning_id.replace("'", "''")
        self._table.delete(f"id = '{escaped}'")
        return True

    def update_project(self, old_project: str, new_project: str) -> int:
        """Re-key all learnings from *old_project* to *new_project*.

        Returns the number of rows updated.
        """
        escaped_old = old_project.replace("'", "''")
        count = self._table.count_rows(filter=f"project = '{escaped_old}'")
        if count == 0:
            return 0

        self._table.update(
            where=f"project = '{escaped_old}'",
            values={"project": new_project},
        )
        return count

    def count(self, project: str | None = None) -> int:
        """Return the number of learnings, optionally filtered by project."""
        if project is None:
            return self._table.count_rows()
        escaped = project.replace("'", "''")
        return self._table.count_rows(filter=f"project = '{escaped}'")

    def distinct_tags(self, project: str | None = None) -> list[str]:
        """Return unique tags, optionally filtered by project, sorted alphabetically."""
        if self._table.count_rows() == 0:
            return []

        query = self._table.search().select(["tags"])
        if project is not None:
            escaped = project.replace("'", "''")
            query = query.where(f"project = '{escaped}'")

        arrow_table = query.limit(self._table.count_rows()).to_arrow()
        if arrow_table.num_rows == 0:
            return []

        flat = pc.list_flatten(arrow_table.column("tags"))
        unique = flat.unique().to_pylist()
        logger.debug(
            "distinct_tags: %d unique tags from %d learnings (project=%s)",
            len(unique),
            arrow_table.num_rows,
            project,
        )
        return sorted(unique)

    def has_session(self, session_id: str) -> bool:
        """Check if any learnings exist for a given session ID."""
        escaped = session_id.replace("'", "''")
        results = self._table.search().where(f"session_id = '{escaped}'").limit(1).to_list()
        return len(results) > 0

    def _has_exact_text(self, text: str) -> bool:
        escaped = text.replace("'", "''")
        results = self._table.search().where(f"text = '{escaped}'").limit(1).to_list()
        return len(results) > 0

    def _find_near_duplicate(self, vector: list[float]) -> tuple[str, float] | None:
        """Find the nearest learning above the dedup threshold.

        Returns (matched_text, similarity) if a near-duplicate exists, else None.
        """
        results = self._table.search(vector).metric("cosine").limit(1).to_list()
        if not results:
            return None
        similarity = 1.0 - results[0]["_distance"]
        if similarity >= self._dedup_threshold:
            return results[0]["text"], similarity
        return None

    def _fetch_prune_candidates(
        self,
        max_age_days: int,
        retrieval_interval_days: int,
        now: datetime,
    ) -> list[PruneCandidate]:
        """Find learnings eligible for pruning without deleting them."""
        cutoff = now - timedelta(days=max_age_days)
        cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%S+00:00")

        where = f"timestamp < timestamp '{cutoff_str}'"
        rows = self._table.search().where(where).limit(self._table.count_rows()).to_list()

        if not rows:
            return []

        return _find_prunable(rows, retrieval_interval_days, now)

    def prune(
        self,
        max_age_days: int,
        retrieval_interval_days: int,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> int | list[PruneCandidate]:
        """Delete old learnings with insufficient retrieval activity.

        A learning older than max_age_days survives only if it has been
        retrieved at least once per retrieval_interval_days over its lifetime.
        The required count scales with age: a 90-day-old learning with a
        30-day interval needs 3 retrievals, a 120-day-old needs 4, etc.

        This gives active learnings an extended life, but not an indefinite
        one — they must keep proving their value as they age.

        Args:
            max_age_days: Learnings older than this are candidates. 0 disables.
            retrieval_interval_days: Required retrieval frequency.
            now: Current time (for testing). Defaults to UTC now.
            dry_run: If True, return list of PruneCandidate without deleting.

        Returns:
            Count of deleted learnings, or list of PruneCandidate if dry_run.
        """
        if max_age_days <= 0:
            return [] if dry_run else 0

        if self._table.count_rows() == 0:
            return [] if dry_run else 0

        if now is None:
            now = datetime.now(UTC)

        prunable = self._fetch_prune_candidates(max_age_days, retrieval_interval_days, now)

        if dry_run:
            return prunable

        if not prunable:
            return 0

        ids_to_delete = [c.id for c in prunable]
        escaped_ids = [lid.replace("'", "''") for lid in ids_to_delete]
        id_list = ", ".join(f"'{eid}'" for eid in escaped_ids)
        self._table.delete(f"id IN ({id_list})")

        logger.info(
            "prune: deleted %d learnings older than %d days",
            len(ids_to_delete),
            max_age_days,
        )
        return len(ids_to_delete)

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

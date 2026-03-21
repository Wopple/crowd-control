"""Schema version tracking and migration runner for LanceDB."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import lancedb
import pyarrow as pa

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION: int = 1
_METADATA_TABLE: str = "_metadata"

_METADATA_SCHEMA = pa.schema(
    [
        pa.field("key", pa.string()),
        pa.field("value", pa.string()),
    ]
)


@dataclass(frozen=True)
class Migration:
    """A single schema migration step.

    Attributes:
        from_version: The version this migration upgrades from.
        to_version: The version this migration upgrades to.
        description: Human-readable summary for log messages.
        migrate: Function that performs the schema change. Receives the DB
            connection (for metadata operations) and the learnings table
            (for add_columns, alter_columns, etc.). Must be idempotent —
            guard with ``if column not in table.schema.names`` before
            calling ``add_columns``, since LanceDB raises RuntimeError
            if the column already exists.
    """

    from_version: int
    to_version: int
    description: str
    migrate: Callable[[lancedb.db.LanceDBConnection, lancedb.table.Table], None]


# Migration registry. Starts empty — the first entry is added when the
# schema changes from v1.
_MIGRATIONS: list[Migration] = []


def read_schema_version(db: lancedb.db.LanceDBConnection) -> int:
    """Read the schema version from the metadata table.

    Returns 1 (baseline) if the metadata table does not exist, has no
    ``schema_version`` key, or contains an invalid value.
    """
    if _METADATA_TABLE not in db.list_tables().tables:
        return 1

    table = db.open_table(_METADATA_TABLE)
    rows = table.search().where("key = 'schema_version'").limit(1).to_list()
    if not rows:
        return 1

    raw = rows[0].get("value")
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid schema_version value %r in metadata — treating as v1",
            raw,
        )
        return 1


def _write_schema_version(db: lancedb.db.LanceDBConnection, version: int) -> None:
    """Write schema version to the metadata table, creating it if absent."""
    if _METADATA_TABLE not in db.list_tables().tables:
        table = db.create_table(_METADATA_TABLE, schema=_METADATA_SCHEMA)
    else:
        table = db.open_table(_METADATA_TABLE)

    # Remove any existing schema_version row (no-op on empty table)
    table.delete("key = 'schema_version'")
    table.add([{"key": "schema_version", "value": str(version)}])


def stamp_initial_version(db: lancedb.db.LanceDBConnection) -> None:
    """Stamp a newly-created database at the current schema version.

    Called from the create path in ``LearningStore.__init__``.
    """
    _write_schema_version(db, CURRENT_SCHEMA_VERSION)


def run_migrations(
    db: lancedb.db.LanceDBConnection,
    table_name: str,
    *,
    migrations: list[Migration] | None = None,
    target_version: int | None = None,
) -> None:
    """Run pending schema migrations on an existing learnings table.

    Called from ``LearningStore.__init__`` after detecting an existing table.

    Concurrent init safety: migrations are idempotent (each function guards
    against already-applied changes) so two processes running the same
    migration simultaneously is safe. The second process either skips the
    already-applied change or overwrites the metadata with the same value.

    Args:
        db: The LanceDB connection.
        table_name: Name of the learnings table.
        migrations: Override for testing. Defaults to ``_MIGRATIONS``.
        target_version: Override for testing. Defaults to
            ``CURRENT_SCHEMA_VERSION``.
    """
    effective_migrations = migrations if migrations is not None else _MIGRATIONS
    target = target_version if target_version is not None else CURRENT_SCHEMA_VERSION

    stored = read_schema_version(db)

    if stored > target:
        raise RuntimeError(
            f"Database schema v{stored} is newer than this version of "
            f"Crowd Control (expects v{target}). Upgrade Crowd Control or "
            f"restore a backup created with an older version."
        )

    if stored < target:
        table = db.open_table(table_name)

        pending = sorted(
            [m for m in effective_migrations if m.from_version >= stored],
            key=lambda m: m.from_version,
        )

        for migration in pending:
            if migration.from_version != stored:
                continue
            logger.info(
                "Migrating schema v%d → v%d: %s",
                migration.from_version,
                migration.to_version,
                migration.description,
            )
            try:
                migration.migrate(db, table)
            except Exception:
                logger.error(
                    "Schema migration v%d → v%d failed",
                    migration.from_version,
                    migration.to_version,
                    exc_info=True,
                )
                raise
            _write_schema_version(db, migration.to_version)
            stored = migration.to_version
            logger.info("Schema migration to v%d complete", stored)

        if stored != target:
            logger.warning(
                "Schema at v%d but target is v%d — missing migrations?",
                stored,
                target,
            )
    else:
        logger.debug("Schema version %d is current", stored)

    # Ensure metadata table exists (bootstrap for pre-migration databases).
    if _METADATA_TABLE not in db.list_tables().tables:
        _write_schema_version(db, stored)
        logger.info("Created schema metadata (version %d)", stored)

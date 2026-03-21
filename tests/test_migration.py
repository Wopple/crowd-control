"""Tests for schema migration and version tracking."""

from __future__ import annotations

from datetime import UTC, datetime

import lancedb
import pyarrow as pa
import pytest

from crowd_control.storage.db import LearningStore, _make_schema
from crowd_control.storage.migration import (
    CURRENT_SCHEMA_VERSION,
    Migration,
    _METADATA_TABLE,
    _write_schema_version,
    read_schema_version,
    run_migrations,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _create_v1_table(db_path: str, vector_dimensions: int = 8, num_rows: int = 3) -> None:
    """Create a learnings table at v1 schema WITHOUT a metadata table.

    Simulates a pre-migration database — the state users have before
    upgrading to a migration-aware version.
    """
    db = lancedb.connect(db_path)
    schema = _make_schema(vector_dimensions)
    table = db.create_table("learnings", schema=schema)
    if num_rows > 0:
        norm = vector_dimensions**0.5
        rows = [
            {
                "id": f"fixture-{i}",
                "vector": [float(i + 1) / norm] * vector_dimensions,
                "text": f"Fixture learning {i}",
                "category": "debugging_insight",
                "tags": ["python"],
                "project": "/test/project",
                "session_id": "sess-fixture",
                "git_sha": "abc123",
                "timestamp": datetime(2025, 1, 15, 10, 0, tzinfo=UTC),
                "confidence": 0.8,
                "active_count": 0,
                "stale": False,
                "shared": False,
            }
            for i in range(num_rows)
        ]
        table.add(rows)


def _create_v1_table_with_metadata(
    db_path: str, vector_dimensions: int = 8, num_rows: int = 3
) -> None:
    """Create a v1 table WITH metadata stamped at version 1."""
    _create_v1_table(db_path, vector_dimensions, num_rows)
    db = lancedb.connect(db_path)
    _write_schema_version(db, 1)


def _sample_migration(
    from_version: int,
    to_version: int,
    column_name: str,
    column_type: pa.DataType = pa.string(),
) -> Migration:
    """Create a test migration that adds a column with an idempotency guard."""

    def _migrate(db: lancedb.db.LanceDBConnection, table: lancedb.table.Table) -> None:
        if column_name not in table.schema.names:
            table.add_columns(pa.field(column_name, column_type))

    return Migration(
        from_version=from_version,
        to_version=to_version,
        description=f"Add {column_name} column",
        migrate=_migrate,
    )


# ---------------------------------------------------------------------------
# read_schema_version / _write_schema_version
# ---------------------------------------------------------------------------


class TestReadSchemaVersion:
    def test_no_metadata_table(self, tmp_path):
        """Returns 1 when metadata table does not exist."""
        db_path = str(tmp_path / "db")
        _create_v1_table(db_path)
        db = lancedb.connect(db_path)
        assert read_schema_version(db) == 1

    def test_with_metadata(self, tmp_path):
        """Returns the stamped version from metadata."""
        db_path = str(tmp_path / "db")
        db = lancedb.connect(db_path)
        _write_schema_version(db, 3)
        assert read_schema_version(db) == 3


class TestWriteSchemaVersion:
    def test_creates_table(self, tmp_path):
        """Creates metadata table when it does not exist."""
        db_path = str(tmp_path / "db")
        db = lancedb.connect(db_path)
        _write_schema_version(db, 2)
        assert _METADATA_TABLE in db.list_tables().tables
        assert read_schema_version(db) == 2

    def test_updates_existing(self, tmp_path):
        """Overwrites existing version without creating duplicate rows."""
        db_path = str(tmp_path / "db")
        db = lancedb.connect(db_path)
        _write_schema_version(db, 1)
        _write_schema_version(db, 2)
        assert read_schema_version(db) == 2
        # Verify exactly one row (no duplicates)
        table = db.open_table(_METADATA_TABLE)
        rows = table.search().where("key = 'schema_version'").to_list()
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# stamp_initial_version (via LearningStore create path)
# ---------------------------------------------------------------------------


class TestStampInitialVersion:
    def test_fresh_table_stamped_with_current_version(self, tmp_path):
        """New LearningStore stamps metadata at CURRENT_SCHEMA_VERSION."""
        db_path = str(tmp_path / "db")
        LearningStore(db_path, vector_dimensions=8)
        db = lancedb.connect(db_path)
        assert _METADATA_TABLE in db.list_tables().tables
        assert read_schema_version(db) == CURRENT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# run_migrations
# ---------------------------------------------------------------------------


class TestRunMigrations:
    def test_bootstrap_existing_table_no_metadata(self, tmp_path):
        """Pre-migration DB gets metadata table created on first init."""
        db_path = str(tmp_path / "db")
        _create_v1_table(db_path)

        # Open via LearningStore — this triggers run_migrations
        store = LearningStore(db_path)

        db = lancedb.connect(db_path)
        assert _METADATA_TABLE in db.list_tables().tables
        assert read_schema_version(db) == CURRENT_SCHEMA_VERSION
        # Original data preserved
        assert store.count() == 3

    def test_downgrade_raises(self, tmp_path):
        """Opening a DB with a newer schema than the code expects raises."""
        db_path = str(tmp_path / "db")
        _create_v1_table(db_path)
        db = lancedb.connect(db_path)
        _write_schema_version(db, 5)

        with pytest.raises(RuntimeError, match="newer than this version"):
            run_migrations(db, "learnings", target_version=1)

    def test_migration_applies_pending_steps(self, tmp_path):
        """Two migrations (v1->v2, v2->v3) both apply and update version."""
        db_path = str(tmp_path / "db")
        _create_v1_table_with_metadata(db_path)

        migrations = [
            _sample_migration(1, 2, "col_a"),
            _sample_migration(2, 3, "col_b"),
        ]

        db = lancedb.connect(db_path)
        run_migrations(db, "learnings", migrations=migrations, target_version=3)

        assert read_schema_version(db) == 3
        table = db.open_table("learnings")
        assert "col_a" in table.schema.names
        assert "col_b" in table.schema.names
        # Original rows preserved
        assert table.count_rows() == 3

    def test_migration_idempotent(self, tmp_path):
        """Running the same migration sequence twice causes no errors."""
        db_path = str(tmp_path / "db")
        _create_v1_table_with_metadata(db_path)

        migrations = [
            _sample_migration(1, 2, "col_a"),
            _sample_migration(2, 3, "col_b"),
        ]

        db = lancedb.connect(db_path)
        run_migrations(db, "learnings", migrations=migrations, target_version=3)
        # Second run — should be a no-op
        run_migrations(db, "learnings", migrations=migrations, target_version=3)

        assert read_schema_version(db) == 3
        table = db.open_table("learnings")
        assert "col_a" in table.schema.names
        assert "col_b" in table.schema.names
        assert table.count_rows() == 3

    def test_per_step_version_update(self, tmp_path):
        """Version is committed after each step; partial failure preserves progress."""
        db_path = str(tmp_path / "db")
        _create_v1_table_with_metadata(db_path)

        def _failing_migrate(db: lancedb.db.LanceDBConnection, table: lancedb.table.Table) -> None:
            raise RuntimeError("intentional failure")

        migrations = [
            _sample_migration(1, 2, "col_ok"),
            Migration(
                from_version=2,
                to_version=3,
                description="Failing migration",
                migrate=_failing_migrate,
            ),
        ]

        db = lancedb.connect(db_path)
        with pytest.raises(RuntimeError, match="intentional failure"):
            run_migrations(db, "learnings", migrations=migrations, target_version=3)

        # v1->v2 succeeded and was committed
        assert read_schema_version(db) == 2
        table = db.open_table("learnings")
        assert "col_ok" in table.schema.names

    def test_version_skip_applies_all_intermediate(self, tmp_path):
        """Three-step chain (v1->v2->v3->v4) applies all steps."""
        db_path = str(tmp_path / "db")
        _create_v1_table_with_metadata(db_path)

        migrations = [
            _sample_migration(1, 2, "col_x"),
            _sample_migration(2, 3, "col_y"),
            _sample_migration(3, 4, "col_z"),
        ]

        db = lancedb.connect(db_path)
        run_migrations(db, "learnings", migrations=migrations, target_version=4)

        assert read_schema_version(db) == 4
        table = db.open_table("learnings")
        assert "col_x" in table.schema.names
        assert "col_y" in table.schema.names
        assert "col_z" in table.schema.names

    def test_no_migrations_needed(self, tmp_path):
        """Reopening a store at the current version is a no-op."""
        db_path = str(tmp_path / "db")
        store1 = LearningStore(db_path, vector_dimensions=8)
        del store1

        # Reopen — no migrations should run
        store2 = LearningStore(db_path)
        db = lancedb.connect(db_path)
        assert read_schema_version(db) == CURRENT_SCHEMA_VERSION
        assert store2.count() == 0


# ---------------------------------------------------------------------------
# Data preservation
# ---------------------------------------------------------------------------


class TestDataPreservation:
    def test_migration_with_existing_data_preserved(self, tmp_path):
        """All original field values survive a migration that adds a column."""
        db_path = str(tmp_path / "db")
        _create_v1_table_with_metadata(db_path, num_rows=5)

        migrations = [_sample_migration(1, 2, "new_col")]

        db = lancedb.connect(db_path)
        run_migrations(db, "learnings", migrations=migrations, target_version=2)

        table = db.open_table("learnings")
        rows = table.search().limit(10).to_list()
        assert len(rows) == 5
        for i, row in enumerate(sorted(rows, key=lambda r: r["id"])):
            assert row["id"] == f"fixture-{i}"
            assert row["text"] == f"Fixture learning {i}"
            assert row["category"] == "debugging_insight"

    def test_migration_on_empty_table(self, tmp_path):
        """Migration on an empty table succeeds and updates schema."""
        db_path = str(tmp_path / "db")
        _create_v1_table_with_metadata(db_path, num_rows=0)

        migrations = [_sample_migration(1, 2, "empty_col")]

        db = lancedb.connect(db_path)
        run_migrations(db, "learnings", migrations=migrations, target_version=2)

        assert read_schema_version(db) == 2
        table = db.open_table("learnings")
        assert "empty_col" in table.schema.names
        assert table.count_rows() == 0

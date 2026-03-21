"""Tests for hook handler logic."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from crowd_control.config import CrowdControlConfig, IngestionConfig
from crowd_control.hooks import (
    _CREATE_NEW_PROCESS_GROUP,
    _DETACHED_PROCESS,
    _build_worker_command,
    _detach_kwargs,
    handle_session_end_hook,
    spawn_worker,
)


@pytest.fixture
def config(tmp_path):
    return CrowdControlConfig(storage_dir=str(tmp_path / "cc"))


@pytest.fixture
def session_file(tmp_path):
    """Create a fake session transcript file."""
    f = tmp_path / "session.jsonl"
    f.write_text('{"type":"message"}\n')
    return f


def _make_event(session_file, **overrides):
    event = {
        "session_id": "sess-abc123",
        "transcript_path": str(session_file),
        "cwd": "/some/project",
        "hook_event_name": "SessionEnd",
        "reason": "prompt_input_exit",
    }
    event.update(overrides)
    return event


class TestHandleSessionEndHook:
    def test_queues_session(self, config, session_file):
        event = _make_event(session_file)
        with patch("crowd_control.hooks.spawn_worker", return_value=True):
            result = handle_session_end_hook(event, config)

        assert result.session_path == session_file
        assert result.queue_file is not None
        assert result.skipped_reason is None

        # Verify queue file contents
        data = json.loads(result.queue_file.read_text())
        assert data["session_id"] == "sess-abc123"
        assert data["session_path"] == str(session_file)
        assert data["project"] == "/some/project"
        assert "queued_at" in data

    def test_creates_queue_dir(self, config, session_file):
        queue_dir = Path(config.storage_dir) / "queue"
        assert not queue_dir.exists()

        event = _make_event(session_file)
        with patch("crowd_control.hooks.spawn_worker", return_value=True):
            result = handle_session_end_hook(event, config)

        assert queue_dir.exists()
        assert result.queue_file.parent == queue_dir

    def test_missing_transcript(self, config, tmp_path):
        event = _make_event(
            tmp_path / "nonexistent.jsonl",
            transcript_path=str(tmp_path / "nonexistent.jsonl"),
        )
        result = handle_session_end_hook(event, config)
        assert result.skipped_reason is not None
        assert "not found" in result.skipped_reason

    def test_missing_required_fields(self, config):
        result = handle_session_end_hook({}, config)
        assert result.skipped_reason is not None
        assert "missing fields" in result.skipped_reason
        assert "session_id" in result.skipped_reason

    def test_missing_some_fields(self, config):
        event = {"session_id": "abc"}
        result = handle_session_end_hook(event, config)
        assert "transcript_path" in result.skipped_reason
        assert "cwd" in result.skipped_reason

    def test_idempotent(self, config, session_file):
        event = _make_event(session_file)
        with patch("crowd_control.hooks.spawn_worker", return_value=True):
            result1 = handle_session_end_hook(event, config)
            result2 = handle_session_end_hook(event, config)

        assert result1.queue_file == result2.queue_file
        # Only one file in queue directory
        queue_dir = Path(config.storage_dir) / "queue"
        assert len(list(queue_dir.glob("*.json"))) == 1

    def test_auto_ingest_disabled(self, tmp_path, session_file):
        config = CrowdControlConfig(
            storage_dir=str(tmp_path / "cc"),
            ingestion=IngestionConfig(auto_ingest=False),
        )
        event = _make_event(session_file)
        result = handle_session_end_hook(event, config)
        assert result.skipped_reason == "auto_ingest disabled"

        # No queue file written
        queue_dir = Path(config.storage_dir) / "queue"
        assert not queue_dir.exists() or not list(queue_dir.glob("*.json"))

    def test_sanitizes_session_id(self, config, session_file):
        event = _make_event(session_file, session_id="../../etc/passwd")
        with patch("crowd_control.hooks.spawn_worker", return_value=True):
            result = handle_session_end_hook(event, config)

        # Queue file should not contain path separators
        assert "/" not in result.queue_file.name
        assert "\\" not in result.queue_file.name
        assert ".." not in result.queue_file.name

    def test_spawns_worker(self, config, session_file):
        event = _make_event(session_file)
        with patch("crowd_control.hooks.spawn_worker", return_value=True) as mock_spawn:
            handle_session_end_hook(event, config)

        mock_spawn.assert_called_once_with(config)

    def test_spawn_failure_nonfatal(self, config, session_file):
        event = _make_event(session_file)
        with patch("crowd_control.hooks.spawn_worker", return_value=False):
            result = handle_session_end_hook(event, config)

        # Queue file persists even if spawn fails
        assert result.queue_file is not None
        assert result.queue_file.exists()
        assert result.skipped_reason is None
        assert result.worker_spawned is False

    def test_worker_spawned_true(self, config, session_file):
        event = _make_event(session_file)
        with patch("crowd_control.hooks.spawn_worker", return_value=True):
            result = handle_session_end_hook(event, config)

        assert result.worker_spawned is True


class TestSpawnWorker:
    def test_removes_claudecode(self, config, monkeypatch):
        monkeypatch.setenv("CLAUDECODE", "1")

        with patch("crowd_control.hooks.subprocess.Popen") as mock_popen:
            spawn_worker(config)

        call_kwargs = mock_popen.call_args[1]
        assert "CLAUDECODE" not in call_kwargs["env"]

    def test_detached_uses_platform_kwargs(self, config):
        with patch("crowd_control.hooks.subprocess.Popen") as mock_popen:
            spawn_worker(config)

        call_kwargs = mock_popen.call_args[1]
        expected = _detach_kwargs()
        for key, value in expected.items():
            assert call_kwargs[key] == value

    def test_uses_build_worker_command(self, config):
        with (
            patch("crowd_control.hooks.subprocess.Popen") as mock_popen,
            patch(
                "crowd_control.hooks._build_worker_command",
                return_value=["/usr/bin/crowd-control", "worker"],
            ),
        ):
            spawn_worker(config)

        cmd = mock_popen.call_args[0][0]
        assert cmd == ["/usr/bin/crowd-control", "worker"]

    def test_returns_false_on_failure(self, config):
        with patch("crowd_control.hooks.subprocess.Popen", side_effect=OSError("fail")):
            assert spawn_worker(config) is False


class TestBuildWorkerCommand:
    def test_prefers_console_script(self):
        with patch("crowd_control.hooks.shutil.which", return_value="/usr/local/bin/crowd-control"):
            cmd = _build_worker_command()

        assert cmd == ["/usr/local/bin/crowd-control", "worker"]

    def test_falls_back_to_python_m(self):
        with patch("crowd_control.hooks.shutil.which", return_value=None):
            cmd = _build_worker_command()

        assert cmd[1:] == ["-m", "crowd_control", "worker"]


class TestDetachKwargs:
    def test_posix_uses_start_new_session(self, monkeypatch):
        monkeypatch.setattr("crowd_control.hooks.os.name", "posix")
        assert _detach_kwargs() == {"start_new_session": True}

    def test_windows_uses_creationflags(self, monkeypatch):
        monkeypatch.setattr("crowd_control.hooks.os.name", "nt")
        result = _detach_kwargs()
        assert "creationflags" in result
        flags = result["creationflags"]
        assert flags & _CREATE_NEW_PROCESS_GROUP
        assert flags & _DETACHED_PROCESS

    def test_no_start_new_session_on_windows(self, monkeypatch):
        monkeypatch.setattr("crowd_control.hooks.os.name", "nt")
        assert "start_new_session" not in _detach_kwargs()

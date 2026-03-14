"""Integration tests for CLI hook subcommands."""

from __future__ import annotations

import json
from unittest.mock import patch

from click.testing import CliRunner

from crowd_control.cli import main


class TestHookSessionEnd:
    def test_reads_stdin_and_calls_handler(self, tmp_path):
        session_file = tmp_path / "session.jsonl"
        session_file.write_text('{"type":"message"}\n')

        event = {
            "session_id": "cli-test-123",
            "transcript_path": str(session_file),
            "cwd": "/test/project",
            "hook_event_name": "SessionEnd",
            "reason": "prompt_input_exit",
        }

        runner = CliRunner()
        with patch("crowd_control.hooks.spawn_worker", return_value=True):
            result = runner.invoke(
                main,
                ["hook", "session-end"],
                input=json.dumps(event),
            )

        assert result.exit_code == 0

    def test_invalid_json_exits_zero(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["hook", "session-end"],
            input="not json{{{",
        )
        # Should exit 0 (don't block Claude Code)
        assert result.exit_code == 0
        assert "Invalid JSON" in result.output

    def test_empty_stdin_reports_missing_fields(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["hook", "session-end"],
            input="",
        )
        assert result.exit_code == 0

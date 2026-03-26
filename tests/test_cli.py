from click.testing import CliRunner

from crowd_control.cli import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Crowd Control" in result.output


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0


def test_status_command():
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0


def test_search_command_exists():
    runner = CliRunner()
    result = runner.invoke(main, ["search", "--help"])
    assert result.exit_code == 0
    assert "QUERY" in result.output
    assert "--limit" in result.output
    assert "--project" in result.output
    assert "--category" in result.output


def test_add_command_exists():
    runner = CliRunner()
    result = runner.invoke(main, ["add", "--help"])
    assert result.exit_code == 0
    assert "TEXT" in result.output
    assert "--category" in result.output
    assert "--tag" in result.output
    assert "--project" in result.output


def test_add_invalid_category():
    runner = CliRunner()
    result = runner.invoke(main, ["add", "some text", "--category", "bogus"])
    assert result.exit_code != 0
    assert "Invalid category" in result.output


def test_status_command_has_project_option():
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--help"])
    assert result.exit_code == 0
    assert "--project" in result.output


def test_list_command_has_all_flag():
    runner = CliRunner()
    result = runner.invoke(main, ["list", "--help"])
    assert result.exit_code == 0
    assert "--all" in result.output


def test_list_all_and_project_conflict():
    runner = CliRunner()
    result = runner.invoke(main, ["list", "--all", "--project", "/foo"])
    assert result.exit_code != 0
    assert "Cannot use --all and --project" in result.output


def test_export_command_has_all_flag():
    runner = CliRunner()
    result = runner.invoke(main, ["export", "--help"])
    assert result.exit_code == 0
    assert "--all" in result.output


def test_export_all_and_project_conflict():
    runner = CliRunner()
    result = runner.invoke(main, ["export", "--all", "--project", "/foo"])
    assert result.exit_code != 0
    assert "Cannot use --all and --project" in result.output

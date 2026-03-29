"""Tests for project identity resolution (.crowd-control file)."""

import os

import pytest

from crowd_control.project import (
    ProjectConfig,
    find_project_file,
    load_project_config,
    resolve_project,
    validate_project_name,
)

# ---------------------------------------------------------------------------
# validate_project_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["my-app", "My App", "project_123", "a", "a" * 128],
)
def test_validate_project_name_valid(name: str) -> None:
    validate_project_name(name)  # should not raise


def test_validate_project_name_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_project_name("")


def test_validate_project_name_rejects_absolute_path() -> None:
    with pytest.raises(ValueError, match="path"):
        validate_project_name("/usr/local/my-app")


def test_validate_project_name_rejects_windows_drive() -> None:
    with pytest.raises(ValueError):
        validate_project_name("C:\\Users\\me\\project")


@pytest.mark.parametrize("name", ["my/app", "my\\app"])
def test_validate_project_name_rejects_separators(name: str) -> None:
    with pytest.raises(ValueError):
        validate_project_name(name)


def test_validate_project_name_rejects_oversized() -> None:
    with pytest.raises(ValueError, match="128"):
        validate_project_name("a" * 129)


# ---------------------------------------------------------------------------
# find_project_file
# ---------------------------------------------------------------------------


def test_find_project_file_at_start_dir(tmp_path) -> None:
    config_file = tmp_path / ".crowd-control"
    config_file.write_text('[project]\nname = "x"\n')

    assert find_project_file(tmp_path) == config_file


def test_find_project_file_in_ancestor(tmp_path) -> None:
    config_file = tmp_path / ".crowd-control"
    config_file.write_text('[project]\nname = "x"\n')

    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)

    assert find_project_file(nested) == config_file


def test_find_project_file_not_found(tmp_path) -> None:
    # tmp_path itself has no .crowd-control, and the walk-up won't find one
    # in any ancestor within the temp directory tree.  This test relies on
    # there being no .crowd-control between tmp_path and /.
    assert find_project_file(tmp_path) is None


def test_find_project_file_ignores_directories(tmp_path) -> None:
    """A directory named .crowd-control should not be treated as the config."""
    (tmp_path / ".crowd-control").mkdir()
    assert find_project_file(tmp_path) is None


# ---------------------------------------------------------------------------
# load_project_config
# ---------------------------------------------------------------------------


def test_load_project_config_valid(tmp_path) -> None:
    path = tmp_path / ".crowd-control"
    path.write_text('[project]\nname = "my-app"\n')

    config = load_project_config(path)
    assert config == ProjectConfig(name="my-app")


def test_load_project_config_extra_sections_ignored(tmp_path) -> None:
    path = tmp_path / ".crowd-control"
    path.write_text('[project]\nname = "x"\n\n[other]\nfoo = 1\n')

    config = load_project_config(path)
    assert config.name == "x"


def test_load_project_config_invalid_toml(tmp_path) -> None:
    path = tmp_path / ".crowd-control"
    path.write_text("not valid {{{ toml")

    with pytest.raises(ValueError, match="Invalid TOML"):
        load_project_config(path)


def test_load_project_config_missing_project_section(tmp_path) -> None:
    path = tmp_path / ".crowd-control"
    path.write_text("[other]\nfoo = 1\n")

    with pytest.raises(ValueError, match="Missing \\[project\\] section"):
        load_project_config(path)


def test_load_project_config_missing_name(tmp_path) -> None:
    path = tmp_path / ".crowd-control"
    path.write_text("[project]\n")

    with pytest.raises(ValueError, match="Missing or non-string"):
        load_project_config(path)


def test_load_project_config_invalid_name(tmp_path) -> None:
    path = tmp_path / ".crowd-control"
    path.write_text('[project]\nname = "/looks/like/a/path"\n')

    with pytest.raises(ValueError, match="path"):
        load_project_config(path)


# ---------------------------------------------------------------------------
# resolve_project
# ---------------------------------------------------------------------------


def test_resolve_project_with_config_file(tmp_path) -> None:
    (tmp_path / ".crowd-control").write_text('[project]\nname = "my-app"\n')

    assert resolve_project(tmp_path) == "my-app"


def test_resolve_project_no_config_file(tmp_path) -> None:
    result = resolve_project(tmp_path)
    assert result == str(tmp_path.resolve())


def test_resolve_project_walks_up_from_subdirectory(tmp_path) -> None:
    (tmp_path / ".crowd-control").write_text('[project]\nname = "my-app"\n')

    nested = tmp_path / "src" / "deep" / "nested"
    nested.mkdir(parents=True)

    assert resolve_project(nested) == "my-app"


def test_resolve_project_malformed_toml_falls_back(tmp_path, caplog) -> None:
    (tmp_path / ".crowd-control").write_text("not {{{ valid toml")

    result = resolve_project(tmp_path)
    assert result == str(tmp_path.resolve())
    assert "falling back to path" in caplog.text


def test_resolve_project_missing_name_falls_back(tmp_path) -> None:
    (tmp_path / ".crowd-control").write_text("[project]\n")

    result = resolve_project(tmp_path)
    assert result == str(tmp_path.resolve())


def test_resolve_project_missing_project_section_falls_back(tmp_path) -> None:
    (tmp_path / ".crowd-control").write_text("[other]\nfoo = 1\n")

    result = resolve_project(tmp_path)
    assert result == str(tmp_path.resolve())


@pytest.mark.skipif(os.name == "nt", reason="chmod not enforced on Windows")
def test_resolve_project_unreadable_file_falls_back(tmp_path, caplog) -> None:
    config_file = tmp_path / ".crowd-control"
    config_file.write_text('[project]\nname = "my-app"\n')
    config_file.chmod(0o000)

    try:
        result = resolve_project(tmp_path)
        assert result == str(tmp_path.resolve())
        assert "falling back to path" in caplog.text
    finally:
        config_file.chmod(0o644)

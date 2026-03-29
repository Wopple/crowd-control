"""Project identity resolution via .crowd-control configuration file."""

from __future__ import annotations

import logging
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_FILE_NAME = ".crowd-control"
_MAX_NAME_LENGTH = 128
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\]")


@dataclass(frozen=True)
class ProjectConfig:
    """Per-project configuration read from a .crowd-control file."""

    name: str


def validate_project_name(name: str) -> None:
    """Validate a project name.

    Raises ValueError if the name is empty, too long, or looks like a
    filesystem path (starts with ``/``, a Windows drive letter, or contains
    path separators).
    """
    if not name:
        raise ValueError("Project name must not be empty")

    if len(name) > _MAX_NAME_LENGTH:
        raise ValueError(f"Project name must not exceed {_MAX_NAME_LENGTH} characters")

    if name.startswith("/"):
        raise ValueError("Project name must not start with '/' (looks like a path)")

    if _WINDOWS_DRIVE_RE.match(name):
        raise ValueError("Project name must not start with a drive letter (looks like a path)")

    if "/" in name or "\\" in name:
        raise ValueError("Project name must not contain path separators ('/' or '\\\\')")


def find_project_file(start_dir: Path) -> Path | None:
    """Walk from *start_dir* upward looking for a ``.crowd-control`` file.

    Returns the path to the first file found, or ``None`` if the filesystem
    root is reached without finding one.
    """
    current = start_dir.resolve()
    while True:
        candidate = current / PROJECT_FILE_NAME
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


def load_project_config(path: Path) -> ProjectConfig:
    """Parse a ``.crowd-control`` TOML file into a :class:`ProjectConfig`.

    Raises :class:`ValueError` on invalid TOML, missing ``[project]``
    section, missing ``name`` key, or an invalid name.
    """
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Invalid TOML in {path}: {e}") from e

    project_section = raw.get("project")
    if not isinstance(project_section, dict):
        raise ValueError(f"Missing [project] section in {path}")

    name = project_section.get("name")
    if not isinstance(name, str):
        raise ValueError(f"Missing or non-string 'name' in [project] section of {path}")

    validate_project_name(name)
    return ProjectConfig(name=name)


def resolve_project(start_dir: Path | None = None) -> str:
    """Determine the stable project identifier for *start_dir*.

    Walks up from *start_dir* (default :func:`Path.cwd`) looking for a
    ``.crowd-control`` file.  If found and valid, returns the project name.
    Otherwise returns the absolute path as a string (preserving the previous
    behaviour).
    """
    effective_dir = start_dir or Path.cwd()
    effective_dir = effective_dir.resolve()

    project_file = find_project_file(effective_dir)
    if project_file is None:
        logger.debug("No .crowd-control file found, using path: %s", effective_dir)
        return str(effective_dir)

    try:
        config = load_project_config(project_file)
    except (ValueError, OSError) as exc:
        logger.warning(
            "Invalid .crowd-control file at %s, falling back to path: %s",
            project_file,
            exc,
        )
        return str(effective_dir)

    logger.debug("Resolved project name %r from %s", config.name, project_file)
    return config.name

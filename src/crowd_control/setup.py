"""Setup logic for configuring Claude Code hooks and MCP server."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from crowd_control.config import CrowdControlConfig

logger = logging.getLogger(__name__)

MCP_ENTRY = {
    "command": "crowd-control",
    "args": ["serve"],
}

HOOK_ENTRY = {
    "hooks": [
        {
            "type": "command",
            "command": "crowd-control hook session-end",
        }
    ]
}


@dataclass
class SetupResult:
    """Result of running setup, for display by the CLI."""

    scope_label: str
    mcp_path: Path
    hooks_path: Path
    storage_dir: Path
    embedding_label: str
    issues: list[str] = field(default_factory=list)


def run_setup(config: CrowdControlConfig, project_scope: bool = False) -> SetupResult:
    """Configure Claude Code to use crowd-control's hooks and MCP server.

    Returns a SetupResult with paths and labels for the CLI to display.
    """
    issues = check_prerequisites()
    if issues:
        return SetupResult(
            scope_label="",
            mcp_path=Path(),
            hooks_path=Path(),
            storage_dir=Path(),
            embedding_label="",
            issues=issues,
        )

    ensure_storage_dir(config)
    ensure_default_config(config)

    if project_scope:
        target_dir = Path.cwd()
        scope_label = f"project: {target_dir}"
    else:
        target_dir = Path.home()
        scope_label = "global"

    mcp_path = configure_mcp_server(target_dir, project_scope)
    logger.info("Configured MCP server at %s", mcp_path)
    hooks_path = configure_hooks(target_dir)
    logger.info("Configured hooks at %s", hooks_path)

    return SetupResult(
        scope_label=scope_label,
        mcp_path=mcp_path,
        hooks_path=hooks_path,
        storage_dir=Path(config.storage_dir).expanduser(),
        embedding_label=f"{config.embedding.provider}/{config.embedding.model}",
    )


def check_prerequisites() -> list[str]:
    """Check that prerequisites are met. Returns list of issues (empty = all good)."""
    issues = []
    if not shutil.which("crowd-control"):
        issues.append("crowd-control is not on PATH")
    return issues


def ensure_storage_dir(config: CrowdControlConfig) -> None:
    """Create the storage directory if it doesn't exist."""
    storage = Path(config.storage_dir).expanduser()
    storage.mkdir(parents=True, exist_ok=True)


def ensure_default_config(config: CrowdControlConfig) -> None:
    """Write default config.toml if it doesn't exist."""
    config_path = Path(config.storage_dir).expanduser() / "config.toml"
    if config_path.exists():
        return

    default_template = Path(__file__).parent / "default_config.toml"
    if default_template.exists():
        config_path.write_text(default_template.read_text())


def configure_mcp_server(target_dir: Path, project_scope: bool) -> Path:
    """Add/update the crowd-control MCP server entry."""
    if project_scope:
        mcp_path = target_dir / ".mcp.json"
    else:
        mcp_path = target_dir / ".claude.json"

    existing = _read_json(mcp_path)
    if "mcpServers" not in existing:
        existing["mcpServers"] = {}
    existing["mcpServers"]["crowd-control"] = MCP_ENTRY
    _write_json(mcp_path, existing)
    return mcp_path


def configure_hooks(target_dir: Path) -> Path:
    """Add/update the crowd-control hook entries."""
    hooks_path = target_dir / ".claude" / "settings.json"

    existing = _read_json(hooks_path)
    if "hooks" not in existing:
        existing["hooks"] = {}

    # Merge SessionEnd hooks: replace any existing crowd-control entry, preserve others
    session_end_hooks = existing["hooks"].get("SessionEnd", [])
    session_end_hooks = _replace_crowd_control_hook(session_end_hooks)
    existing["hooks"]["SessionEnd"] = session_end_hooks

    _write_json(hooks_path, existing)
    return hooks_path


def _replace_crowd_control_hook(hook_entries: list[dict]) -> list[dict]:
    """Replace or append the crowd-control hook in a list of hook entries."""
    result = []
    found = False
    for entry in hook_entries:
        if _is_crowd_control_hook(entry):
            result.append(HOOK_ENTRY)
            found = True
        else:
            result.append(entry)
    if not found:
        result.append(HOOK_ENTRY)
    return result


def _is_crowd_control_hook(entry: dict) -> bool:
    """Check if a hook entry belongs to crowd-control."""
    hooks = entry.get("hooks", [])
    for h in hooks:
        command = h.get("command", "")
        if "crowd-control" in command:
            return True
    return False


def _read_json(path: Path) -> dict:
    """Read a JSON file, returning empty dict if missing or invalid."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, data: dict) -> None:
    """Write a dict as formatted JSON, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from src.core.exceptions import ToolError

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30


async def _run_git(args: list[str], cwd: str, timeout: int = _DEFAULT_TIMEOUT) -> dict:
    """Run a git subcommand and return stdout/stderr/returncode."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise ToolError(
            f"Failed to start git: {exc}",
            tool_name="git_commit",
            code="SUBPROCESS_ERROR",
        ) from exc

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.communicate()
        raise ToolError(
            f"git {args[0]} timed out after {timeout}s",
            tool_name="git_commit",
            code="TIMEOUT",
        ) from exc

    return {
        "stdout": stdout_bytes.decode("utf-8", errors="replace"),
        "stderr": stderr_bytes.decode("utf-8", errors="replace"),
        "returncode": proc.returncode,
    }


async def git_commit_files(
    files: list[str],
    message: str,
    repo_path: str = ".",
    push: bool = False,
    timeout: int = _DEFAULT_TIMEOUT,
) -> dict:
    """Stage files, create a commit, and optionally push.

    Args:
        files: File paths relative to repo_path to stage.
        message: Commit message.
        repo_path: Absolute or relative path to the git repository root.
        push: If True, run `git push` after committing.
        timeout: Per-command timeout in seconds.

    Returns:
        {"success": bool, "stdout": str, "stderr": str}

    Raises:
        ToolError if git add or git commit exits non-zero.
    """
    cwd = str(Path(repo_path).resolve())
    combined_out: list[str] = []
    combined_err: list[str] = []

    add_result = await _run_git(["add", "--"] + files, cwd=cwd, timeout=timeout)
    combined_out.append(add_result["stdout"])
    combined_err.append(add_result["stderr"])
    if add_result["returncode"] != 0:
        raise ToolError(
            f"git add failed: {add_result['stderr'].strip()}",
            tool_name="git_commit",
            code="GIT_ADD_FAILED",
        )

    commit_result = await _run_git(["commit", "-m", message], cwd=cwd, timeout=timeout)
    combined_out.append(commit_result["stdout"])
    combined_err.append(commit_result["stderr"])
    if commit_result["returncode"] != 0:
        raise ToolError(
            f"git commit failed: {commit_result['stderr'].strip()}",
            tool_name="git_commit",
            code="GIT_COMMIT_FAILED",
        )

    if push:
        push_result = await _run_git(["push"], cwd=cwd, timeout=timeout)
        combined_out.append(push_result["stdout"])
        combined_err.append(push_result["stderr"])
        if push_result["returncode"] != 0:
            raise ToolError(
                f"git push failed: {push_result['stderr'].strip()}",
                tool_name="git_commit",
                code="GIT_PUSH_FAILED",
            )

    return {
        "success": True,
        "stdout": "\n".join(combined_out).strip(),
        "stderr": "\n".join(combined_err).strip(),
    }

from __future__ import annotations

import asyncio
import sys

from src.core.exceptions import ToolError

_DEFAULT_TIMEOUT = 10


async def run_code(code: str, timeout: int = _DEFAULT_TIMEOUT) -> dict:
    """Execute Python code in a subprocess sandbox.

    Returns {"stdout": str, "stderr": str, "returncode": int}.
    Raises ToolError on timeout or process creation failure.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise ToolError(
            f"Failed to start subprocess: {exc}",
            tool_name="run_code",
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
            f"Code execution timed out after {timeout}s",
            tool_name="run_code",
            code="TIMEOUT",
        ) from exc

    return {
        "stdout": stdout_bytes.decode("utf-8", errors="replace"),
        "stderr": stderr_bytes.decode("utf-8", errors="replace"),
        "returncode": proc.returncode,
    }

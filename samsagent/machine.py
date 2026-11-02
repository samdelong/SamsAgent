import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from mcp.server.fastmcp import FastMCP

from .process import identity
from .store import atomic_write, uid


def build_server(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    server = FastMCP("SamsAgent machine")
    children = {}

    def job_path(job_id):
        if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise ValueError("Invalid process handle")
        return root / f"{job_id}.json"

    @server.tool()
    def file_read(path: str, offset: int = 0, limit: int = 12000) -> dict:
        """Read a UTF-8 file page; offsets count characters. Use pages for large files."""
        if offset < 0 or not 1 <= limit <= 64000:
            raise ValueError("offset >= 0 and 1 <= limit <= 64000 required")
        with open(Path(path).expanduser(), errors="replace") as f:
            # Bound memory even when paging a very large experiment log.
            remaining = offset
            while remaining:
                chunk = f.read(min(remaining, 64000))
                if not chunk:
                    break
                remaining -= len(chunk)
            text = f.read(limit + 1)
        return {"text": text[:limit], "next_offset": offset + limit if len(text) > limit else None}

    @server.tool()
    def file_write(path: str, content: str) -> dict:
        """Atomically replace a UTF-8 file, creating parent directories."""
        target = Path(path).expanduser().resolve()
        atomic_write(target, content)
        return {"path": str(target), "characters": len(content)}

    @server.tool()
    def file_search(path: str, query: str, max_matches: int = 50) -> list[dict]:
        """Search literal text in a file, returning matching line numbers and previews."""
        if not query or not 1 <= max_matches <= 200:
            raise ValueError("Nonempty query and max_matches 1..200 required")
        matches = []
        with open(Path(path).expanduser(), errors="replace") as f:
            for n, line in enumerate(f, 1):
                if query in line:
                    matches.append({"line": n, "text": line[:2000]})
                    if len(matches) == max_matches:
                        break
        return matches

    @server.tool()
    async def process_start(command: str, cwd: str, owner: str = "manual") -> dict:
        """Start a shell command and return a durable process handle; poll for output."""
        directory = Path(cwd).expanduser().resolve()
        if not command.strip() or not directory.is_dir():
            raise ValueError("A command and existing working directory are required")
        handle = uid()
        path = job_path(handle)
        job = {"id": handle, "command": command, "cwd": str(directory),
               "owner": owner, "status": "starting", "pid": None}
        atomic_write(path, json.dumps(job))
        process = subprocess.Popen([sys.executable, "-m", "samsagent.process", str(path)],
                                   start_new_session=True, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        children[handle] = process
        for _ in range(100):
            current = json.loads(path.read_text())
            if current["status"] != "starting":
                return current
            await asyncio.sleep(.01)
        return {**job, "status": "uncertain", "detail": "Supervisor did not acknowledge startup"}

    @server.tool()
    def process_status(job_id: str) -> dict:
        """Inspect a process, including its exit code if available."""
        job = json.loads(job_path(job_id).read_text())
        child = children.get(job_id)
        if child:
            child.poll()
        if job["status"] in {"running", "starting"}:
            pid = job.get("pid")
            alive = (child is not None and child.returncode is None) or (
                pid and job.get("identity") and identity(pid) == job["identity"])
            if not alive:
                job["status"] = "uncertain"
        return job

    @server.tool()
    def process_output(job_id: str, offset: int = 0, limit: int = 12000) -> dict:
        """Read a page of combined stdout/stderr from a process."""
        path = job_path(job_id).with_suffix(".log")
        return file_read(str(path), offset, limit) if path.exists() else {"text": "", "next_offset": None}

    @server.tool()
    async def process_terminate(job_id: str) -> dict:
        """Terminate a tracked process group; verify PID identity before sending signals."""
        job = process_status(job_id)
        if job["status"] != "running":
            return job
        pid = job["pid"]
        child = children.get(job_id)
        if not ((job.get("identity") and identity(pid) == job["identity"]) or
                (child and child.poll() is None and child.pid == pid)):
            raise ValueError("Cannot verify process identity; inspect it manually")
        try:
            os.killpg(pid, signal.SIGTERM)
            await asyncio.sleep(.2)
            # The PID belongs to a new session; its surviving children share the group.
            try:
                os.killpg(pid, signal.SIGKILL)
            except PermissionError:
                # macOS can return EPERM, not ESRCH, for an already exited group.
                if not child or child.poll() is None:
                    raise
        except ProcessLookupError:
            pass
        job["status"] = "cancelled"
        atomic_write(job_path(job_id), json.dumps(job))
        if child:
            await asyncio.to_thread(child.wait)
        return job

    @server.tool()
    async def process_terminate_owner(owner: str) -> list[dict]:
        """Terminate tracked commands belonging to a cancelled session."""
        results = []
        for path in root.glob("*.json"):
            if json.loads(path.read_text()).get("owner") == owner:
                results.append(await process_terminate(path.stem))
        return results

    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True, type=Path)
    args = parser.parse_args()
    build_server(args.state_dir).run(transport="stdio")


if __name__ == "__main__":
    main()


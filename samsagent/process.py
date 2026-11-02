"""Detached process supervisor: writes exit state even if the MCP server restarts."""
import json
import os
from pathlib import Path
import subprocess
import sys

from .store import atomic_write


def identity(pid):
    try:
        # starttime guards against signalling an unrelated process after PID reuse.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def main():
    path = Path(sys.argv[1])
    job = json.loads(path.read_text())
    job.update(pid=os.getpid(), identity=identity(os.getpid()), status="running")
    atomic_write(path, json.dumps(job))
    try:
        with open(path.with_suffix(".log"), "ab", buffering=0) as log:
            result = subprocess.run(job["command"], shell=True, executable="/bin/sh",
                                    cwd=job["cwd"], stdout=log, stderr=log)
        job.update(status="completed", returncode=result.returncode)
    except Exception as exc:
        job.update(status="failed", error=str(exc))
    atomic_write(path, json.dumps(job))


if __name__ == "__main__":
    main()


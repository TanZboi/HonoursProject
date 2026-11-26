from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

# Libraries whose versions affect results. Missing ones (gsplat and sam2 on the Mac)
# are recorded as null.
TRACKED_PACKAGES = ["seg3d", "numpy", "scipy", "scikit-learn", "torch", "gsplat", "sam2"]


def sha256_file(path: str | Path, chunk_bytes: int = 1 << 24) -> str:
    """SHA-256 of a file, read in 16 MB chunks so a large splat never sits in memory twice."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> dict[str, Any]:
    """Current commit hash and whether there are uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def default_run_dir(command: str) -> Path:
    """runs/<YYYYmmdd-HHMMSS>-<command>, inside the repository."""
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return REPO_ROOT / "runs" / f"{stamp}-{command}"


class RunLog:
    """Collects log lines, metrics and provenance for one command, and writes them out."""

    def __init__(self, run_dir: str | Path, command: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.command = command
        self.stages: dict[str, dict] = {}  # filled by seg3d.utils.timing.stage
        self._log_file = open(self.run_dir / "log.txt", "a", encoding="utf-8")

    def log(self, message: str) -> None:
        stamp = _dt.datetime.now().strftime("%H:%M:%S")
        self._log_file.write(f"[{stamp}] {message}\n")
        self._log_file.flush()

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump({"command": self.command, **metrics}, f, indent=2)

    def write_provenance(
        self,
        config: dict[str, Any] | None,
        input_files: list[str | Path] = (),
    ) -> None:
        provenance = {
            "command": self.command,
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "inputs": {str(p): {"sha256": sha256_file(p)} for p in input_files},
            "config": config,
            "seed": None if config is None else config.get("seed"),
            "git": git_commit(),
            "versions": library_versions(),
            "stages": self.stages,
        }
        with open(self.run_dir / "provenance.json", "w") as f:
            json.dump(provenance, f, indent=2)

    def close(self) -> None:
        self._log_file.close()

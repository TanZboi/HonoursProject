"""T0.3: `seg3d inspect tests/fixtures/toy.ply` prints the expected counts."""

import json
import subprocess
import sys

from conftest import REPO_ROOT


def test_inspect_prints_expected_counts(toy_ply, tmp_path):
    run_dir = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, "-m", "seg3d.cli", "inspect", str(toy_ply), "--run-dir", str(run_dir)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    out = result.stdout
    assert "Gaussians:       3,000" in out
    assert "SH degree:       3" in out
    assert "Properties:      60" in out
    assert "Extra props:     confidence" in out
    assert "Non-finite rows: 0" in out

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["count"] == 3000 and metrics["sh_degree"] == 3
    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert str(toy_ply) in provenance["inputs"]
    assert "load" in provenance["stages"]
    assert (run_dir / "log.txt").exists()


def test_inspect_does_not_change_the_input(toy_ply):
    """The input splat is read-only (SPEC 3.2 rule 3)."""
    before = toy_ply.read_bytes()
    subprocess.run([sys.executable, "-m", "seg3d.cli", "inspect", str(toy_ply)],
                   cwd=REPO_ROOT, capture_output=True, check=True)
    assert toy_ply.read_bytes() == before

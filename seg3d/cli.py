from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

import numpy as np
import tyro
from scipy.special import expit  # numerically stable sigmoid

from seg3d.config import load_config
from seg3d.io.ply import read_ply, sh_degree_from_properties
from seg3d.utils.runlog import RunLog
from seg3d.utils.timing import stage

# Properties a standard 3DGS PLY contains. Anything else is reported as "extra".
STANDARD_PROPERTY = re.compile(r"^(x|y|z|nx|ny|nz|f_dc_\d+|f_rest_\d+|opacity|scale_\d|rot_\d)$")
REQUIRED = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
            "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def summarise_splat(header, data: np.ndarray) -> dict:
    """Counts and statistics of a splat, computed from the stored (raw) values.

    Opacity is stored as a logit and scale as a log, so they are activated (sigmoid,
    exp) here for reporting only; the data itself is never changed.
    """
    names = list(data.dtype.names)
    missing = [n for n in REQUIRED if n not in names]
    if missing:
        raise ValueError(f"Not a 3DGS splat: missing properties {missing}")

    means = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    opacity = expit(data["opacity"].astype(np.float64))
    scales = np.exp(np.column_stack([data[f"scale_{i}"] for i in range(3)]).astype(np.float64))

    # Non-finite values would poison every statistic below, so count them first.
    as_float = np.column_stack([data[n].astype(np.float64) for n in names])
    n_nonfinite_rows = int((~np.isfinite(as_float)).any(axis=1).sum())
    finite = np.isfinite(means).all(axis=1)

    return {
        "count": int(len(data)),
        "format": header.format,
        "sh_degree": sh_degree_from_properties(names),
        "n_properties": len(names),
        "properties": [f"{p.name}:{p.type_name}" for p in header.element("vertex").properties],
        "extra_properties": [n for n in names if not STANDARD_PROPERTY.match(n)],
        "nonfinite_rows": n_nonfinite_rows,
        "bounds_min": means[finite].min(axis=0).round(4).tolist(),
        "bounds_max": means[finite].max(axis=0).round(4).tolist(),
        # 1st-99th percentile box: ignores the few far-away "floater" Gaussians.
        "bounds_p01": np.percentile(means[finite], 1, axis=0).round(4).tolist(),
        "bounds_p99": np.percentile(means[finite], 99, axis=0).round(4).tolist(),
        "opacity": {
            "min": round(float(opacity.min()), 4),
            "median": round(float(np.median(opacity)), 4),
            "mean": round(float(opacity.mean()), 4),
            "max": round(float(opacity.max()), 4),
            "fraction_above_half": round(float((opacity > 0.5).mean()), 4),
        },
        "scale": {
            "median_per_axis": np.median(scales, axis=0).round(5).tolist(),
            "max_per_axis": scales.max(axis=0).round(5).tolist(),
            # Largest axis of each Gaussian; its percentiles show if there are huge ones.
            "largest_axis_p50": round(float(np.percentile(scales.max(axis=1), 50)), 5),
            "largest_axis_p99": round(float(np.percentile(scales.max(axis=1), 99)), 5),
        },
    }


def format_summary(path: Path, s: dict) -> str:
    def vec(v):
        return "[" + ", ".join(f"{x:.3f}" for x in v) + "]"

    lines = [
        f"File:            {path}",
        f"Gaussians:       {s['count']:,}",
        f"Format:          {s['format']}",
        f"SH degree:       {s['sh_degree']}",
        f"Properties:      {s['n_properties']}",
        f"Extra props:     {', '.join(s['extra_properties']) or '(none)'}",
        f"Non-finite rows: {s['nonfinite_rows']}",
        f"Bounds min/max:  {vec(s['bounds_min'])} / {vec(s['bounds_max'])}",
        f"Bounds 1%/99%:   {vec(s['bounds_p01'])} / {vec(s['bounds_p99'])}",
        "Opacity:         min {min:.3f}  median {median:.3f}  mean {mean:.3f}  max {max:.3f}  "
        "(>0.5: {fraction_above_half:.1%})".format(**s["opacity"]),
        f"Scale median:    {vec(s['scale']['median_per_axis'])}",
        f"Scale max:       {vec(s['scale']['max_per_axis'])}",
        f"Largest axis:    p50 {s['scale']['largest_axis_p50']:.4f}  p99 {s['scale']['largest_axis_p99']:.4f}",
    ]
    return "\n".join(lines)


def inspect(
    splat: tyro.conf.Positional[Path],
    config: Path | None = None,
    run_dir: Path | None = None,
) -> None:
    """Print Gaussian count, SH degree, properties, bounds, opacity and scale stats."""
    cfg = load_config(config)
    timings: dict = {}
    with stage("load", timings):
        header, data = read_ply(splat)
    summary = summarise_splat(header, data)
    text = format_summary(splat, summary)
    print(text)
    print(f"Load time:       {timings['load']['wall_s']:.2f} s")

    if run_dir is not None:
        run = RunLog(run_dir, "inspect")
        run.stages = timings
        run.log(text)
        run.write_metrics(summary)
        run.write_provenance(cfg.to_dict(), input_files=[splat])
        run.close()


COMMANDS = {
    "inspect": inspect,
}


def main() -> None:
    tyro.extras.subcommand_cli_from_dict(COMMANDS)


if __name__ == "__main__":
    main()

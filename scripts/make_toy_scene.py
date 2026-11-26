from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Allow `python scripts/make_toy_scene.py` from the repo root without installing.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seg3d.io.ply import write_ply  # noqa: E402

SH_C0 = 0.28209479177387814  # degree-0 real spherical-harmonic basis, 1 / (2 sqrt(pi))
SH_DEGREE = 3
N_REST_PER_CHANNEL = (SH_DEGREE + 1) ** 2 - 1  # 15

# (id, name, rgb colour, share of all Gaussians)
OBJECTS = [
    (1, "floor", (0.5, 0.5, 0.5), 0.40),
    (2, "sphere", (0.9, 0.1, 0.1), 0.15),
    (3, "box", (0.1, 0.2, 0.9), 0.15),
    (4, "cylinder", (0.1, 0.8, 0.2), 0.15),
    (5, "cone", (0.9, 0.85, 0.1), 0.15),
]

# Each returns (points (n, 3), surface area). Sampling is uniform over the area.
def sample_floor(rng, n, half_size=2.0):
    xy = rng.uniform(-half_size, half_size, size=(n, 2))
    points = np.column_stack([xy, np.zeros(n)])
    return points, (2 * half_size) ** 2

def sample_sphere(rng, n, centre, radius):
    # A normal distribution in 3D is the same in every direction, so normalised
    # normal samples are uniform on the sphere.
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.asarray(centre) + radius * v, 4 * np.pi * radius**2

def sample_box(rng, n, centre, side):
    # Five faces (the bottom one touches the floor and cannot be seen), all of equal
    # area, so each point picks a face at random.
    faces = [(0, +1), (0, -1), (1, +1), (1, -1), (2, +1)]  # (axis, sign)
    face_index = rng.integers(len(faces), size=n)
    points = rng.uniform(-side / 2, side / 2, size=(n, 3))
    for i, (axis, sign) in enumerate(faces):
        on_face = face_index == i
        points[on_face, axis] = sign * side / 2
    return np.asarray(centre) + points, 5 * side**2

def sample_cylinder(rng, n, base_xy, radius, height):
    # Side wall and top cap, chosen in proportion to their areas.
    side_area = 2 * np.pi * radius * height
    top_area = np.pi * radius**2
    on_top = rng.uniform(size=n) < top_area / (side_area + top_area)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    # On the cap, r = R * sqrt(u) is uniform over the disc's area (plain u would bunch
    # points near the centre). On the side wall, r = R.
    r = np.where(on_top, radius * np.sqrt(rng.uniform(size=n)), radius)
    z = np.where(on_top, height, rng.uniform(0, height, size=n))
    points = np.column_stack([base_xy[0] + r * np.cos(theta), base_xy[1] + r * np.sin(theta), z])
    return points, side_area + top_area

def sample_cone(rng, n, base_xy, radius, height):
    # Lateral surface only. With t the fraction of the way from apex to base, the
    # circle at t has circumference proportional to t, so t = sqrt(u) is uniform by area.
    t = np.sqrt(rng.uniform(size=n))
    theta = rng.uniform(0, 2 * np.pi, size=n)
    r = radius * t
    points = np.column_stack([base_xy[0] + r * np.cos(theta), base_xy[1] + r * np.sin(theta), height * (1 - t)])
    slant = np.hypot(radius, height)
    return points, np.pi * radius * slant

def sample_object(rng, name, n):
    if name == "floor":
        return sample_floor(rng, n)
    if name == "sphere":
        return sample_sphere(rng, n, centre=(-1.0, -1.0, 0.4), radius=0.4)
    if name == "box":
        return sample_box(rng, n, centre=(1.0, -1.0, 0.3), side=0.6)
    if name == "cylinder":
        return sample_cylinder(rng, n, base_xy=(-1.0, 1.0), radius=0.3, height=0.8)
    if name == "cone":
        return sample_cone(rng, n, base_xy=(1.0, 1.0), radius=0.35, height=0.8)
    raise ValueError(name)

def vertex_dtype() -> np.dtype:
    """Property order of a standard 3DGS PLY, plus the extra `confidence` at the end."""
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(3 * N_REST_PER_CHANNEL)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    names += ["confidence"]
    return np.dtype([(name, "<f4") for name in names])

def make_toy_scene(n: int = 50_000, seed: int = 0):
    """Build the toy splat. Returns (structured array, int32 object IDs, {id: name})."""
    rng = np.random.default_rng(seed)

    # Split n between objects by their share; the floor takes any rounding remainder.
    counts = [int(round(share * n)) for (_, _, _, share) in OBJECTS[1:]]
    counts = [n - sum(counts)] + counts

    blocks, ids = [], []
    for (obj_id, name, rgb, _), count in zip(OBJECTS, counts):
        points, area = sample_object(rng, name, count)

        # Spacing between neighbouring Gaussians on this surface ~ sqrt(area / count).
        # A scale of 0.7 x spacing makes neighbours overlap a little, closing the surface.
        base_scale = max(0.7 * np.sqrt(area / count), 0.005)
        log_scales = np.log(base_scale) + rng.normal(0, 0.1, size=(count, 3))

        colour = np.clip(np.asarray(rgb) + rng.normal(0, 0.03, size=(count, 3)), 0, 1)
        f_dc = (colour - 0.5) / SH_C0
        f_rest = rng.normal(0, 0.02, size=(count, 3 * N_REST_PER_CHANNEL))

        opacity = rng.uniform(0.7, 0.98, size=count)
        opacity_logit = np.log(opacity / (1 - opacity))  # inverse of sigmoid

        quats = rng.normal(size=(count, 4))
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)

        confidence = rng.uniform(0, 1, size=count)

        blocks.append(np.column_stack([points, f_dc, f_rest, opacity_logit, log_scales, quats, confidence]))
        ids.append(np.full(count, obj_id, dtype=np.int32))

    columns = np.concatenate(blocks).astype(np.float32)
    gt_ids = np.concatenate(ids)

    order = rng.permutation(n)  # shuffle rows; keep the IDs aligned
    columns, gt_ids = columns[order], gt_ids[order]

    # Pack the plain (n, 60) float array into the structured PLY row layout.
    dtype = vertex_dtype()
    data = np.empty(n, dtype=dtype)
    for i, name in enumerate(dtype.names):
        data[name] = columns[:, i]

    objects = {obj_id: name for (obj_id, name, _, _) in OBJECTS}
    return data, gt_ids, objects

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=50_000, help="number of Gaussians")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("data/toy"))
    parser.add_argument("--overwrite", action="store_true", help="replace existing files")
    args = parser.parse_args()

    data, gt_ids, objects = make_toy_scene(args.n, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    comment = f"seg3d toy scene: scripts/make_toy_scene.py --n {args.n} --seed {args.seed}"
    write_ply(args.out_dir / "toy.ply", data, comments=[comment], overwrite=args.overwrite)
    np.save(args.out_dir / "toy_gt.npy", gt_ids)
    with open(args.out_dir / "toy_objects.json", "w") as f:
        json.dump({str(k): v for k, v in objects.items()}, f, indent=2)

    size_kb = (args.out_dir / "toy.ply").stat().st_size / 1024
    print(f"Wrote {args.n} Gaussians ({size_kb:.0f} KB) to {args.out_dir}")

if __name__ == "__main__":
    main()

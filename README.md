# HonoursProject

Repository to store my 3DGS honours project across the year.

## Setup

There are two machines:

- **MacBook (development):** writes code and runs the CPU tests. gsplat needs an NVIDIA
  GPU, so rendering and everything after it does not run here.
- **Home PC / RBB-335 lab machine (NVIDIA, CUDA 12.x):** runs the GPU tests and the
  real scene.

## Licence

MIT

# Third-party notices

seg3d uses the libraries below as installed dependencies. **No third-party source code is
copied into this repository.** In particular, nothing comes from
`graphdeco-inria/gaussian-splatting`, `diff-gaussian-rasterization`, LangSplat or
GaussianCut, whose licences do not allow it (SPEC 3.2). No GPL or AGPL packages are used;
the PLY reader/writer is written from scratch instead of using `plyfile` (GPL-3.0).

| Package | Used for | Licence |
|---|---|---|
| PyTorch | tensors, autograd, training | BSD-3-Clause |
| gsplat | rasterising Gaussians (GPU machine only) | Apache-2.0 |
| SAM 2 (`facebookresearch/sam2`) and its checkpoints | 2D masks (GPU machine only) | Apache-2.0 |
| NumPy | arrays, PLY body I/O | BSD-3-Clause |
| SciPy | sparse matrices, KD-tree, Hungarian matching | BSD-3-Clause |
| scikit-learn | HDBSCAN, k-means | BSD-3-Clause |
| PyYAML | config files | MIT |
| tyro | command-line interface | MIT |
| imageio | image I/O | BSD-2-Clause |
| Pillow | image I/O | MIT-CMU (HPND) |
| tqdm | progress bars | MIT and MPL-2.0 |
| rich | console output | MIT |
| pytest | tests (development only) | MIT |

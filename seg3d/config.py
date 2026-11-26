from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

# configs/ sits next to the seg3d/ package in the repository.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@dataclass
class SceneConfig:
    up: list[float] | None


@dataclass
class RenderConfig:
    long_side: int
    max_batch: int
    bg_rgb: list[float]
    bg_values: float
    channel_chunk: int


@dataclass
class CamerasConfig:
    n_views: int
    interior_fraction: float
    margin: float
    height_range: list[float]
    yaws: int
    pitches_deg: list[float]
    orbit_radii: list[float]
    fov_deg: float
    filter_long_side: int
    min_coverage: float
    min_depth: float
    max_near_fraction: float
    exclude_radius: float
    exclude_angle_deg: float


@dataclass
class MasksConfig:
    model: str
    points_per_side: int
    pred_iou_thresh: float
    stability_score_thresh: float
    min_mask_region_area: int
    min_pixels: int


@dataclass
class VotingConfig:
    eps: float
    merge_thresh: float
    min_share: float
    min_views: int


@dataclass
class PostConfig:
    min_gaussians: int
    knn: int
    max_dist: float


@dataclass
class TrainConfig:
    loss: str
    dim: int
    long_side: int
    batch_pixels: int
    temperature: float
    margin_same: float
    margin_diff: float
    smooth_weight: float
    smooth_samples: int
    smooth_knn: int
    lr: float
    iters: int
    log_every: int
    ckpt_every: int


@dataclass
class ClusterConfig:
    method: str
    max_samples: int
    min_cluster_size: int
    min_sim: float


@dataclass
class Config:
    seed: int
    scene: SceneConfig
    render: RenderConfig
    cameras: CamerasConfig
    masks: MasksConfig
    voting: VotingConfig
    post: PostConfig
    train: TrainConfig
    cluster: ClusterConfig

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict, for saving into the run folder."""
        return asdict(self)


def _deep_merge(base: dict, update: dict, where: str = "") -> dict:
    """Return `base` with values from `update` written over it, section by section.

    A key in `update` that does not exist in `base` is an error, because it is almost
    certainly a typo (e.g. `camera.n_views` instead of `cameras.n_views`).
    """
    result = copy.deepcopy(base)
    for key, value in update.items():
        if key not in result:
            raise KeyError(f"Unknown config key '{where}{key}'")
        if isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value, where=f"{where}{key}.")
        else:
            result[key] = value
    return result


def _parse_override(text: str) -> dict:
    """Turn 'cameras.n_views=100' into {'cameras': {'n_views': 100}}.

    The value is parsed as YAML, so numbers, lists ('[0,-1,0]'), booleans and null
    all work the same way they do in the config files.
    """
    if "=" not in text:
        raise ValueError(f"Override '{text}' must look like key=value")
    dotted_key, raw_value = text.split("=", 1)
    value = yaml.safe_load(raw_value)
    nested: dict = {}
    current = nested
    parts = dotted_key.strip().split(".")
    for part in parts[:-1]:
        current[part] = {}
        current = current[part]
    current[parts[-1]] = value
    return nested


def _from_dict(data: dict) -> Config:
    """Build the Config dataclass, one section at a time."""
    sections = {}
    for f in fields(Config):
        if f.name == "seed":
            continue
        section_class = f.type if isinstance(f.type, type) else globals()[f.type]
        # SectionClass(**dict) raises TypeError on unknown or missing keys.
        sections[f.name] = section_class(**data[f.name])
    return Config(seed=int(data["seed"]), **sections)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load default.yaml, then the optional run config, then the overrides."""
    with open(DEFAULT_CONFIG_PATH) as f:
        data = yaml.safe_load(f)

    if path is not None and Path(path).resolve() != DEFAULT_CONFIG_PATH:
        with open(path) as f:
            run_values = yaml.safe_load(f) or {}
        data = _deep_merge(data, run_values)

    for text in overrides or []:
        data = _deep_merge(data, _parse_override(text))

    return _from_dict(data)

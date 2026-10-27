#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
青藏高原潜在地质灾害点识别与居民地影响分析 v4.0（SHAP数据驱动权重）

用途
----
1. 以 DEM 为统一分析格网，构造历史灾情分析得到的11项因子：
   多年平均降水、高程、距地震点、距冰湖、坡度、冻土类型、
   距冰川、土地覆盖、距道路、距河流和坡向。
2. 使用 CatBoost + SHAP 得到的相对贡献率计算数据驱动易发性评分。
3. 从高值区提取候选隐患点，并依据环境条件判识潜在灾种。
4. 多年平均降水、距历史地震点和冻土等均进入易发性评分；
   “未来强降雨、未来地震”等具体事件仍只作为潜在触发条件说明。
5. 从隐患点沿 DEM 最陡下降方向追踪 D8 近似下泄路径，生成影响廊道。
6. 将影响廊道与居民地相交，并结合高程和人口栅格生成影响关系表。

重要边界
--------
- 输出是区域尺度的“易发性/潜在影响筛查”，不是灾害发生时间预报。
- 不要求未来降水或未来地震数据；结果表示基于历史空间规律的区域易发性。
- 11项SHAP贡献率由本项目历史灾害样本得到，并按原比例归一化为1.00。
- Mean(|SHAP|)只表示重要程度，不表示正负方向；距离类因子按“越近风险越高”、
  降水/高程/坡度按项目设定的单调方向转换，土地覆盖、冻土和坡向按历史灾害
  在各类别中的相对富集程度转换。
- D8 路径和影响廊道仍属于区域筛查模型，不等同于工程级滑坡运动、
  泥石流堆积或冰湖溃决洪水淹没模拟。

运行前请修改 CONFIG 区域中的路径、字段名、岩性评分映射和阈值。

依赖
----
pip install numpy pandas scipy geopandas rasterio shapely pyproj pyogrio
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import MergeAlg, Resampling
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds
from rasterio.transform import rowcol, xy
from rasterio.warp import calculate_default_transform, reproject
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    maximum_filter,
    minimum_filter,
)
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, box, mapping


SCRIPT_VERSION = "4.0.1-SHAP-DATA-DRIVEN-PERMAFROST-COVERAGE"


# =============================================================================
# CONFIG：运行前主要修改这里
# =============================================================================


@dataclass
class Config:
    # ---------- 输入数据 ----------
    gdb_path: Path = Path("data/NewDefault.gdb")
    dem_path: Path = Path("data/rasters/TNR_DEM_RS.tif")
    glacier_layer: str = "TNR_GLACIER_PG"
    glacial_lake_layer: str = "TNR_GLACIALLAKE_PG"
    historical_hazard_layer: str = "HAZ_LANDSLIDE_PT"
    water_layer: str = "TNR_RIVER_LN"
    road_layer: str = "EMG_ROAD_LN"
    settlement_point_layer: str | None = "BASE_SETTLEMENT_PT"
    settlement_polygon_layer: str | None = "BASE_SETTLEMENT_PG"
    settlement_name_layer: str | None = "BASE_ADMINNAME_PT"

    # 本次SHAP模型纳入评分的栅格因子。
    precipitation_dir: Path | None = Path("data/rasters/LCC_投影后")
    precipitation_pattern: str = "TNR_PRECIP_ANNUAL_RS_*_LCC.tif"
    earthquake_layer: str | None = "HAZ_EARTHQUAKE_PT"
    permafrost_path: Path | None = Path("data/rasters/TNR_PERMAFROST_RS.tif")
    landcover_path: Path | None = Path("data/rasters/LCC_投影后/TNR_LANDCOVER_RS_LCC.tif")
    population_path: Path | None = Path("data/rasters/BASE_POPULATION_RS.tif")
    slope_path: Path | None = Path("data/rasters/TNR_SLOPE_RS.tif")
    aspect_path: Path | None = Path("data/rasters/TNR_ASPECT_RS.tif")
    relief_path: Path | None = Path("data/rasters/TNR_RELIEF_RS.tif")

    # ---------- 输出 ----------
    output_dir: Path = Path("outputs/hazard_analysis")
    analysis_id: str = "QTP_HAZARD_V4_SHAP_001"

    # ---------- 统一格网 ----------
    # 青藏高原区域 Lambert 等角圆锥投影；距离单位为米。
    # 若项目已经确定其他 Lambert 参数，应替换为报告中的正式参数。
    target_crs: str = (
        "+proj=lcc +lat_1=27.5 +lat_2=37.5 +lat_0=32.5 +lon_0=89.5 "
        "+ellps=GRS80 +units=m +no_defs"
    )
    cell_size_m: float = 1000.0
    output_nodata: float = -9999.0

    # ---------- 字段 ----------
    settlement_id_field: str = "ID"
    settlement_name_field: str = "name"
    settlement_name_search_m: float = 5000.0
    earthquake_magnitude_field: str = "magnitude"

    # ---------- 历史灾情 CatBoost + SHAP 数据驱动权重 ----------
    # 原始贡献率合计100.00182%（显示舍入误差），以下按原比例归一化为1.00。
    weights: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "SHAP_DATA_DRIVEN": {
                "precipitation": 0.2678660148,
                "altitude": 0.2290484913,
                "earthquake_distance": 0.1766754745,
                "glacial_lake_distance": 0.0763798499,
                "slope": 0.0605789975,
                "permafrost": 0.0553365529,
                "glacier_distance": 0.0494396002,
                "landcover": 0.0369973066,
                "road_distance": 0.0256293835,
                "river_distance": 0.0157841927,
                "aspect": 0.0062641361,
            },
        }
    )

    # ---------- 因子构造参数 ----------
    relief_window_m: float = 5000.0
    ice_decay_m: float = 10000.0
    water_decay_m: float = 5000.0
    road_decay_m: float = 5000.0
    earthquake_decay_m: float = 30000.0
    history_density_sigma_m: float = 15000.0
    earthquake_density_sigma_m: float = 30000.0

    # 坡度评分：<=low 为0，>=high 为1，中间线性变化。
    slope_low_deg: float = 5.0
    slope_high_deg: float = 45.0

    # ---------- 隐患点提取与分类 ----------
    candidate_min_score: float = 0.55
    candidate_min_spacing_m: float = 8000.0
    max_candidate_points: int = 5000
    ice_type_min_score: float = 0.55
    seismic_type_min_score: float = 0.55
    debris_water_min_score: float = 0.55
    debris_relief_min_score: float = 0.50
    collapse_slope_deg: float = 45.0
    thaw_slope_min_deg: float = 3.0
    thaw_slope_max_deg: float = 15.0

    # ---------- 动态触发说明阈值 ----------
    precip_trigger_percentile: float = 0.80
    earthquake_trigger_density: float = 0.65
    river_trigger_score: float = 0.70
    permafrost_trigger_value: float = 0.50

    # ---------- D8 路径与影响廊道：区域筛查参数 ----------
    path_max_length_m: dict[str, float] = field(
        default_factory=lambda: {
            "冰湖溃决洪水": 50000.0,
            "冰岩崩塌": 10000.0,
            "地震次生滑坡": 8000.0,
            "泥石流": 20000.0,
            "崩塌": 3000.0,
            "滑坡": 8000.0,
            "冻融型滑坡": 8000.0,
        }
    )
    corridor_half_width_m: dict[str, float] = field(
        default_factory=lambda: {
            "冰湖溃决洪水": 1500.0,
            "冰岩崩塌": 700.0,
            "地震次生滑坡": 600.0,
            "泥石流": 800.0,
            "崩塌": 400.0,
            "滑坡": 500.0,
            "冻融型滑坡": 500.0,
        }
    )
    min_d8_drop_m: float = 0.01
    population_round_digits: int = 0
    elevation_tolerance_m: float = 20.0
    include_points_without_settlement: bool = True

    # True：11项评分因子必须在 DEM 有效区具有数据；不允许静默缺失。
    strict_factor_coverage: bool = True
    # 冻土数据通常不覆盖非冻土区及边缘。覆盖率达到该值时，范围外按
    # “无冻土贡献=0”处理，同时保留原始覆盖率记录，不重分配权重。
    permafrost_min_coverage_for_zero_fill: float = 0.90


CONFIG = Config()


FACTOR_NAMES = (
    "precipitation",
    "altitude",
    "earthquake_distance",
    "glacial_lake_distance",
    "slope",
    "permafrost",
    "glacier_distance",
    "landcover",
    "road_distance",
    "river_distance",
    "aspect",
)

FACTOR_LABELS = {
    "precipitation": "多年平均降水",
    "altitude": "高程",
    "earthquake_distance": "距地震点距离",
    "glacial_lake_distance": "距冰湖距离",
    "slope": "坡度",
    "permafrost": "冻土类型",
    "glacier_distance": "距冰川距离",
    "landcover": "土地覆盖",
    "road_distance": "距道路距离",
    "river_distance": "距河流距离",
    "aspect": "坡向",
}

SHAP_ORIGINAL_PERCENT = {
    "precipitation": 26.787089,
    "altitude": 22.905266,
    "earthquake_distance": 17.667869,
    "glacial_lake_distance": 7.638124,
    "slope": 6.058010,
    "permafrost": 5.533756,
    "glacier_distance": 4.944050,
    "landcover": 3.699798,
    "road_distance": 2.562985,
    "river_distance": 1.578448,
    "aspect": 0.626425,
}


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "hazard_analysis_v4.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )


def serializable_config(cfg: Config) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value

    return convert(asdict(cfg))


def validate_weights(cfg: Config) -> None:
    expected = set(FACTOR_NAMES)
    for scenario, weights in cfg.weights.items():
        keys = set(weights)
        if keys != expected:
            raise ValueError(
                f"{scenario} 权重因子不完整；缺少={sorted(expected - keys)}，"
                f"多余={sorted(keys - expected)}"
            )
        total = sum(weights.values())
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(f"{scenario} 权重和为 {total:.12f}，必须等于 1.00")
        if any(v < 0 for v in weights.values()):
            raise ValueError(f"{scenario} 存在负权重")


def required_paths(cfg: Config) -> dict[str, Path]:
    paths: dict[str, Path] = {
        "File GDB": cfg.gdb_path,
        "DEM": cfg.dem_path,
    }
    for name, path in {
        "年度降水目录": cfg.precipitation_dir,
        "冻土": cfg.permafrost_path,
        "土地覆盖": cfg.landcover_path,
    }.items():
        if path is None:
            raise ValueError(f"SHAP评分必需输入未配置：{name}")
        paths[name] = path
    return paths


def optional_paths(cfg: Config) -> dict[str, Path | None]:
    return {
        "人口": cfg.population_path,
        "坡度": cfg.slope_path,
        "坡向": cfg.aspect_path,
        "地形起伏度": cfg.relief_path,
    }


def required_layers(cfg: Config) -> dict[str, str]:
    if cfg.earthquake_layer is None:
        raise ValueError("SHAP评分必需图层未配置：历史地震")
    return {
        "冰川": cfg.glacier_layer,
        "冰湖": cfg.glacial_lake_layer,
        "历史滑坡泥石流事件": cfg.historical_hazard_layer,
        "水系/沟道": cfg.water_layer,
        "道路": cfg.road_layer,
        "历史地震": cfg.earthquake_layer,
    }


def optional_layers(cfg: Config) -> dict[str, str | None]:
    return {
        "居民地点": cfg.settlement_point_layer,
        "居民地面": cfg.settlement_polygon_layer,
        "行政居民地名称": cfg.settlement_name_layer,
    }


def list_gdb_layers(path: Path) -> set[str]:
    try:
        import pyogrio

        return {str(row[0]) for row in pyogrio.list_layers(path)}
    except Exception:
        try:
            import fiona

            return set(fiona.listlayers(path))
        except Exception as exc:
            raise RuntimeError(
                "无法读取 File GDB 图层目录；请确认已安装 pyogrio 或 fiona，"
                f"并且 GDB 可读：{path}；原始错误={exc}"
            ) from exc


def validate_paths(cfg: Config) -> None:
    missing_required = [f"{name}: {path}" for name, path in required_paths(cfg).items() if not path.exists()]
    if missing_required:
        raise FileNotFoundError(
            "以下必需输入不存在，请修改 CONFIG：\n  - " + "\n  - ".join(missing_required)
        )
    for name, path in optional_paths(cfg).items():
        if path is not None and not path.exists():
            logging.warning("可选输入不存在，将跳过 %s：%s", name, path)
    if not cfg.gdb_path.exists():
        return
    layers = list_gdb_layers(cfg.gdb_path)
    missing_layers = [
        f"{name}: {layer}"
        for name, layer in required_layers(cfg).items()
        if layer not in layers
    ]
    if missing_layers:
        raise FileNotFoundError(
            "以下必需 GDB 图层不存在：\n  - " + "\n  - ".join(missing_layers)
        )
    for name, layer in optional_layers(cfg).items():
        if layer is not None and layer not in layers:
            logging.warning("可选 GDB 图层不存在，将跳过 %s：%s", name, layer)


def robust_scale(array: np.ndarray, valid: np.ndarray, low_q: float = 5, high_q: float = 95) -> np.ndarray:
    out = np.full(array.shape, np.nan, dtype="float32")
    values = array[valid & np.isfinite(array)]
    if values.size == 0:
        return out
    lo, hi = np.nanpercentile(values, [low_q, high_q])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out[valid] = 0.5
        return out
    out[valid] = np.clip((array[valid] - lo) / (hi - lo), 0.0, 1.0)
    return out


def linear_score(array: np.ndarray, low: float, high: float, valid: np.ndarray) -> np.ndarray:
    if high <= low:
        raise ValueError("线性评分上限必须大于下限")
    out = np.full(array.shape, np.nan, dtype="float32")
    out[valid] = np.clip((array[valid] - low) / (high - low), 0.0, 1.0)
    return out


def read_dem_to_analysis_grid(cfg: Config) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(cfg.dem_path) as src:
        if src.crs is None:
            raise ValueError("DEM 缺少 CRS，不能进行距离和面积分析")
        transform, width, height = calculate_default_transform(
            src.crs,
            cfg.target_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=cfg.cell_size_m,
        )
        dem = np.full((height, width), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dem,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=cfg.target_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype="float32",
            crs=cfg.target_crs,
            transform=transform,
            nodata=cfg.output_nodata,
            compress="deflate",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )
    return dem, profile


def analysis_bounds(profile: dict[str, Any]) -> tuple[float, float, float, float]:
    t = profile["transform"]
    width, height = profile["width"], profile["height"]
    left, top = t * (0, 0)
    right, bottom = t * (width, height)
    return min(left, right), min(bottom, top), max(left, right), max(bottom, top)


def read_vector(
    path: Path,
    profile: dict[str, Any],
    layer_name: str,
    source_layer: str | None = None,
) -> gpd.GeoDataFrame:
    kwargs = {"layer": source_layer} if source_layer else {}
    gdf = gpd.read_file(path, **kwargs)
    if gdf.empty:
        raise ValueError(f"{layer_name} 图层为空：{path}")
    if gdf.crs is None:
        raise ValueError(f"{layer_name} 缺少 CRS：{path}")
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    gdf = gdf.to_crs(profile["crs"])
    bounds_geom = box(*analysis_bounds(profile))
    gdf = gdf[gdf.geometry.intersects(bounds_geom)].copy()
    if gdf.empty:
        raise ValueError(f"{layer_name} 与 DEM 分析范围没有交集：{path}")
    return gdf


def rasterize_binary(gdf: gpd.GeoDataFrame, profile: dict[str, Any]) -> np.ndarray:
    return rasterize(
        ((geom, 1) for geom in gdf.geometry),
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        dtype="uint8",
        all_touched=True,
    )


def distance_score_from_vector(
    gdf: gpd.GeoDataFrame,
    profile: dict[str, Any],
    decay_m: float,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = rasterize_binary(gdf, profile).astype(bool)
    if not mask.any():
        raise ValueError("矢量要素栅格化后没有有效像元；请检查范围、CRS和格网分辨率")
    distance_m = distance_transform_edt(
        ~mask,
        sampling=(abs(profile["transform"].e), abs(profile["transform"].a)),
    ).astype("float32")
    score = np.exp(-distance_m / max(decay_m, 1.0)).astype("float32")
    score[~valid] = np.nan
    distance_m[~valid] = np.nan
    return score, distance_m


def ice_score_with_size(
    cfg: Config,
    profile: dict[str, Any],
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    glacier = read_vector(cfg.gdb_path, profile, "冰川", cfg.glacier_layer)
    lake = read_vector(cfg.gdb_path, profile, "冰湖", cfg.glacial_lake_layer)

    glacier_area = glacier.geometry.area.to_numpy() / 1_000_000.0
    lake_area = lake.geometry.area.to_numpy() / 1_000_000.0
    all_areas = np.concatenate([glacier_area, lake_area])
    lo, hi = np.nanpercentile(np.log1p(all_areas), [5, 95])

    def area_score(values: np.ndarray) -> np.ndarray:
        values = np.log1p(values)
        if hi <= lo:
            return np.full(values.shape, 0.5)
        return np.clip((values - lo) / (hi - lo), 0.0, 1.0)

    glacier = glacier.assign(_size=area_score(glacier_area))
    lake = lake.assign(_size=area_score(lake_area))
    combined = pd.concat([glacier, lake], ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=profile["crs"])

    size_surface = rasterize(
        ((geom, float(size)) for geom, size in zip(combined.geometry, combined["_size"])),
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0.0,
        dtype="float32",
        all_touched=True,
        merge_alg=MergeAlg.replace,
    )
    mask = size_surface > 0
    # 极小面可能获得0分，仍应作为冰川/冰湖存在。
    mask |= rasterize_binary(combined, profile).astype(bool)
    dist, indices = distance_transform_edt(
        ~mask,
        sampling=(abs(profile["transform"].e), abs(profile["transform"].a)),
        return_indices=True,
    )
    nearest_size = size_surface[indices[0], indices[1]]
    score = np.exp(-dist / max(cfg.ice_decay_m, 1.0)) * (0.5 + 0.5 * nearest_size)
    score = score.astype("float32")

    _, glacier_dist = distance_score_from_vector(glacier, profile, cfg.ice_decay_m, valid)
    _, lake_dist = distance_score_from_vector(lake, profile, cfg.ice_decay_m, valid)
    score[~valid] = np.nan
    return score, glacier_dist, lake_dist


def point_density_score(
    gdf: gpd.GeoDataFrame,
    profile: dict[str, Any],
    sigma_m: float,
    valid: np.ndarray,
    weight_field: str | None = None,
) -> np.ndarray:
    if weight_field and weight_field in gdf.columns:
        weights = pd.to_numeric(gdf[weight_field], errors="coerce").fillna(1.0).clip(lower=0.0)
    else:
        weights = pd.Series(np.ones(len(gdf)), index=gdf.index)
    counts = rasterize(
        ((geom, float(w)) for geom, w in zip(gdf.geometry, weights)),
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0.0,
        dtype="float32",
        all_touched=True,
        merge_alg=MergeAlg.add,
    )
    sigma_cells = max(sigma_m / max(abs(profile["transform"].a), 1.0), 0.5)
    density = gaussian_filter(counts, sigma=sigma_cells, mode="constant")
    return robust_scale(density, valid, 1, 99)


def categorical_concentration_score(
    categories: np.ndarray,
    history_points: gpd.GeoDataFrame,
    profile: dict[str, Any],
    valid: np.ndarray,
) -> np.ndarray:
    """按历史灾害在各类别中的单位面积富集程度生成0—1风险分值。"""
    point_counts = rasterize(
        ((geom, 1.0) for geom in history_points.geometry),
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0.0,
        dtype="float32",
        all_touched=True,
        merge_alg=MergeAlg.add,
    )
    usable = valid & np.isfinite(categories)
    out = np.full(categories.shape, np.nan, dtype="float32")
    if not usable.any():
        return out

    # 栅格类别应为整数；浮点存储造成的极小误差在此统一舍入。
    category_codes = np.rint(categories[usable]).astype("int64")
    unique_codes = np.unique(category_codes)
    rates: dict[int, float] = {}
    total_events = float(point_counts[usable].sum())
    total_cells = int(usable.sum())
    global_rate = (total_events + 1.0) / (total_cells + 1.0)
    rounded_all = np.zeros(categories.shape, dtype="int64")
    rounded_all[usable] = category_codes
    for code in unique_codes:
        mask = usable & (rounded_all == code)
        cells = int(mask.sum())
        events = float(point_counts[mask].sum())
        # Jeffreys型平滑，避免无事件类别直接得到无限或零比值。
        rate = (events + 0.5) / (cells + 1.0)
        rates[int(code)] = math.log1p(rate / max(global_rate, 1e-12))

    values = np.asarray(list(rates.values()), dtype="float64")
    lo, hi = float(values.min()), float(values.max())
    for code, value in rates.items():
        score = 0.5 if hi <= lo else (value - lo) / (hi - lo)
        out[usable & (rounded_all == code)] = float(score)
    return out


def derive_terrain_factors(
    dem: np.ndarray,
    profile: dict[str, Any],
    cfg: Config,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    valid = np.isfinite(dem)
    filled = dem.copy()
    median = float(np.nanmedian(dem[valid]))
    filled[~valid] = median
    cell_y = abs(profile["transform"].e)
    cell_x = abs(profile["transform"].a)
    dz_dy, dz_dx = np.gradient(filled, cell_y, cell_x)
    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype("float32")

    # 坡向：0北、90东、180南、270西。
    aspect_deg = (np.degrees(np.arctan2(dz_dx, -dz_dy)) + 360.0) % 360.0
    slope = linear_score(slope_deg, cfg.slope_low_deg, cfg.slope_high_deg, valid)

    window_cells = max(3, int(round(cfg.relief_window_m / max(cell_x, cell_y))))
    if window_cells % 2 == 0:
        window_cells += 1
    local_max = maximum_filter(filled, size=window_cells, mode="nearest")
    local_min = minimum_filter(filled, size=window_cells, mode="nearest")
    relief_m = (local_max - local_min).astype("float32")
    relief = robust_scale(relief_m, valid, 5, 95)

    raw = {
        "slope_deg": slope_deg,
        "aspect_deg": aspect_deg.astype("float32"),
        "relief_m": relief_m,
    }
    common = {"slope": slope, "relief": relief}
    for array in raw.values():
        array[~valid] = np.nan
    return common, raw


def load_or_derive_terrain_factors(
    dem: np.ndarray,
    profile: dict[str, Any],
    cfg: Config,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """优先使用已入库坡度、坡向、起伏度；缺少时再由 DEM 推导。"""
    common, raw = derive_terrain_factors(dem, profile, cfg)
    valid = np.isfinite(dem)
    slope_raw = align_optional_raster(cfg.slope_path, profile, Resampling.bilinear)
    relief_raw = align_optional_raster(cfg.relief_path, profile, Resampling.bilinear)
    aspect_raw = align_optional_raster(cfg.aspect_path, profile, Resampling.nearest)
    if slope_raw is not None:
        slope_valid = valid & np.isfinite(slope_raw)
        common["slope"] = linear_score(
            slope_raw, cfg.slope_low_deg, cfg.slope_high_deg, slope_valid
        )
        raw["slope_deg"] = slope_raw.astype("float32")
        logging.info("坡度因子使用正式坡度栅格")
    if relief_raw is not None:
        relief_valid = valid & np.isfinite(relief_raw)
        common["relief"] = robust_scale(relief_raw, relief_valid, 5, 95)
        raw["relief_m"] = relief_raw.astype("float32")
        logging.info("地形起伏度因子使用正式起伏度栅格")
    if aspect_raw is not None:
        raw["aspect_deg"] = np.mod(aspect_raw, 360.0).astype("float32")
        raw["aspect_deg"][~valid] = np.nan
        logging.info("坡向因子使用正式坡向栅格")
    return common, raw


def aspect_preference_score(
    aspect_deg: np.ndarray,
    preferred_deg: float,
    valid: np.ndarray,
) -> np.ndarray:
    # 余弦方向相似度：偏好方向=1，反方向=0。
    delta = np.radians((aspect_deg - preferred_deg + 180.0) % 360.0 - 180.0)
    out = ((1.0 + np.cos(delta)) / 2.0).astype("float32")
    out[~valid] = np.nan
    return out


def align_optional_raster(
    path: Path | None,
    profile: dict[str, Any],
    resampling: Resampling,
) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    out = np.full((profile["height"], profile["width"]), np.nan, dtype="float32")
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"栅格缺少 CRS：{path}")
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return out


def align_precipitation_mean(
    cfg: Config,
    profile: dict[str, Any],
) -> tuple[np.ndarray | None, list[str]]:
    if cfg.precipitation_dir is None or not cfg.precipitation_dir.exists():
        return None, []
    paths = sorted(cfg.precipitation_dir.glob(cfg.precipitation_pattern))
    if not paths:
        return None, []
    total = np.zeros((profile["height"], profile["width"]), dtype="float64")
    count = np.zeros((profile["height"], profile["width"]), dtype="uint16")
    for path in paths:
        aligned = align_optional_raster(path, profile, Resampling.bilinear)
        if aligned is None:
            continue
        good = np.isfinite(aligned)
        total[good] += aligned[good]
        count[good] += 1
    mean = np.full(total.shape, np.nan, dtype="float32")
    good = count > 0
    mean[good] = (total[good] / count[good]).astype("float32")
    return mean, [str(path) for path in paths]


def build_factors(
    cfg: Config,
    dem: np.ndarray,
    profile: dict[str, Any],
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    valid = np.isfinite(dem)
    common, raw = load_or_derive_terrain_factors(dem, profile, cfg)

    history_points = read_vector(
        cfg.gdb_path,
        profile,
        "历史滑坡泥石流事件",
        cfg.historical_hazard_layer,
    )

    glacier = read_vector(cfg.gdb_path, profile, "冰川", cfg.glacier_layer)
    glacial_lake = read_vector(cfg.gdb_path, profile, "冰湖", cfg.glacial_lake_layer)
    glacier_distance, glacier_distance_m = distance_score_from_vector(
        glacier, profile, cfg.ice_decay_m, valid
    )
    glacial_lake_distance, lake_distance_m = distance_score_from_vector(
        glacial_lake, profile, cfg.ice_decay_m, valid
    )

    water_gdf = read_vector(cfg.gdb_path, profile, "水系/沟道", cfg.water_layer)
    river_distance, water_distance_m = distance_score_from_vector(
        water_gdf, profile, cfg.water_decay_m, valid
    )
    road_gdf = read_vector(cfg.gdb_path, profile, "道路", cfg.road_layer)
    road_distance, road_distance_m = distance_score_from_vector(
        road_gdf, profile, cfg.road_decay_m, valid
    )

    if cfg.earthquake_layer is None:
        raise ValueError("SHAP评分要求配置历史地震图层")
    earthquakes = read_vector(
        cfg.gdb_path, profile, "历史地震", cfg.earthquake_layer
    )
    earthquake_distance, earthquake_distance_m = distance_score_from_vector(
        earthquakes, profile, cfg.earthquake_decay_m, valid
    )
    earthquake_density = point_density_score(
        earthquakes,
        profile,
        cfg.earthquake_density_sigma_m,
        valid,
        cfg.earthquake_magnitude_field,
    )

    precipitation_raw, precipitation_files = align_precipitation_mean(cfg, profile)
    if precipitation_raw is None:
        raise FileNotFoundError("未找到年度降水LCC栅格，不能计算SHAP数据驱动评分")
    precipitation_score = robust_scale(
        precipitation_raw, valid & np.isfinite(precipitation_raw), 5, 95
    )

    permafrost_raw = align_optional_raster(
        cfg.permafrost_path, profile, Resampling.nearest
    )
    landcover_raw = align_optional_raster(
        cfg.landcover_path, profile, Resampling.nearest
    )
    if permafrost_raw is None or landcover_raw is None:
        raise FileNotFoundError("冻土或土地覆盖栅格缺失，不能计算SHAP数据驱动评分")

    permafrost_score = categorical_concentration_score(
        permafrost_raw, history_points, profile, valid
    )
    landcover_score = categorical_concentration_score(
        landcover_raw, history_points, profile, valid
    )
    # 坡向分成8个45°方向类别，再按历史灾害富集程度赋分。
    aspect_bins = np.floor(((raw["aspect_deg"] + 22.5) % 360.0) / 45.0)
    aspect_score = categorical_concentration_score(
        aspect_bins, history_points, profile, valid
    )
    altitude_score = robust_scale(dem, valid, 5, 95)

    factors = {
        "precipitation": precipitation_score,
        "altitude": altitude_score,
        "earthquake_distance": earthquake_distance,
        "glacial_lake_distance": glacial_lake_distance,
        "slope": common["slope"],
        "permafrost": permafrost_score,
        "glacier_distance": glacier_distance,
        "landcover": landcover_score,
        "road_distance": road_distance,
        "river_distance": river_distance,
        "aspect": aspect_score,
    }
    factors_by_scenario = {"SHAP_DATA_DRIVEN": factors}

    raw.update(
        {
            "glacier_distance_m": glacier_distance_m,
            "lake_distance_m": lake_distance_m,
            "water_distance_m": water_distance_m,
            "road_distance_m": road_distance_m,
            "earthquake_distance_m": earthquake_distance_m,
            "precipitation_raw": precipitation_raw,
            "permafrost_raw": permafrost_raw,
            "landcover_raw": landcover_raw,
        }
    )

    triggers = {
        "precipitation_raw": precipitation_raw,
        "precipitation_score": precipitation_score,
        "permafrost_score": permafrost_score,
        "earthquake_density": earthquake_density,
        "precipitation_files": precipitation_files,
    }
    return factors_by_scenario, raw, triggers


def validate_factor_coverage(
    cfg: Config,
    factors_by_scenario: dict[str, dict[str, np.ndarray]],
    dem_valid: np.ndarray,
) -> dict[str, float]:
    factors = factors_by_scenario["SHAP_DATA_DRIVEN"]
    coverage = {
        name: float(np.count_nonzero(np.isfinite(array) & dem_valid) / np.count_nonzero(dem_valid))
        for name, array in factors.items()
    }
    if cfg.strict_factor_coverage:
        permafrost_coverage = coverage.get("permafrost", 0.0)
        if (
            cfg.permafrost_min_coverage_for_zero_fill
            <= permafrost_coverage
            < 0.95
        ):
            permafrost = factors["permafrost"]
            fill_mask = dem_valid & ~np.isfinite(permafrost)
            permafrost[fill_mask] = 0.0
            logging.warning(
                "冻土源数据覆盖率为%.1f%%；其余%.1f%%按无冻土贡献0处理，"
                "原始覆盖率写入元数据，11项权重保持不变。",
                permafrost_coverage * 100.0,
                (1.0 - permafrost_coverage) * 100.0,
            )
        too_low = {
            name: ratio
            for name, ratio in coverage.items()
            if ratio < 0.95
            and not (
                name == "permafrost"
                and ratio >= cfg.permafrost_min_coverage_for_zero_fill
            )
        }
        if too_low:
            detail = ", ".join(f"{k}={v:.1%}" for k, v in too_low.items())
            raise ValueError(
                "评分因子在 DEM 有效区的覆盖率不足要求，未执行权重重分配：" + detail
            )
    return coverage


def weighted_score(
    factors: dict[str, np.ndarray],
    weights: dict[str, float],
    valid: np.ndarray,
) -> np.ndarray:
    stack = np.stack([factors[name] for name in FACTOR_NAMES])
    complete = valid & np.all(np.isfinite(stack), axis=0)
    score = np.full(valid.shape, np.nan, dtype="float32")
    weighted = np.zeros(valid.shape, dtype="float32")
    for name, weight in weights.items():
        weighted += np.nan_to_num(factors[name], nan=0.0) * weight
    score[complete] = np.clip(weighted[complete], 0.0, 1.0)
    return score


def score_level(score_100: float) -> str:
    if score_100 >= 85:
        return "极高"
    if score_100 >= 70:
        return "高"
    if score_100 >= 50:
        return "中"
    if score_100 >= 40:
        return "较低"
    return "低"


def safe_value(array: np.ndarray | None, r: int, c: int) -> float | None:
    if array is None:
        return None
    value = float(array[r, c])
    return value if math.isfinite(value) else None


def classify_hazard(
    r: int,
    c: int,
    scores: dict[str, np.ndarray],
    factors: dict[str, np.ndarray],
    raw: dict[str, np.ndarray],
    triggers: dict[str, Any],
    cfg: Config,
) -> tuple[str, str, float, str, str]:
    total_score = float(scores["SHAP_DATA_DRIVEN"][r, c])
    slope_deg = float(raw["slope_deg"][r, c])
    slope_score = float(factors["slope"][r, c])
    lake_score = float(factors["glacial_lake_distance"][r, c])
    glacier_score = float(factors["glacier_distance"][r, c])
    river_score = float(factors["river_distance"][r, c])
    precip_score = float(factors["precipitation"][r, c])
    eq_distance_score = float(factors["earthquake_distance"][r, c])
    permafrost = float(factors["permafrost"][r, c])
    lake_distance = float(raw["lake_distance_m"][r, c])
    glacier_distance = float(raw["glacier_distance_m"][r, c])

    # 灾种由机理相关因子的组合强度判识；综合易发性仍统一采用SHAP总分。
    candidates: list[tuple[str, float, str]] = []
    if max(lake_score, glacier_score) >= cfg.ice_type_min_score:
        if lake_distance <= glacier_distance:
            candidates.append(
                (
                    "冰湖溃决洪水",
                    float(np.mean([lake_score, precip_score, slope_score])),
                    "邻近冰湖，且降水与坡度条件共同支持下游影响",
                )
            )
        else:
            candidates.append(
                (
                    "冰岩崩塌",
                    float(np.mean([glacier_score, slope_score, precip_score])),
                    "邻近冰川，且坡度与降水条件共同支持冰岩失稳",
                )
            )

    if (
        eq_distance_score >= cfg.seismic_type_min_score
        and slope_deg >= cfg.slope_low_deg
    ):
        candidates.append(
            (
                "地震次生滑坡",
                float(np.mean([eq_distance_score, slope_score])),
                "邻近历史地震且斜坡条件较明显",
            )
        )

    if (
        permafrost >= cfg.permafrost_trigger_value
        and cfg.thaw_slope_min_deg <= slope_deg <= cfg.thaw_slope_max_deg
    ):
        candidates.append(
            (
                "冻融型滑坡",
                float(np.mean([permafrost, slope_score, precip_score])),
                "冻土类别富集度较高且处于缓坡或中缓坡",
            )
        )
    if (
        river_score >= cfg.debris_water_min_score
        and precip_score >= cfg.precip_trigger_percentile
        and slope_deg >= cfg.slope_low_deg
    ):
        candidates.append(
            (
                "泥石流",
                float(np.mean([river_score, precip_score, slope_score])),
                "邻近沟道，且降水与坡度条件较强",
            )
        )
    if slope_deg >= cfg.collapse_slope_deg:
        candidates.append(("崩塌", slope_score, "陡坡条件突出"))
    if total_score >= cfg.candidate_min_score or not candidates:
        candidates.append(("滑坡", total_score, "11项历史灾情因子综合易发性较高"))

    candidates.sort(key=lambda item: item[1], reverse=True)
    primary = candidates[0]
    secondary_names = list(dict.fromkeys(item[0] for item in candidates[1:]))
    return (
        primary[0],
        "SHAP_DATA_DRIVEN",
        total_score,
        primary[2],
        "；".join(secondary_names),
    )


def build_trigger_description(
    hazard_type: str,
    r: int,
    c: int,
    factors: dict[str, np.ndarray],
    raw: dict[str, np.ndarray],
    triggers: dict[str, Any],
    cfg: Config,
) -> tuple[str, str]:
    names: list[str] = []
    evidence: list[str] = []
    precip_score = safe_value(triggers["precipitation_score"], r, c)
    precip_raw = safe_value(triggers["precipitation_raw"], r, c)
    eq_density = safe_value(triggers["earthquake_density"], r, c)
    permafrost = safe_value(triggers["permafrost_score"], r, c)

    if precip_score is not None and precip_score >= cfg.precip_trigger_percentile:
        names.append("强降雨或持续降水（潜在触发）")
        raw_text = f"，原始值={precip_raw:.2f}" if precip_raw is not None else ""
        evidence.append(f"多年降水标准化值={precip_score:.2f}{raw_text}")
    if eq_density is not None and eq_density >= cfg.earthquake_trigger_density:
        names.append("地震动（潜在触发）")
        evidence.append(f"历史地震密度标准化值={eq_density:.2f}")
    ice_score = max(
        float(factors["glacial_lake_distance"][r, c]),
        float(factors["glacier_distance"][r, c]),
    )
    if hazard_type in {"冰湖溃决洪水", "冰岩崩塌"} or ice_score >= 0.60:
        names.append("升温融冰或冰雪消融（潜在触发）")
        evidence.append(f"冰川/冰湖邻近因子={ice_score:.2f}")
    if permafrost is not None and permafrost >= cfg.permafrost_trigger_value:
        names.append("升温与冻融循环（潜在触发）")
        evidence.append(f"冻土因子={permafrost:.2f}")
    if factors["river_distance"][r, c] >= cfg.river_trigger_score:
        names.append("河流冲刷或沟谷汇流（潜在触发）")
        evidence.append(f"距水系={raw['water_distance_m'][r, c]:.0f}m")
    if not names:
        names.append("未识别明确动态触发条件")
        evidence.append("当前可选触发数据未达到设定阈值或数据缺失")
    # 保持顺序并去重。
    names = list(dict.fromkeys(names))
    evidence = list(dict.fromkeys(evidence))
    return "；".join(names), "；".join(evidence)


def extract_candidate_points(
    cfg: Config,
    dem: np.ndarray,
    profile: dict[str, Any],
    factors_by_scenario: dict[str, dict[str, np.ndarray]],
    raw: dict[str, np.ndarray],
    triggers: dict[str, Any],
    scores: dict[str, np.ndarray],
) -> gpd.GeoDataFrame:
    valid = np.isfinite(dem)
    combined = np.nanmax(np.stack(list(scores.values())), axis=0)
    spacing_cells = max(1, int(round(cfg.candidate_min_spacing_m / cfg.cell_size_m)))
    filter_size = 2 * spacing_cells + 1
    work = np.where(np.isfinite(combined), combined, -np.inf)
    local_max = maximum_filter(work, size=filter_size, mode="constant", cval=-np.inf)
    candidate_mask = valid & (combined >= cfg.candidate_min_score) & np.isclose(work, local_max)
    rows, cols = np.where(candidate_mask)
    if rows.size == 0:
        raise RuntimeError("没有提取到隐患点，请检查因子范围或降低 candidate_min_score")
    order = np.argsort(combined[rows, cols])[::-1][: cfg.max_candidate_points]
    rows, cols = rows[order], cols[order]
    model_factors = factors_by_scenario["SHAP_DATA_DRIVEN"]

    records: list[dict[str, Any]] = []
    for index, (r, c) in enumerate(zip(rows, cols), start=1):
        hazard_type, scenario, selected_score, type_reason, secondary_types = classify_hazard(
            int(r), int(c), scores, model_factors, raw, triggers, cfg
        )
        x, y = xy(profile["transform"], int(r), int(c), offset="center")
        possible_triggers, trigger_evidence = build_trigger_description(
            hazard_type, int(r), int(c), model_factors, raw, triggers, cfg
        )
        record: dict[str, Any] = {
            "analysis_id": cfg.analysis_id,
            "hazard_point_id": f"HZ{index:06d}",
            "hazard_type": hazard_type,
            "secondary_hazard_types": secondary_types,
            "weight_scenario": scenario,
            "score": round(selected_score * 100.0, 2),
            "hazard_level": score_level(selected_score * 100.0),
            "type_reason": type_reason,
            "possible_triggers": possible_triggers,
            "trigger_evidence": trigger_evidence,
            "elevation_m": round(float(dem[r, c]), 2),
            "slope_deg": round(float(raw["slope_deg"][r, c]), 2),
            "aspect_deg": round(float(raw["aspect_deg"][r, c]), 2),
            "relief_m": round(float(raw["relief_m"][r, c]), 2),
            "score_SHAP": round(
                float(scores["SHAP_DATA_DRIVEN"][r, c]) * 100.0, 2
            ),
            "model_version": SCRIPT_VERSION,
            "analysis_time": datetime.now().isoformat(timespec="seconds"),
            "grid_row": int(r),
            "grid_col": int(c),
            "geometry": Point(float(x), float(y)),
        }
        for name in FACTOR_NAMES:
            record[f"f_{name}"] = round(float(model_factors[name][r, c]), 4)
        records.append(record)

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=profile["crs"])
    geographic = gdf.to_crs("EPSG:4490")
    gdf["point_lon"] = geographic.geometry.x.round(7)
    gdf["point_lat"] = geographic.geometry.y.round(7)
    return gdf


def load_settlements(cfg: Config, profile: dict[str, Any]) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    sources = [
        (cfg.settlement_point_layer, "PT"),
        (cfg.settlement_polygon_layer, "PG"),
    ]
    available = list_gdb_layers(cfg.gdb_path)
    for layer, source_type in sources:
        if layer is None or layer not in available:
            continue
        gdf = read_vector(cfg.gdb_path, profile, f"居民地{source_type}", layer)
        id_values = (
            gdf[cfg.settlement_id_field].astype(str)
            if cfg.settlement_id_field in gdf.columns
            else pd.Series([f"{source_type}_{i + 1:06d}" for i in range(len(gdf))], index=gdf.index)
        )
        name_values = (
            gdf[cfg.settlement_name_field].astype(str)
            if cfg.settlement_name_field in gdf.columns
            else pd.Series(["未命名"] * len(gdf), index=gdf.index)
        )
        point_geometry = gdf.geometry.representative_point()
        frame = gpd.GeoDataFrame(
            {
                "settlement_id": id_values.to_numpy(),
                "settlement_name": name_values.to_numpy(),
                "source_geometry": source_type,
            },
            geometry=point_geometry,
            crs=gdf.crs,
        )
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("居民地点和居民地面均不可用，不能执行居民地影响分析")
    combined = pd.concat(frames, ignore_index=True)
    settlements = gpd.GeoDataFrame(combined, geometry="geometry", crs=profile["crs"])
    # 点、面重复时按编号优先保留点状表达。
    settlements["_priority"] = settlements["source_geometry"].map({"PT": 0, "PG": 1})
    settlements = (
        settlements.sort_values(["settlement_id", "_priority"])
        .drop_duplicates("settlement_id", keep="first")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )
    # 居民地图层缺少名称时，以 BASE_ADMINNAME_PT 最近邻名称补充，
    # 仅作为输出显示名称，不改变居民地几何或编号。
    if cfg.settlement_name_layer is not None and cfg.settlement_name_layer in available:
        name_points = read_vector(
            cfg.gdb_path,
            profile,
            "行政居民地名称",
            cfg.settlement_name_layer,
        )
        if cfg.settlement_name_field in name_points.columns and not name_points.empty:
            valid_names = name_points[cfg.settlement_name_field].notna()
            name_points = name_points.loc[valid_names].copy()
            if not name_points.empty:
                tree = cKDTree(
                    np.column_stack([name_points.geometry.x, name_points.geometry.y])
                )
                distances, indexes = tree.query(
                    np.column_stack([settlements.geometry.x, settlements.geometry.y]), k=1
                )
                missing_name = settlements["settlement_name"].isin(["未命名", "", "None", "nan"])
                fill = missing_name.to_numpy() & (distances <= cfg.settlement_name_search_m)
                nearest_names = name_points.iloc[indexes][cfg.settlement_name_field].astype(str).to_numpy()
                settlements.loc[fill, "settlement_name"] = nearest_names[fill]
                settlements["name_match_distance_m"] = np.where(fill, distances, np.nan)
            else:
                settlements["name_match_distance_m"] = np.nan
        else:
            settlements["name_match_distance_m"] = np.nan
    else:
        settlements["name_match_distance_m"] = np.nan
    return settlements


def sample_array_at_points(
    points: Iterable[Point],
    array: np.ndarray,
    profile: dict[str, Any],
) -> np.ndarray:
    values: list[float] = []
    height, width = array.shape
    for point in points:
        r, c = rowcol(profile["transform"], point.x, point.y)
        if 0 <= r < height and 0 <= c < width and np.isfinite(array[r, c]):
            values.append(float(array[r, c]))
        else:
            values.append(np.nan)
    return np.asarray(values, dtype="float64")


def impact_level(impact_score: float) -> str:
    if impact_score >= 0.80:
        return "极高"
    if impact_score >= 0.60:
        return "高"
    if impact_score >= 0.35:
        return "中"
    return "低"


def trace_d8_downslope(
    start_row: int,
    start_col: int,
    dem: np.ndarray,
    profile: dict[str, Any],
    max_length_m: float,
    min_drop_m: float,
) -> tuple[LineString, float, float]:
    """从起点逐格选择单位距离下降最快的邻格，返回路径、长度和总落差。"""
    height, width = dem.shape
    cell_x = abs(float(profile["transform"].a))
    cell_y = abs(float(profile["transform"].e))
    neighbours = [
        (-1, -1, math.hypot(cell_x, cell_y)),
        (-1, 0, cell_y),
        (-1, 1, math.hypot(cell_x, cell_y)),
        (0, -1, cell_x),
        (0, 1, cell_x),
        (1, -1, math.hypot(cell_x, cell_y)),
        (1, 0, cell_y),
        (1, 1, math.hypot(cell_x, cell_y)),
    ]
    row, col = int(start_row), int(start_col)
    start_z = float(dem[row, col])
    cells = [(row, col)]
    visited = {(row, col)}
    path_length = 0.0
    while path_length < max_length_m:
        current_z = float(dem[row, col])
        best: tuple[float, int, int, float] | None = None
        for dr, dc, step in neighbours:
            rr, cc = row + dr, col + dc
            if not (0 <= rr < height and 0 <= cc < width):
                continue
            if (rr, cc) in visited or not np.isfinite(dem[rr, cc]):
                continue
            drop = current_z - float(dem[rr, cc])
            if drop < min_drop_m:
                continue
            gradient = drop / step
            if best is None or gradient > best[0]:
                best = (gradient, rr, cc, step)
        if best is None:
            break
        _, row, col, step = best
        if path_length + step > max_length_m:
            break
        path_length += step
        cells.append((row, col))
        visited.add((row, col))

    coordinates = [
        tuple(map(float, xy(profile["transform"], r, c, offset="center")))
        for r, c in cells
    ]
    if len(coordinates) == 1:
        x, y = coordinates[0]
        coordinates.append((x + min(cell_x, 1.0), y))
    end_z = float(dem[cells[-1][0], cells[-1][1]])
    return LineString(coordinates), path_length, max(0.0, start_z - end_z)


def build_impact_geometries(
    cfg: Config,
    hazard_points: gpd.GeoDataFrame,
    dem: np.ndarray,
    profile: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    path_records: list[dict[str, Any]] = []
    zone_records: list[dict[str, Any]] = []
    for _, hazard in hazard_points.iterrows():
        hazard_type = str(hazard["hazard_type"])
        max_length = cfg.path_max_length_m.get(hazard_type, 8000.0)
        half_width = cfg.corridor_half_width_m.get(hazard_type, 500.0)
        line, length_m, elevation_drop_m = trace_d8_downslope(
            int(hazard["grid_row"]),
            int(hazard["grid_col"]),
            dem,
            profile,
            max_length,
            cfg.min_d8_drop_m,
        )
        zone = line.buffer(half_width, cap_style=2, join_style=2)
        common = {
            "analysis_id": cfg.analysis_id,
            "hazard_point_id": hazard["hazard_point_id"],
            "hazard_type": hazard_type,
            "path_length_km": round(length_m / 1000.0, 3),
            "path_elevation_drop_m": round(elevation_drop_m, 2),
            "corridor_half_width_m": half_width,
            "model_version": SCRIPT_VERSION,
        }
        path_records.append({**common, "geometry": line})
        zone_records.append({**common, "geometry": zone})
    paths = gpd.GeoDataFrame(path_records, geometry="geometry", crs=profile["crs"])
    zones = gpd.GeoDataFrame(zone_records, geometry="geometry", crs=profile["crs"])
    return paths, zones


def align_population_raster(
    cfg: Config,
    profile: dict[str, Any],
) -> np.ndarray | None:
    if cfg.population_path is None or not cfg.population_path.exists():
        return None
    out = np.zeros((profile["height"], profile["width"]), dtype="float32")
    with rasterio.open(cfg.population_path) as src:
        if src.crs is None:
            raise ValueError(f"人口栅格缺少 CRS：{cfg.population_path}")
        resampling = getattr(Resampling, "sum", Resampling.average)
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            dst_nodata=0.0,
            resampling=resampling,
        )
    out[~np.isfinite(out) | (out < 0)] = 0.0
    return out


def sum_population_in_geometry(
    geometry: Any,
    population: np.ndarray | None,
    profile: dict[str, Any],
) -> float | None:
    if population is None or geometry is None or geometry.is_empty:
        return None
    raw_window = from_bounds(*geometry.bounds, transform=profile["transform"])
    row_start = max(0, int(math.floor(raw_window.row_off)))
    col_start = max(0, int(math.floor(raw_window.col_off)))
    row_stop = min(population.shape[0], int(math.ceil(raw_window.row_off + raw_window.height)))
    col_stop = min(population.shape[1], int(math.ceil(raw_window.col_off + raw_window.width)))
    if row_stop <= row_start or col_stop <= col_start:
        return 0.0
    window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
    local_transform = rasterio.windows.transform(window, profile["transform"])
    mask = rasterize(
        [(mapping(geometry), 1)],
        out_shape=(row_stop - row_start, col_stop - col_start),
        transform=local_transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)
    data = population[row_start:row_stop, col_start:col_stop]
    return float(np.nansum(data[mask]))


def link_impacted_settlements(
    cfg: Config,
    hazard_points: gpd.GeoDataFrame,
    flow_paths: gpd.GeoDataFrame,
    impact_zones: gpd.GeoDataFrame,
    settlements: gpd.GeoDataFrame,
    dem: np.ndarray,
    profile: dict[str, Any],
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    settlement_elevation = sample_array_at_points(settlements.geometry, dem, profile)
    geographic_settlements = settlements.to_crs("EPSG:4490")
    population = align_population_raster(cfg, profile)
    links: list[dict[str, Any]] = []
    link_geometries: list[LineString | None] = []

    for hazard_index, hazard in hazard_points.iterrows():
        path = flow_paths.iloc[hazard_index].geometry
        zone = impact_zones.iloc[hazard_index].geometry
        zone_population = sum_population_in_geometry(zone, population, profile)
        try:
            candidate_indices = list(settlements.sindex.query(zone, predicate="intersects"))
        except Exception:
            candidate_indices = [
                i for i, geom in enumerate(settlements.geometry) if geom.intersects(zone)
            ]
        accepted = 0
        for settlement_index in candidate_indices:
            settlement = settlements.iloc[settlement_index]
            settlement_z = settlement_elevation[settlement_index]
            hazard_z = float(hazard["elevation_m"])
            # 地震次生类型仍以滑坡影响为对象，也执行坡下筛选。
            if np.isfinite(settlement_z) and settlement_z > hazard_z + cfg.elevation_tolerance_m:
                continue
            straight_distance_m = float(hazard.geometry.distance(settlement.geometry))
            path_distance_m = float(path.project(settlement.geometry))
            max_length = cfg.path_max_length_m.get(str(hazard["hazard_type"]), 8000.0)
            distance_decay = max(0.0, 1.0 - path_distance_m / max(max_length, 1.0))
            if np.isfinite(settlement_z):
                elevation_drop = max(0.0, hazard_z - settlement_z)
                elevation_factor = min(1.0, 0.5 + elevation_drop / 1000.0)
            else:
                elevation_drop = np.nan
                elevation_factor = 0.5
            impact_score_value = (float(hazard["score"]) / 100.0) * distance_decay * elevation_factor
            geo_settlement = geographic_settlements.iloc[settlement_index]
            links.append(
                {
                    "analysis_id": cfg.analysis_id,
                    "hazard_point_id": hazard["hazard_point_id"],
                    "point_lon": hazard["point_lon"],
                    "point_lat": hazard["point_lat"],
                    "hazard_type": hazard["hazard_type"],
                    "secondary_hazard_types": hazard["secondary_hazard_types"],
                    "susceptibility_score": hazard["score"],
                    "susceptibility_level": hazard["hazard_level"],
                    "possible_triggers": hazard["possible_triggers"],
                    "trigger_evidence": hazard["trigger_evidence"],
                    "settlement_id": settlement["settlement_id"],
                    "settlement_name": settlement["settlement_name"],
                    "settlement_source_geometry": settlement["source_geometry"],
                    "name_match_distance_m": (
                        round(float(settlement["name_match_distance_m"]), 1)
                        if np.isfinite(settlement["name_match_distance_m"])
                        else np.nan
                    ),
                    "settlement_lon": round(float(geo_settlement.geometry.x), 7),
                    "settlement_lat": round(float(geo_settlement.geometry.y), 7),
                    "path_distance_km": round(path_distance_m / 1000.0, 3),
                    "straight_distance_km": round(straight_distance_m / 1000.0, 3),
                    "elevation_drop_m": (
                        round(float(elevation_drop), 2) if np.isfinite(elevation_drop) else np.nan
                    ),
                    "impact_score": round(impact_score_value * 100.0, 2),
                    "impact_level": impact_level(impact_score_value),
                    "exposed_population": (
                        round(zone_population, cfg.population_round_digits)
                        if zone_population is not None
                        else np.nan
                    ),
                    "population_scope": "该隐患点整个影响廊道内的估算人口",
                    "impact_method": "区域筛查：DEM最陡下降D8路径+影响廊道+居民地高程筛选",
                    "confidence_level": "中" if np.isfinite(settlement_z) else "低",
                    "model_version": SCRIPT_VERSION,
                }
            )
            link_geometries.append(LineString([hazard.geometry, settlement.geometry]))
            accepted += 1

        if accepted == 0 and cfg.include_points_without_settlement:
            links.append(
                {
                    "analysis_id": cfg.analysis_id,
                    "hazard_point_id": hazard["hazard_point_id"],
                    "point_lon": hazard["point_lon"],
                    "point_lat": hazard["point_lat"],
                    "hazard_type": hazard["hazard_type"],
                    "secondary_hazard_types": hazard["secondary_hazard_types"],
                    "susceptibility_score": hazard["score"],
                    "susceptibility_level": hazard["hazard_level"],
                    "possible_triggers": hazard["possible_triggers"],
                    "trigger_evidence": hazard["trigger_evidence"],
                    "settlement_id": None,
                    "settlement_name": "影响范围内未识别居民地",
                    "settlement_source_geometry": None,
                    "name_match_distance_m": np.nan,
                    "settlement_lon": np.nan,
                    "settlement_lat": np.nan,
                    "path_distance_km": np.nan,
                    "straight_distance_km": np.nan,
                    "elevation_drop_m": np.nan,
                    "impact_score": 0.0,
                    "impact_level": "未识别",
                    "exposed_population": (
                        round(zone_population, cfg.population_round_digits)
                        if zone_population is not None
                        else np.nan
                    ),
                    "population_scope": "该隐患点整个影响廊道内的估算人口",
                    "impact_method": "区域筛查：DEM最陡下降D8路径+影响廊道+居民地高程筛选",
                    "confidence_level": "低",
                    "model_version": SCRIPT_VERSION,
                }
            )
            link_geometries.append(None)

    table = pd.DataFrame(links)
    link_gdf = gpd.GeoDataFrame(table.copy(), geometry=link_geometries, crs=profile["crs"])
    return table, link_gdf


def write_raster(
    path: Path,
    array: np.ndarray,
    profile: dict[str, Any],
    cfg: Config,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.where(np.isfinite(array), array, cfg.output_nodata).astype("float32")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def write_outputs(
    cfg: Config,
    profile: dict[str, Any],
    scores: dict[str, np.ndarray],
    hazard_points: gpd.GeoDataFrame,
    flow_paths: gpd.GeoDataFrame,
    impact_zones: gpd.GeoDataFrame,
    impact_table: pd.DataFrame,
    impact_links: gpd.GeoDataFrame,
    coverage: dict[str, float],
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for scenario, score in scores.items():
        write_raster(
            cfg.output_dir / f"QTP_RES_SUSCEPTIBILITY_{scenario}_RS.tif",
            score * 100.0,
            profile,
            cfg,
        )

    gpkg_path = cfg.output_dir / "QTP_RES_HAZARD_ANALYSIS_V4.gpkg"
    hazard_points.to_file(gpkg_path, layer="RESULT_SUSCEPTIBILITY_PT", driver="GPKG")
    flow_paths.to_file(gpkg_path, layer="RESULT_FLOWPATH_LN", driver="GPKG")
    impact_zones.to_file(gpkg_path, layer="RESULT_IMPACTZONE_PG", driver="GPKG")
    spatial_links = impact_links[
        ~impact_links.geometry.is_empty & impact_links.geometry.notna()
    ].copy()
    if not spatial_links.empty:
        spatial_links.to_file(gpkg_path, layer="RESULT_HAZARD_SETTLEMENT_LN", driver="GPKG")

    hazard_csv = hazard_points.drop(columns="geometry").copy()
    hazard_csv.to_csv(
        cfg.output_dir / "QTP_RES_SUSCEPTIBILITY_PT.csv",
        index=False,
        encoding="utf-8-sig",
    )
    impact_table.to_csv(
        cfg.output_dir / "QTP_RES_HAZARD_SETTLEMENT_TB.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = (
        hazard_points.groupby(["hazard_type", "hazard_level"], dropna=False)
        .size()
        .rename("hazard_point_count")
        .reset_index()
    )
    summary.to_csv(
        cfg.output_dir / "QTP_RES_HAZARD_SUMMARY_TB.csv",
        index=False,
        encoding="utf-8-sig",
    )
    model_weights = cfg.weights["SHAP_DATA_DRIVEN"]
    weight_table = pd.DataFrame(
        [
            {
                "factor_code": name,
                "factor_name": FACTOR_LABELS[name],
                "original_shap_weight_pct": SHAP_ORIGINAL_PERCENT[name],
                "normalized_model_weight": model_weights[name],
                "normalized_model_weight_pct": model_weights[name] * 100.0,
            }
            for name in FACTOR_NAMES
        ]
    )
    weight_table.to_csv(
        cfg.output_dir / "QTP_RES_MODEL_WEIGHT_TB.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "script_version": SCRIPT_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_id": cfg.analysis_id,
        "method_scope": "区域易发性与潜在影响筛查，不是发生时间预报或工程级灾害模拟",
        "static_score_factors": list(FACTOR_NAMES),
        "weight_source": "本项目历史灾害样本CatBoost模型Mean(|SHAP|)贡献率，按100.00182%归一化为1.00",
        "factor_direction_policy": "距离类越近得分越高；降水、高程和坡度按单调风险转换；土地覆盖、冻土和坡向按历史灾害类别富集度转换",
        "dynamic_trigger_policy": "多年平均降水、距历史地震点和冻土进入易发性评分；未来强降雨、未来地震及升温冻融仅作潜在触发解释",
        "impact_method": "DEM最陡下降D8近似路径、影响廊道、居民地高程筛选和人口栅格估算",
        "factor_coverage": coverage,
        "config": serializable_config(cfg),
    }
    with (cfg.output_dir / "QTP_RES_HAZARD_ANALYSIS_METADATA.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def run_analysis(cfg: Config) -> None:
    setup_logging(cfg.output_dir)
    validate_weights(cfg)
    validate_paths(cfg)
    logging.info("读取并建立统一 DEM 分析格网")
    dem, profile = read_dem_to_analysis_grid(cfg)
    valid = np.isfinite(dem)
    logging.info(
        "分析格网：%s × %s，像元 %.1f m，有效像元 %s",
        profile["width"],
        profile["height"],
        cfg.cell_size_m,
        int(valid.sum()),
    )

    logging.info("构造11项SHAP数据驱动评分因子及潜在触发说明")
    factors_by_scenario, raw, triggers = build_factors(cfg, dem, profile)
    coverage = validate_factor_coverage(cfg, factors_by_scenario, valid)
    logging.info("因子覆盖率：%s", ", ".join(f"{k}={v:.1%}" for k, v in coverage.items()))

    scores = {
        scenario: weighted_score(factors_by_scenario[scenario], weights, valid)
        for scenario, weights in cfg.weights.items()
    }
    logging.info("提取候选隐患点并判识潜在灾种")
    hazard_points = extract_candidate_points(
        cfg,
        dem,
        profile,
        factors_by_scenario,
        raw,
        triggers,
        scores,
    )
    logging.info("提取隐患点 %s 个", len(hazard_points))

    logging.info("关联可能受影响居民地")
    settlements = load_settlements(cfg, profile)
    flow_paths, impact_zones = build_impact_geometries(
        cfg, hazard_points, dem, profile
    )
    impact_table, impact_links = link_impacted_settlements(
        cfg,
        hazard_points,
        flow_paths,
        impact_zones,
        settlements,
        dem,
        profile,
    )
    logging.info("生成隐患点—居民地记录 %s 条", len(impact_table))

    write_outputs(
        cfg,
        profile,
        scores,
        hazard_points,
        flow_paths,
        impact_zones,
        impact_table,
        impact_links,
        coverage,
    )
    logging.info("完成。输出目录：%s", cfg.output_dir)


def print_configuration_check(cfg: Config) -> None:
    validate_weights(cfg)
    print("v4 SHAP数据驱动权重检查：通过")
    for scenario, weights in cfg.weights.items():
        print(f"  {scenario}: {sum(weights.values()):.2f}")
    print("\n必需输入：")
    for name, path in required_paths(cfg).items():
        print(f"  [{'OK' if path.exists() else '缺失'}] {name}: {path}")
    print("\n可选输入：")
    for name, path in optional_paths(cfg).items():
        state = "未配置" if path is None else ("OK" if path.exists() else "缺失/跳过")
        print(f"  [{state}] {name}: {path}")
    print("\n必需 GDB 图层：")
    layers: set[str] = set()
    if cfg.gdb_path.exists():
        try:
            layers = list_gdb_layers(cfg.gdb_path)
        except Exception as exc:
            print(f"  [无法读取] {exc}")
    for name, layer in required_layers(cfg).items():
        print(f"  [{'OK' if layer in layers else '缺失'}] {name}: {layer}")
    print("\n可选 GDB 图层：")
    for name, layer in optional_layers(cfg).items():
        state = "未配置" if layer is None else ("OK" if layer in layers else "缺失/跳过")
        print(f"  [{state}] {name}: {layer}")
    precip_count = 0
    if cfg.precipitation_dir is not None and cfg.precipitation_dir.exists():
        precip_count = len(list(cfg.precipitation_dir.glob(cfg.precipitation_pattern)))
    print(f"\n年度降水文件匹配数：{precip_count}（预期23）")
    print("正式投影参数：中央经线89.5°，原点纬度32.5°，标准纬线27.5°和37.5°。")
    print("\n注意：不需要未来降水或未来地震数据。")
    print("多年平均降水、距历史地震点和冻土已进入11项SHAP数据驱动评分。")
    print("未来强降雨、未来地震和升温冻融只作为潜在触发条件解释。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="青藏高原潜在地质灾害点与居民地影响分析 v4（SHAP数据驱动）"
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="只检查权重、路径和配置，不执行空间分析",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_config:
        print_configuration_check(CONFIG)
        return
    run_analysis(CONFIG)


if __name__ == "__main__":
    main()

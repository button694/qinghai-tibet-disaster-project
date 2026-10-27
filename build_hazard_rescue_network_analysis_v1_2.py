#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
青藏高原危险点—应急救援设施网络关联分析 v1.2

功能
----
1. 读取 NewDefault.gdb 中正式保存的 RES_SUSCEPTIBILITY_PT。
2. 将公路 EMG_ROAD_LN 与铁路 EMG_RAILWAY_LN 构造成无向交通网络。
3. 对每个危险点分别寻找网络距离最近的医院、应急救援站和消防站，
   每条完整最短路线形成一个线要素并赋予唯一 ROUTE_CODE。
4. 对机场只计算直线距离最近者，不将机场接入公路铁路救援网络。
5. 输出 RES 前缀的路线、设施、机场关系表及图层。

方法边界
--------
- 医院、应急救援站和消防站的“最近”是公路＋铁路网络最短距离。
- 机场的“最近”是投影坐标系下的平面直线距离，不生成地面路线。
- 危险点和设施通过直线连接到最近网络节点；接入距离单独记录。
- 公路与铁路只在几何节点重合或落入节点吸附容差时连通。
- 未提供道路等级、限速、交通管制、桥隧状态和铁路换乘站，因此本脚本
  不计算真实救援时间，也不等同于导航系统或灾时交通仿真。
- 所有输入只读，脚本不修改 NewDefault.gdb，也不写入 SQL Server。

依赖
----
python -m pip install numpy pandas scipy geopandas shapely pyogrio networkx
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
try:
    import networkx as nx
except ModuleNotFoundError as exc:
    raise SystemExit(
        "缺少 networkx。请先在当前 Python 环境执行："
        "python -m pip install networkx"
    ) from exc
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point


SCRIPT_VERSION = "1.2.0-20260916-RES-ROUTE-AIRPORT-SEPARATE"


@dataclass
class FacilityConfig:
    code: str
    name: str
    layer: str


@dataclass
class Config:
    # ---------- 输入 ----------
    project_root: Path = Path(".")
    gdb_path: Path = Path("data/NewDefault.gdb")
    # 潜在灾害点已经正式写入 NewDefault.gdb，直接读取该图层。
    hazard_layer: str = "RES_SUSCEPTIBILITY_PT"
    road_layer: str = "EMG_ROAD_LN"
    railway_layer: str = "EMG_RAILWAY_LN"
    facilities: tuple[FacilityConfig, ...] = field(
        default_factory=lambda: (
            FacilityConfig("HOSPITAL", "医院", "EMG_HOSPITAL_PT"),
            FacilityConfig("RESCUE_STATION", "应急救援站", "EMG_RESCUESTATION_PT"),
            FacilityConfig("FIRE_STATION", "消防站", "EMG_FIRESTATION_PT"),
        )
    )
    airport: FacilityConfig = field(
        default_factory=lambda: FacilityConfig(
            "AIRPORT", "机场", "EMG_AIRPORT_PT"
        )
    )

    # 与危险点脚本一致的正式分析坐标系，单位为米。
    target_crs: str = (
        "+proj=lcc +lat_1=27.5 +lat_2=37.5 +lat_0=32.5 +lon_0=89.5 "
        "+ellps=GRS80 +units=m +no_defs"
    )

    # ---------- 网络构造 ----------
    # 将长线段按此长度细分，以减小点到最近网络节点的接入误差。
    max_graph_segment_m: float = 2000.0
    # 端点在该容差内视为同一交通节点。
    node_snap_tolerance_m: float = 20.0
    # 危险点或设施离网络过远时，不把直线接入段误称为可达路线。
    max_hazard_snap_m: float = 20000.0
    max_facility_snap_m: float = 10000.0

    # ---------- 字段候选 ----------
    hazard_id_fields: tuple[str, ...] = (
        "hazard_point_id", "HAZARD_POINT_ID", "ID", "SOURCE_OID", "OBJECTID"
    )
    hazard_type_fields: tuple[str, ...] = (
        "hazard_type", "HAZARD_TYPE"
    )
    hazard_level_fields: tuple[str, ...] = (
        "hazard_level", "HAZARD_LEVEL"
    )
    hazard_score_fields: tuple[str, ...] = (
        "score", "score_SHAP", "SUSCEPTIBILITY_SCORE"
    )
    hazard_analysis_id_fields: tuple[str, ...] = (
        "analysis_id", "ANALYSIS_ID"
    )
    facility_id_fields: tuple[str, ...] = (
        "ID", "id", "SOURCE_OID", "source_oid", "OBJECTID", "FID"
    )
    facility_name_fields: tuple[str, ...] = (
        "name", "NAME", "Name", "facility_name", "FACILITY_NAME",
        "站名", "名称", "医院名称"
    )

    # ---------- 输出 ----------
    output_dir: Path = Path("outputs/rescue_network")
    hazard_analysis_id_default: str = "QTP_HAZARD_SHAP"
    analysis_id: str = "QTP_RES_HAZARD_RESCUE_SHAP"


CONFIG = Config()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                output_dir / "hazard_rescue_network_analysis_v1.log",
                encoding="utf-8",
            ),
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
        if hasattr(value, "__dataclass_fields__"):
            return convert(asdict(value))
        return value

    return convert(asdict(cfg))


def list_layers(path: Path) -> set[str]:
    try:
        import pyogrio

        return {str(row[0]) for row in pyogrio.list_layers(path)}
    except Exception:
        try:
            import fiona

            return set(fiona.listlayers(path))
        except Exception as exc:
            raise RuntimeError(f"无法列出图层：{path}；{exc}") from exc


def choose_field(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    columns = list(columns)
    exact = set(columns)
    for candidate in candidates:
        if candidate in exact:
            return candidate
    lower_map = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        found = lower_map.get(str(candidate).lower())
        if found is not None:
            return found
    return None


def read_layer(path: Path, layer: str, label: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer)
    if gdf.empty:
        raise ValueError(f"{label}图层为空：{path} / {layer}")
    if gdf.crs is None:
        raise ValueError(f"{label}缺少坐标系：{path} / {layer}")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ValueError(f"{label}没有有效几何：{path} / {layer}")
    return gdf


def validate_config(cfg: Config) -> set[str]:
    if cfg.node_snap_tolerance_m <= 0 or cfg.max_graph_segment_m <= 0:
        raise ValueError("网络节点容差和最大线段长度必须大于0")
    if not cfg.gdb_path.exists():
        raise FileNotFoundError(f"File GDB不存在：{cfg.gdb_path}")
    gdb_layers = list_layers(cfg.gdb_path)
    required_gdb = {
        cfg.hazard_layer,
        cfg.road_layer,
        cfg.railway_layer,
        cfg.airport.layer,
        *(facility.layer for facility in cfg.facilities),
    }
    missing = sorted(required_gdb - gdb_layers)
    if missing:
        raise FileNotFoundError("GDB缺少必需图层：" + "、".join(missing))
    return gdb_layers


def line_parts(geometry: Any) -> Iterable[LineString]:
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms
    elif getattr(geometry, "geom_type", "") == "GeometryCollection":
        for part in geometry.geoms:
            yield from line_parts(part)


def node_key(x: float, y: float, tolerance_m: float) -> tuple[int, int]:
    return int(round(x / tolerance_m)), int(round(y / tolerance_m))


def node_xy(key: tuple[int, int], tolerance_m: float) -> tuple[float, float]:
    return key[0] * tolerance_m, key[1] * tolerance_m


def subdivide_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    max_length_m: float,
) -> list[tuple[float, float]]:
    x1, y1 = start[:2]
    x2, y2 = end[:2]
    length = math.hypot(x2 - x1, y2 - y1)
    pieces = max(1, int(math.ceil(length / max_length_m)))
    return [
        (x1 + (x2 - x1) * i / pieces, y1 + (y2 - y1) * i / pieces)
        for i in range(pieces + 1)
    ]


def add_network_lines(
    graph: nx.MultiGraph,
    lines: gpd.GeoDataFrame,
    mode: str,
    cfg: Config,
) -> int:
    added = 0
    for geometry in lines.geometry:
        for part in line_parts(geometry):
            coords = list(part.coords)
            for start, end in zip(coords[:-1], coords[1:]):
                densified = subdivide_segment(start, end, cfg.max_graph_segment_m)
                for a, b in zip(densified[:-1], densified[1:]):
                    u = node_key(a[0], a[1], cfg.node_snap_tolerance_m)
                    v = node_key(b[0], b[1], cfg.node_snap_tolerance_m)
                    if u == v:
                        continue
                    ux, uy = node_xy(u, cfg.node_snap_tolerance_m)
                    vx, vy = node_xy(v, cfg.node_snap_tolerance_m)
                    length_m = math.hypot(vx - ux, vy - uy)
                    if length_m <= 0:
                        continue
                    graph.add_node(u, x=ux, y=uy)
                    graph.add_node(v, x=vx, y=vy)
                    graph.add_edge(
                        u,
                        v,
                        length_m=length_m,
                        mode=mode,
                    )
                    added += 1
    return added


def build_network(
    cfg: Config,
) -> tuple[nx.MultiGraph, list[tuple[int, int]], cKDTree]:
    road = read_layer(cfg.gdb_path, cfg.road_layer, "公路").to_crs(cfg.target_crs)
    railway = read_layer(
        cfg.gdb_path, cfg.railway_layer, "铁路"
    ).to_crs(cfg.target_crs)
    graph = nx.MultiGraph()
    road_edges = add_network_lines(graph, road, "ROAD", cfg)
    railway_edges = add_network_lines(graph, railway, "RAILWAY", cfg)
    if graph.number_of_nodes() == 0:
        raise RuntimeError("公路和铁路没有构成有效网络")
    nodes = list(graph.nodes)
    coords = np.asarray(
        [[graph.nodes[node]["x"], graph.nodes[node]["y"]] for node in nodes],
        dtype="float64",
    )
    logging.info(
        "交通网络：节点%s，边%s（公路%s，铁路%s）",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        road_edges,
        railway_edges,
    )
    return graph, nodes, cKDTree(coords)


def snap_geometries_to_nodes(
    geometries: gpd.GeoSeries,
    nodes: list[tuple[int, int]],
    tree: cKDTree,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    coords = np.asarray([[geom.x, geom.y] for geom in geometries], dtype="float64")
    distances, indices = tree.query(coords, k=1)
    return [nodes[int(index)] for index in indices], np.asarray(distances)


def normalize_hazards(cfg: Config) -> gpd.GeoDataFrame:
    hazards = read_layer(
        cfg.gdb_path, cfg.hazard_layer, "潜在危险点"
    ).to_crs(cfg.target_crs)
    id_field = choose_field(hazards.columns, cfg.hazard_id_fields)
    type_field = choose_field(hazards.columns, cfg.hazard_type_fields)
    level_field = choose_field(hazards.columns, cfg.hazard_level_fields)
    score_field = choose_field(hazards.columns, cfg.hazard_score_fields)
    analysis_id_field = choose_field(
        hazards.columns, cfg.hazard_analysis_id_fields
    )
    hazards["_hazard_id"] = (
        hazards[id_field].astype(str)
        if id_field
        else [f"HZ{i:06d}" for i in range(1, len(hazards) + 1)]
    )
    hazards["_hazard_type"] = (
        hazards[type_field].astype(str) if type_field else "未分类"
    )
    hazards["_hazard_level"] = (
        hazards[level_field].astype(str) if level_field else "未分级"
    )
    hazards["_hazard_score"] = (
        pd.to_numeric(hazards[score_field], errors="coerce")
        if score_field
        else np.nan
    )
    hazards["_hazard_analysis_id"] = (
        hazards[analysis_id_field].fillna(cfg.hazard_analysis_id_default).astype(str)
        if analysis_id_field
        else cfg.hazard_analysis_id_default
    )
    return hazards


def normalize_facilities(
    cfg: Config,
    facility_cfg: FacilityConfig,
) -> gpd.GeoDataFrame:
    facilities = read_layer(
        cfg.gdb_path, facility_cfg.layer, facility_cfg.name
    ).to_crs(cfg.target_crs)
    # 理论上应为点；非点几何采用代表点，避免质心落到面外。
    facilities["geometry"] = facilities.geometry.apply(
        lambda geom: geom if isinstance(geom, Point) else geom.representative_point()
    )
    id_field = choose_field(facilities.columns, cfg.facility_id_fields)
    name_field = choose_field(facilities.columns, cfg.facility_name_fields)
    facilities["_facility_id"] = (
        facilities[id_field].astype(str)
        if id_field
        else [f"{facility_cfg.code}_{i:06d}" for i in range(1, len(facilities) + 1)]
    )
    facilities["_facility_name"] = (
        facilities[name_field].fillna("").astype(str)
        if name_field
        else ""
    )
    facilities["_facility_type_code"] = facility_cfg.code
    facilities["_facility_type_name"] = facility_cfg.name
    facilities["_source_layer"] = facility_cfg.layer
    return facilities


def facility_sources(
    facilities: gpd.GeoDataFrame,
    nodes: list[tuple[int, int]],
    tree: cKDTree,
    cfg: Config,
) -> tuple[dict[tuple[int, int], list[int]], int]:
    snapped_nodes, snap_distances = snap_geometries_to_nodes(
        facilities.geometry, nodes, tree
    )
    facilities["_network_node"] = snapped_nodes
    facilities["_snap_distance_m"] = snap_distances
    mapping: dict[tuple[int, int], list[int]] = {}
    excluded = 0
    for index, row in facilities.iterrows():
        if float(row["_snap_distance_m"]) > cfg.max_facility_snap_m:
            excluded += 1
            continue
        mapping.setdefault(row["_network_node"], []).append(index)
    return mapping, excluded


def chosen_edge(
    graph: nx.MultiGraph,
    u: tuple[int, int],
    v: tuple[int, int],
) -> dict[str, Any]:
    edge_variants = graph.get_edge_data(u, v)
    if not edge_variants:
        raise RuntimeError(f"路径边不存在：{u} -> {v}")
    return min(edge_variants.values(), key=lambda item: float(item["length_m"]))


def route_geometry_and_modes(
    graph: nx.MultiGraph,
    source_to_hazard_path: list[tuple[int, int]],
    hazard_point: Point,
    facility_point: Point,
) -> tuple[LineString, float, float, str, int, int, int]:
    road_m = 0.0
    railway_m = 0.0
    road_edge_count = 0
    railway_edge_count = 0
    modes: list[str] = []
    for u, v in zip(source_to_hazard_path[:-1], source_to_hazard_path[1:]):
        edge = chosen_edge(graph, u, v)
        mode = str(edge["mode"])
        length_m = float(edge["length_m"])
        if mode == "RAILWAY":
            railway_m += length_m
            railway_edge_count += 1
        else:
            road_m += length_m
            road_edge_count += 1
        if not modes or modes[-1] != mode:
            modes.append(mode)

    # Dijkstra路径方向为设施源节点->危险点节点，输出路线改为危险点->设施。
    route_nodes = list(reversed(source_to_hazard_path))
    coords: list[tuple[float, float]] = [(hazard_point.x, hazard_point.y)]
    coords.extend((graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in route_nodes)
    coords.append((facility_point.x, facility_point.y))
    deduplicated = [coords[0]]
    for coordinate in coords[1:]:
        if coordinate != deduplicated[-1]:
            deduplicated.append(coordinate)
    if len(deduplicated) == 1:
        deduplicated.append(deduplicated[0])
    return (
        LineString(deduplicated),
        road_m,
        railway_m,
        ">".join(reversed(modes)),
        road_edge_count + railway_edge_count,
        road_edge_count,
        railway_edge_count,
    )


def no_route_record(
    cfg: Config,
    hazard: pd.Series,
    facility_cfg: FacilityConfig,
    status: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "analysis_id": cfg.analysis_id,
        "hazard_analysis_id": hazard["_hazard_analysis_id"],
        "hazard_point_id": hazard["_hazard_id"],
        "hazard_type": hazard["_hazard_type"],
        "hazard_level": hazard["_hazard_level"],
        "susceptibility_score": hazard["_hazard_score"],
        "facility_type_code": facility_cfg.code,
        "facility_type_name": facility_cfg.name,
        "facility_id": "",
        "facility_name": "",
        "facility_source_layer": facility_cfg.layer,
        "route_rank": 1,
        "route_status": status,
        "status_detail": detail,
        "route_code": "",
        "network_distance_km": np.nan,
        "hazard_access_km": np.nan,
        "facility_access_km": np.nan,
        "total_route_distance_km": np.nan,
        "straight_distance_km": np.nan,
        "road_distance_km": np.nan,
        "railway_distance_km": np.nan,
        "network_mode_sequence": "",
        "network_edge_count": 0,
        "road_edge_count": 0,
        "railway_edge_count": 0,
        "model_version": SCRIPT_VERSION,
        "analysis_time": datetime.now().isoformat(timespec="seconds"),
    }


def analyze_facility_type(
    cfg: Config,
    graph: nx.MultiGraph,
    nodes: list[tuple[int, int]],
    tree: cKDTree,
    hazards: gpd.GeoDataFrame,
    hazard_nodes: list[tuple[int, int]],
    hazard_snap_m: np.ndarray,
    facility_cfg: FacilityConfig,
    route_counter_start: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    facilities = normalize_facilities(cfg, facility_cfg)
    source_mapping, excluded_facilities = facility_sources(
        facilities, nodes, tree, cfg
    )
    records: list[dict[str, Any]] = []
    route_features: list[dict[str, Any]] = []
    selected_facilities: list[dict[str, Any]] = []
    route_counter = route_counter_start

    if excluded_facilities:
        logging.warning(
            "%s有%s个设施距网络超过%.0fm，未作为网络源点",
            facility_cfg.name,
            excluded_facilities,
            cfg.max_facility_snap_m,
        )
    if not source_mapping:
        for _, hazard in hazards.iterrows():
            records.append(
                no_route_record(
                    cfg, hazard, facility_cfg, "NO_FACILITY_ON_NETWORK",
                    "该类设施均未能在允许距离内接入交通网络",
                )
            )
        return records, route_features, selected_facilities, route_counter

    distances, paths = nx.multi_source_dijkstra(
        graph, set(source_mapping), weight="length_m"
    )

    for position, (_, hazard) in enumerate(hazards.iterrows()):
        hazard_node = hazard_nodes[position]
        hazard_snap = float(hazard_snap_m[position])
        if hazard_snap > cfg.max_hazard_snap_m:
            records.append(
                no_route_record(
                    cfg, hazard, facility_cfg, "HAZARD_TOO_FAR_FROM_NETWORK",
                    f"危险点距最近网络节点{hazard_snap:.0f}m，超过允许值",
                )
            )
            continue
        if hazard_node not in distances or hazard_node not in paths:
            records.append(
                no_route_record(
                    cfg, hazard, facility_cfg, "NETWORK_DISCONNECTED",
                    "危险点所在网络分量内没有可达的该类设施",
                )
            )
            continue

        path = paths[hazard_node]
        source_node = path[0]
        facility_indices = source_mapping[source_node]
        facility_index = min(
            facility_indices,
            key=lambda idx: float(facilities.loc[idx, "_snap_distance_m"]),
        )
        facility = facilities.loc[facility_index]
        facility_point = facility.geometry
        network_m = float(distances[hazard_node])
        facility_snap = float(facility["_snap_distance_m"])
        total_m = hazard_snap + network_m + facility_snap
        straight_m = hazard.geometry.distance(facility_point)
        (
            route_geom,
            road_m,
            railway_m,
            mode_sequence,
            network_edge_count,
            road_edge_count,
            railway_edge_count,
        ) = route_geometry_and_modes(graph, path, hazard.geometry, facility_point)

        route_counter += 1
        # 一条完整的危险点—救援设施最短路线只使用一个路线编码；
        # 路线内部即使经过多个道路/铁路网段，也不拆成多条成果记录。
        route_code = f"RES_ROUTE_{route_counter:08d}"
        analysis_time = datetime.now().isoformat(timespec="seconds")
        record = {
            "analysis_id": cfg.analysis_id,
            "hazard_analysis_id": hazard["_hazard_analysis_id"],
            "hazard_point_id": hazard["_hazard_id"],
            "hazard_type": hazard["_hazard_type"],
            "hazard_level": hazard["_hazard_level"],
            "susceptibility_score": hazard["_hazard_score"],
            "facility_type_code": facility_cfg.code,
            "facility_type_name": facility_cfg.name,
            "facility_id": facility["_facility_id"],
            "facility_name": facility["_facility_name"],
            "facility_source_layer": facility["_source_layer"],
            "route_rank": 1,
            "route_status": "OK",
            "status_detail": "按公路铁路网络距离选出的最近设施",
            "route_code": route_code,
            "network_distance_km": round(network_m / 1000.0, 3),
            "hazard_access_km": round(hazard_snap / 1000.0, 3),
            "facility_access_km": round(facility_snap / 1000.0, 3),
            "total_route_distance_km": round(total_m / 1000.0, 3),
            "straight_distance_km": round(straight_m / 1000.0, 3),
            "road_distance_km": round(road_m / 1000.0, 3),
            "railway_distance_km": round(railway_m / 1000.0, 3),
            "network_mode_sequence": mode_sequence,
            "network_edge_count": network_edge_count,
            "road_edge_count": road_edge_count,
            "railway_edge_count": railway_edge_count,
            "model_version": SCRIPT_VERSION,
            "analysis_time": analysis_time,
        }
        records.append(record)
        route_features.append({**record, "geometry": route_geom})
        selected_facilities.append(
            {
                "analysis_id": cfg.analysis_id,
                "route_code": route_code,
                "hazard_point_id": hazard["_hazard_id"],
                "facility_type_code": facility_cfg.code,
                "facility_type_name": facility_cfg.name,
                "facility_id": facility["_facility_id"],
                "facility_name": facility["_facility_name"],
                "source_layer": facility["_source_layer"],
                "geometry": facility_point,
            }
        )

    return records, route_features, selected_facilities, route_counter


def analyze_nearest_airports(
    cfg: Config,
    hazards: gpd.GeoDataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按直线距离选择最近机场；机场不参与公路铁路网络寻路。"""
    airports = normalize_facilities(cfg, cfg.airport)
    airport_coords = np.asarray(
        [[geom.x, geom.y] for geom in airports.geometry], dtype="float64"
    )
    if airport_coords.size == 0:
        raise ValueError("机场图层没有有效点")
    airport_tree = cKDTree(airport_coords)
    hazard_coords = np.asarray(
        [[geom.x, geom.y] for geom in hazards.geometry], dtype="float64"
    )
    distances, indices = airport_tree.query(hazard_coords, k=1)

    records: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    analysis_time = datetime.now().isoformat(timespec="seconds")
    for position, (_, hazard) in enumerate(hazards.iterrows(), start=1):
        airport = airports.iloc[int(indices[position - 1])]
        relation_code = f"RES_AIRPORT_{position:08d}"
        record = {
            "analysis_id": cfg.analysis_id,
            "hazard_analysis_id": hazard["_hazard_analysis_id"],
            "relation_code": relation_code,
            "hazard_point_id": hazard["_hazard_id"],
            "hazard_type": hazard["_hazard_type"],
            "hazard_level": hazard["_hazard_level"],
            "susceptibility_score": hazard["_hazard_score"],
            "airport_id": airport["_facility_id"],
            "airport_name": airport["_facility_name"],
            "airport_source_layer": airport["_source_layer"],
            "straight_distance_km": round(
                float(distances[position - 1]) / 1000.0, 3
            ),
            "selection_method": "NEAREST_STRAIGHT_DISTANCE",
            "route_code": "",
            "route_status": "AIRPORT_NO_GROUND_ROUTE",
            "model_version": SCRIPT_VERSION,
            "analysis_time": analysis_time,
        }
        records.append(record)
        features.append({**record, "geometry": airport.geometry})
    return records, features


def write_outputs(
    cfg: Config,
    records: list[dict[str, Any]],
    route_features: list[dict[str, Any]],
    selected_facilities: list[dict[str, Any]],
    airport_records: list[dict[str, Any]],
    airport_features: list[dict[str, Any]],
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records)
    table_path = cfg.output_dir / "QTP_RES_HAZARD_RESCUE_ROUTE_TB.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")

    issues = table[table["route_status"] != "OK"].copy()
    issues.to_csv(
        cfg.output_dir / "QTP_RES_HAZARD_RESCUE_ROUTE_ISSUES_TB.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = (
        table.groupby(["facility_type_name", "route_status"], dropna=False)
        .agg(
            record_count=("hazard_point_id", "size"),
            mean_total_distance_km=("total_route_distance_km", "mean"),
            max_total_distance_km=("total_route_distance_km", "max"),
        )
        .reset_index()
    )
    summary.to_csv(
        cfg.output_dir / "QTP_RES_HAZARD_RESCUE_ROUTE_SUMMARY_TB.csv",
        index=False,
        encoding="utf-8-sig",
    )

    airport_table = pd.DataFrame(airport_records)
    airport_table.to_csv(
        cfg.output_dir / "QTP_RES_HAZARD_AIRPORT_TB.csv",
        index=False,
        encoding="utf-8-sig",
    )

    gpkg_path = cfg.output_dir / "QTP_RES_HAZARD_RESCUE.gpkg"
    if gpkg_path.exists():
        gpkg_path.unlink()
    wrote_gpkg = False
    if route_features:
        routes = gpd.GeoDataFrame(
            route_features, geometry="geometry", crs=cfg.target_crs
        )
        routes.to_file(
            gpkg_path,
            layer="RES_HAZARD_RESCUE_ROUTE_LN",
            driver="GPKG",
        )
        wrote_gpkg = True
    if selected_facilities:
        selected = gpd.GeoDataFrame(
            selected_facilities, geometry="geometry", crs=cfg.target_crs
        )
        selected.to_file(
            gpkg_path,
            layer="RES_SELECTED_RESCUE_PT",
            driver="GPKG",
            mode="a" if wrote_gpkg else "w",
        )
        wrote_gpkg = True
    if airport_features:
        nearest_airports = gpd.GeoDataFrame(
            airport_features, geometry="geometry", crs=cfg.target_crs
        )
        nearest_airports.to_file(
            gpkg_path,
            layer="RES_NEAREST_AIRPORT_PT",
            driver="GPKG",
            mode="a" if wrote_gpkg else "w",
        )

    metadata = {
        "script_version": SCRIPT_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_id": cfg.analysis_id,
        "method": (
            "医院/应急救援站/消防站采用公路铁路联合无向网络和多源Dijkstra；"
            "机场仅采用直线最近距离"
        ),
        "unit": "米用于计算，结果表以千米保存",
        "limitations": [
            "无道路等级、速度、管制和灾损信息，不计算真实救援时间",
            "铁路公路仅在重合网络节点处换乘",
            "危险点和设施通过最近网络节点接入，接入段为直线近似",
            "机场不生成公路铁路路线，仅保存最近机场及直线距离",
        ],
        "config": serializable_config(cfg),
    }
    with (cfg.output_dir / "QTP_RES_HAZARD_RESCUE_METADATA.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)


def run_analysis(cfg: Config) -> None:
    setup_logging(cfg.output_dir)
    validate_config(cfg)
    logging.info("读取危险点")
    hazards = normalize_hazards(cfg)
    logging.info("危险点数量：%s", len(hazards))

    logging.info("构建公路铁路联合网络")
    graph, nodes, tree = build_network(cfg)
    hazard_nodes, hazard_snap_m = snap_geometries_to_nodes(
        hazards.geometry, nodes, tree
    )

    all_records: list[dict[str, Any]] = []
    all_routes: list[dict[str, Any]] = []
    all_selected: list[dict[str, Any]] = []
    route_counter = 0
    for facility_cfg in cfg.facilities:
        logging.info("计算最近%s及最短路线", facility_cfg.name)
        records, routes, selected, route_counter = analyze_facility_type(
            cfg,
            graph,
            nodes,
            tree,
            hazards,
            hazard_nodes,
            hazard_snap_m,
            facility_cfg,
            route_counter,
        )
        all_records.extend(records)
        all_routes.extend(routes)
        all_selected.extend(selected)

    logging.info("计算每个危险点直线距离最近的机场（不生成地面路线）")
    airport_records, airport_features = analyze_nearest_airports(cfg, hazards)

    write_outputs(
        cfg,
        all_records,
        all_routes,
        all_selected,
        airport_records,
        airport_features,
    )
    ok_count = sum(record["route_status"] == "OK" for record in all_records)
    logging.info(
        "完成：关系记录%s条，可达路线%s条，问题记录%s条",
        len(all_records),
        ok_count,
        len(all_records) - ok_count,
    )
    logging.info("输出目录：%s", cfg.output_dir)


def print_config_check(cfg: Config) -> None:
    print(f"脚本版本：{SCRIPT_VERSION}")
    print("\n输入文件：")
    print(f"  [{'OK' if cfg.gdb_path.exists() else '缺失'}] File GDB: {cfg.gdb_path}")
    print("\n目标图层：")
    gdb_layers: set[str] = set()
    if cfg.gdb_path.exists():
        try:
            gdb_layers = list_layers(cfg.gdb_path)
        except Exception as exc:
            print(f"  [无法读取GDB] {exc}")
    layer_items = [
        ("潜在危险点", cfg.hazard_layer),
        ("公路", cfg.road_layer),
        ("铁路", cfg.railway_layer),
        *((facility.name, facility.layer) for facility in cfg.facilities),
        (cfg.airport.name, cfg.airport.layer),
    ]
    for label, layer in layer_items:
        print(f"  [{'OK' if layer in gdb_layers else '缺失'}] {label}: {layer}")
    print(f"\n输出目录：{cfg.output_dir}")
    print("每个危险点生成医院、应急救援站、消防站各1条网络最短路线记录。")
    print("机场只生成最近机场及直线距离记录，不生成公路铁路路线。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="青藏高原危险点—应急救援设施网络关联分析"
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="只检查路径和图层，不执行网络分析",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_config:
        print_config_check(CONFIG)
        return
    run_analysis(CONFIG)


if __name__ == "__main__":
    main()

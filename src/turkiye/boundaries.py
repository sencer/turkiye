from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import geopandas as gpd
import shapely

from turkiye._common import (
  LEVELS,
  PACKAGE_ROOT,
  alias_key,
  name_aliases,
  normalize_name,
)

if TYPE_CHECKING:
  from collections.abc import Callable

  from shapely.geometry.base import BaseGeometry

BoundaryLevel = Literal["il", "ilce", "belde", "mahalle"]

MAP_ROOT = PACKAGE_ROOT / "data" / "maps"
TURKIYE_ILCE_SHAPEFILE = MAP_ROOT / "tur_polbna_adm2.shp"
TURKIYE_ILCE_SIMPLIFIED_SHAPEFILE = MAP_ROOT / "tur_polbna_adm2_simplified.shp"
TURKIYE_MAJOR_LAKES_GEOJSON = MAP_ROOT / "turkiye_major_lakes.geojson"


@dataclass(frozen=True, slots=True)
class GeometrySource:
  name: str
  level: BoundaryLevel
  source: str
  license: str
  attribution: str
  access_mode: str
  supported_levels: tuple[BoundaryLevel, ...]
  path: Path | None = None
  source_url: str | None = None
  year: int | None = None
  build_date: str | None = None
  caveats: str = ""
  non_commercial: bool = False
  exportable: bool = True


_SOURCES: dict[str, GeometrySource] = {
  "default": GeometrySource(
    name="default",
    level="ilce",
    source="HDX/OCHA Türkiye COD-AB ADM2",
    license="See HDX/OCHA COD-AB source terms; redistributed as packaged shapefile from local ysk data.",
    attribution="OCHA ROAP, HDX, and original administrative boundary contributors.",
    access_mode="bundled",
    supported_levels=("il", "ilce"),
    path=TURKIYE_ILCE_SHAPEFILE,
    source_url="https://data.humdata.org/",
    build_date="2026-05-22",
    caveats=(
      "Province boundaries are derived by dissolving the ADM2 district layer. "
      "Major lakes from Natural Earth are subtracted from default geometries. "
      "Historical polygons are not provided."
    ),
  ),
  "hgm": GeometrySource(
    name="hgm",
    level="ilce",
    source="Harita Genel Müdürlüğü administrative boundaries and settlement data",
    license="Opt-in offline source; non-commercial use and attribution restrictions may apply.",
    attribution="Harita Genel Müdürlüğü",
    access_mode="offline-user-provided",
    supported_levels=("il", "ilce", "belde", "mahalle"),
    caveats="Not bundled. Register a local HGM file before loading.",
    non_commercial=True,
  ),
  "arcgis_michael_bauer_2024": GeometrySource(
    name="arcgis_michael_bauer_2024",
    level="ilce",
    source="ArcGIS/Michael Bauer Türkiye boundaries 2024",
    license="Reference only subject to upstream service terms and user credentials.",
    attribution="Michael Bauer Research GmbH / ArcGIS service attribution as applicable.",
    access_mode="online-reference-only",
    supported_levels=("il", "ilce"),
    year=2024,
    caveats="Not bundled, cached, exported, or redistributed by this package.",
    exportable=False,
  ),
}


def map_path(*, simplified: bool = False) -> Path:
  shapefile = (
    TURKIYE_ILCE_SIMPLIFIED_SHAPEFILE if simplified else TURKIYE_ILCE_SHAPEFILE
  )
  if shapefile.exists():
    return shapefile
  suffix = "_simplified" if simplified else ""
  msg = (
    "default Türkiye administrative map not found; reinstall a wheel that "
    f"includes data/maps/tur_polbna_adm2{suffix}.*"
  )
  raise FileNotFoundError(msg)


def load_boundaries(
  level: BoundaryLevel = "il",
  *,
  year: int | None = None,
  source: str = "default",
  simplified: bool = False,
) -> gpd.GeoDataFrame:
  if level not in LEVELS:
    msg = f"unknown boundary level {level!r}"
    raise ValueError(msg)
  source_info = get_geometry_source(source)
  if level not in source_info.supported_levels:
    msg = f"source {source!r} does not support {level!r} boundaries"
    raise ValueError(msg)
  if source_info.access_mode == "online-reference-only":
    msg = f"source {source!r} is online-only reference metadata and cannot be loaded"
    raise ValueError(msg)
  if source_info.path is None:
    msg = f"source {source!r} has no registered local geometry path"
    raise ValueError(msg)
  if source == "default":
    return _default_boundaries_cached(level, simplified).copy()
  return _load_registered_source(source_info, level=level, year=year)


def register_geometry_source(  # noqa: PLR0913
  path: str | Path,
  *,
  level: BoundaryLevel,
  source_name: str,
  license: str,
  attribution: str,
  source_url: str | None = None,
  year: int | None = None,
) -> GeometrySource:
  if level not in LEVELS:
    msg = f"unknown boundary level {level!r}"
    raise ValueError(msg)
  geometry_path = Path(path)
  if not geometry_path.exists():
    msg = f"geometry path does not exist: {geometry_path}"
    raise FileNotFoundError(msg)
  if not source_name.strip():
    msg = "source_name is required"
    raise ValueError(msg)
  if not license.strip():
    msg = "license is required"
    raise ValueError(msg)
  if not attribution.strip():
    msg = "attribution is required"
    raise ValueError(msg)
  source = GeometrySource(
    name=source_name,
    level=level,
    source=source_name,
    license=license,
    attribution=attribution,
    access_mode="registered-local",
    supported_levels=(level,),
    path=geometry_path,
    source_url=source_url,
    year=year,
    caveats="User-registered geometry source.",
  )
  _SOURCES[source_name] = source
  _registered_source_cached.cache_clear()
  return source


def get_geometry_source(source: str) -> GeometrySource:
  try:
    return _SOURCES[source]
  except KeyError as exc:
    msg = f"unknown geometry source {source!r}"
    raise ValueError(msg) from exc


def list_geometry_sources() -> dict[str, GeometrySource]:
  return dict(_SOURCES)


@lru_cache(maxsize=8)
def _default_boundaries_cached(
  level: BoundaryLevel,
  simplified: bool,
) -> gpd.GeoDataFrame:
  if level not in {"il", "ilce"}:
    msg = f"default source does not include {level!r} geometry"
    raise ValueError(msg)
  frame = _read_geodataframe(map_path(simplified=simplified))
  if frame.crs is None:
    frame = frame.set_crs("EPSG:4326")
  frame = _with_public_location_columns(frame)
  frame = _without_major_lakes(frame)
  if level == "ilce":
    return frame.sort_values(["il", "ilce"]).reset_index(drop=True)
  dissolve = cast("Callable[..., gpd.GeoDataFrame]", frame.dissolve)
  provinces = dissolve("il", as_index=False)
  return provinces.sort_values("il").reset_index(drop=True)


def _load_registered_source(
  source: GeometrySource,
  *,
  level: BoundaryLevel,
  year: int | None,
) -> gpd.GeoDataFrame:
  frame = _registered_source_cached(str(source.path), level, year).copy()
  return _with_public_location_columns(frame)


@lru_cache(maxsize=16)
def _registered_source_cached(
  path: str,
  _level: BoundaryLevel,
  _year: int | None,
) -> gpd.GeoDataFrame:
  return _read_geodataframe(Path(path))


def _read_geodataframe(path: Path) -> gpd.GeoDataFrame:
  read_file = cast("Callable[[Path], gpd.GeoDataFrame]", gpd.read_file)
  return read_file(path)


def _without_major_lakes(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
  lakes = _major_lakes_union(frame.crs)
  result = frame.copy()
  result.geometry = result.geometry.difference(lakes)
  return result


@lru_cache(maxsize=4)
def _major_lakes_union(crs: Any) -> BaseGeometry:
  lakes = _read_geodataframe(TURKIYE_MAJOR_LAKES_GEOJSON)
  if lakes.crs is None:
    lakes = lakes.set_crs("EPSG:4326")
  if crs is not None and lakes.crs != crs:
    lakes = lakes.to_crs(crs)
  union_all = cast("Callable[..., BaseGeometry]", shapely.union_all)
  return union_all(lakes.geometry.to_numpy())


def plot_boundaries(
  level: BoundaryLevel,
  *,
  simplified: bool,
  simplify_tolerance: float | None,
) -> gpd.GeoDataFrame:
  return _plot_boundaries_cached(level, simplified, simplify_tolerance).copy()


@lru_cache(maxsize=16)
def _plot_boundaries_cached(
  level: BoundaryLevel,
  simplified: bool,
  simplify_tolerance: float | None,
) -> gpd.GeoDataFrame:
  if (
    level == "ilce"
    and simplified
    and simplify_tolerance is None
    and map_path(simplified=True).exists()
  ):
    return load_boundaries("ilce", simplified=True)
  boundaries = load_boundaries(level, simplified=simplified)
  if simplify_tolerance is None:
    return boundaries
  return boundaries.set_geometry(
    _plot_simplified_geometries(
      boundaries,
      simplify_tolerance=simplify_tolerance,
    )
  )


def _plot_simplified_geometries(
  gdf: gpd.GeoDataFrame,
  *,
  simplify_tolerance: float,
) -> gpd.GeoSeries:
  simplified = gdf.geometry.simplify(
    simplify_tolerance,
    preserve_topology=True,
  )
  orient_polygons = cast("Callable[..., object]", shapely.orient_polygons)
  simplified = orient_polygons(simplified.to_numpy(), exterior_cw=True)
  return gpd.GeoSeries(cast("Any", simplified), index=gdf.index, crs=gdf.crs)


def _with_public_location_columns(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
  result = frame.copy()
  if "adm1_en" in result.columns and "il" not in result.columns:
    result["il"] = result["adm1_en"].map(normalize_name)
  if "adm2_en" in result.columns and "ilce" not in result.columns:
    result["ilce"] = result["adm2_en"].map(normalize_name)
  if "il" in result.columns:
    result["il_aliases"] = [
      _row_aliases(row, "il", "adm1_tr", "adm1_en") for _, row in result.iterrows()
    ]
    result["il_key"] = result["il"].map(alias_key)
  if {"il", "ilce"}.issubset(result.columns):
    result["ilce_aliases"] = [_ilce_aliases(row) for _, row in result.iterrows()]
    result["ilce_key"] = result["ilce"].map(alias_key)
  return result


def _row_aliases(row: Any, column: str, *source_columns: str) -> tuple[str, ...]:
  aliases: list[str] = []
  for source_column in (column, *source_columns):
    value = row.get(source_column)
    if value is not None:
      aliases.extend(name_aliases(value))
  return tuple(dict.fromkeys(aliases))


def _ilce_aliases(row: Any) -> tuple[str, ...]:
  aliases = list(_row_aliases(row, "ilce", "adm2_tr", "adm2_en"))
  il_aliases = _row_aliases(row, "il", "adm1_tr", "adm1_en")
  if set(map(alias_key, aliases)).intersection(map(alias_key, il_aliases)):
    aliases.extend(name_aliases("MERKEZ"))
  if alias_key(row["ilce"]).endswith(" MERKEZ"):
    aliases.extend(name_aliases("MERKEZ"))
  return tuple(dict.fromkeys(aliases))

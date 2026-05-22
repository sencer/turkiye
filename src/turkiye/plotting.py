from __future__ import annotations

from collections.abc import Callable, Mapping
import colorsys
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from typing import Any, Literal, cast
import warnings

import geopandas as gpd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd

from turkiye._common import alias_key, normalize_name, turkish_title
from turkiye.boundaries import BoundaryLevel, load_boundaries, plot_boundaries

HoverCallback = Callable[[pd.Series], str]
Backend = Literal["plotly", "matplotlib"]

DEFAULT_SIMPLIFY_TOLERANCE = 0.01
_crosshair_css_installed = False


@dataclass(frozen=True, slots=True)
class _DataSpec:
  level: BoundaryLevel
  frame: pd.DataFrame


@dataclass(frozen=True, slots=True)
class _ChoroplethOptions:
  color: str
  color_map: Mapping[str, str] | None
  height: int
  width: int | None
  plot_options: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _InteractiveMapStyle:
  legend: bool
  categories: list[str]
  height: int
  width: int | None
  lon_range: list[float]
  lat_range: list[float]


def plot(
  data: pd.DataFrame,
  *,
  backend: Backend = "plotly",
  **kwargs: Any,
) -> Any:
  if backend == "plotly":
    return plot_interactive(data, **kwargs)
  if backend == "matplotlib":
    return plot_static(data, **kwargs)
  msg = f"unknown backend {backend!r}"
  raise ValueError(msg)


def plot_interactive(  # noqa: PLR0913
  data: pd.DataFrame,
  *,
  color: str,
  label: str | None = None,
  tooltip_fn: HoverCallback | None = None,
  year: int | None = None,
  level: BoundaryLevel | Literal["auto"] = "auto",
  index_names: Mapping[str, str] | None = None,
  color_map: Mapping[str, str] | None = None,
  **kwargs: Any,
) -> Any:
  _install_crosshair_css()
  px = _plotly_express_module()
  spec = _prepare_data(data, level=level, index_names=index_names)
  _require_column(spec.frame, color)
  map_level = _available_geometry_level(spec.level)
  boundaries = plot_boundaries(
    map_level,
    simplified=True,
    simplify_tolerance=kwargs.pop("simplify_tolerance", DEFAULT_SIMPLIFY_TOLERANCE),
  )
  gdf = _merge_map_data(boundaries, spec.frame, level=map_level)
  gdf = gdf.reset_index(drop=True)
  lon_range, lat_range = _geo_ranges(gdf)
  plot_frame = _plot_frame(
    gdf,
    color=color,
    label=label,
    level=map_level,
    tooltip_fn=tooltip_fn,
  )
  plot_options: dict[str, Any] = dict(kwargs)
  height = plot_options.pop("height", 720)
  width = plot_options.pop("width", None)
  fig, categories = _choropleth_figure(
    px,
    gdf,
    plot_frame,
    _ChoroplethOptions(
      color=color,
      color_map=color_map,
      height=height,
      width=width,
      plot_options=plot_options,
    ),
  )
  _style_interactive_map(
    fig,
    _InteractiveMapStyle(
      legend=True,
      categories=categories,
      height=height,
      width=width,
      lon_range=lon_range,
      lat_range=lat_range,
    ),
  )
  if map_level == "ilce":
    _add_province_borders(fig)
  _disable_scroll_zoom(fig)
  _warn_if_fell_back(spec.level, map_level, year=year)
  return fig


def plot_static(  # noqa: PLR0913, PLR0914
  data: pd.DataFrame,
  *,
  color: str,
  label: str | None = None,
  year: int | None = None,
  level: BoundaryLevel | Literal["auto"] = "auto",
  index_names: Mapping[str, str] | None = None,
  ax: Any = None,
  color_map: Mapping[str, str] | None = None,
  **kwargs: Any,
) -> Any:
  spec = _prepare_data(data, level=level, index_names=index_names)
  _require_column(spec.frame, color)
  map_level = _available_geometry_level(spec.level)
  simplify_tolerance = kwargs.pop("simplify_tolerance", DEFAULT_SIMPLIFY_TOLERANCE)
  legend = kwargs.pop("legend", True)
  legend_kwargs = dict(kwargs.pop("legend_kwargs", {}) or {})
  if label is not None:
    legend_kwargs.setdefault("label", label)
  boundaries = plot_boundaries(
    map_level,
    simplified=True,
    simplify_tolerance=simplify_tolerance,
  )
  gdf = _merge_map_data(boundaries, spec.frame, level=map_level)
  created_axes = ax is None
  if ax is None:
    subplots = cast("Callable[..., tuple[Any, Any]]", plt.subplots)
    _, ax = subplots(figsize=kwargs.pop("figsize", (12, 10)))
  if created_axes and legend:
    ax.figure.subplots_adjust(right=0.84)
  ax.axis("off")
  boundaries.plot(ax=ax, color="#dddddd", edgecolor="white", linewidth=0.2)
  if _is_numeric_column(gdf, color):
    numeric_legend_kwargs = _numeric_legend_kwargs(legend_kwargs)
    gdf.plot(
      ax=ax,
      column=color,
      edgecolor="white",
      linewidth=0.2,
      legend=legend,
      legend_kwds=numeric_legend_kwargs,
      **kwargs,
    )
  else:
    resolved_color_map = _resolved_color_map(gdf[color], color_map=color_map)
    categories = _legend_categories(gdf[color])
    for category in categories:
      raw_category = _raw_category_for_label(gdf[color], category)
      subset = gdf[gdf[color] == raw_category]
      if not subset.empty:
        subset.plot(
          ax=ax,
          color=resolved_color_map[category],
          edgecolor="white",
          linewidth=0.2,
          label=category,
          **kwargs,
        )
    if legend:
      _add_categorical_legend(
        ax,
        categories,
        resolved_color_map,
        legend_kwargs=legend_kwargs,
      )
  if map_level == "ilce":
    province_boundaries = load_boundaries("il")
    if simplify_tolerance is not None:
      province_boundaries = province_boundaries.set_geometry(
        province_boundaries.geometry.simplify(
          simplify_tolerance,
          preserve_topology=True,
        )
      )
    boundary_plot = cast("Callable[..., Any]", province_boundaries.boundary.plot)
    boundary_plot(ax=ax, color="white", linewidth=0.55)
  _warn_if_fell_back(spec.level, map_level, year=year)
  return ax


def _prepare_data(
  data: pd.DataFrame,
  *,
  level: BoundaryLevel | Literal["auto"],
  index_names: Mapping[str, str] | None,
) -> _DataSpec:
  frame = _plain_frame(data, index_names=index_names)
  normalized = frame.copy()
  for column in ("il", "ilce", "belde", "mahalle"):
    if column in normalized.columns:
      normalized[column] = normalized[column].map(normalize_name)
  if "il" in normalized.columns:
    normalized["_il_key"] = normalized["il"].map(alias_key)
  if "ilce" in normalized.columns:
    normalized["_ilce_key"] = normalized["ilce"].map(alias_key)
  inferred = _infer_level(normalized) if level == "auto" else level
  return _DataSpec(level=inferred, frame=normalized)


def _plain_frame(
  data: pd.DataFrame,
  *,
  index_names: Mapping[str, str] | None,
) -> pd.DataFrame:
  frame = data.copy()
  rename = {custom: canonical for canonical, custom in (index_names or {}).items()}
  if isinstance(frame.index, pd.MultiIndex):
    if any(name is None for name in frame.index.names):
      msg = "MultiIndex levels must be named or mapped with index_names"
      raise ValueError(msg)
    frame = frame.reset_index().rename(columns=rename)
  elif frame.index.name is not None:
    frame = frame.reset_index().rename(columns=rename)
  else:
    frame = frame.rename(columns=rename)
  return frame


def _infer_level(frame: pd.DataFrame) -> BoundaryLevel:
  if {"il", "ilce", "belde", "mahalle"}.issubset(frame.columns):
    return "mahalle"
  if {"il", "ilce", "belde"}.issubset(frame.columns):
    return "belde"
  if {"il", "ilce"}.issubset(frame.columns):
    return "ilce"
  if "il" in frame.columns:
    return "il"
  msg = "data must include a named il index/column, or il and ilce for district maps"
  raise ValueError(msg)


def _available_geometry_level(level: BoundaryLevel) -> BoundaryLevel:
  if level in {"belde", "mahalle"}:
    return "ilce"
  return level


def _warn_if_fell_back(
  requested: BoundaryLevel,
  actual: BoundaryLevel,
  *,
  year: int | None,
) -> None:
  if requested == actual:
    return
  warnings.warn(
    f"{requested!r} geometry is not bundled; plotted nearest available {actual!r} geometry"
    + (f" after resolving year {year}" if year is not None else ""),
    stacklevel=3,
  )


def _merge_map_data(
  boundaries: gpd.GeoDataFrame,
  frame: pd.DataFrame,
  *,
  level: BoundaryLevel,
) -> gpd.GeoDataFrame:
  if level == "ilce":
    return _ilce_alias_merge(boundaries, frame)
  return boundaries.merge(
    frame,
    left_on="il_key",
    right_on="_il_key",
    how="left",
    suffixes=("", "_data"),
  )


def _ilce_alias_merge(
  boundaries: gpd.GeoDataFrame,
  frame: pd.DataFrame,
) -> gpd.GeoDataFrame:
  aliased_boundaries = _ilce_alias_lookup(boundaries)
  merge_frame = frame.copy()
  merge_frame["_matched"] = True
  merged = aliased_boundaries.merge(
    merge_frame,
    on=["_il_key", "_ilce_key"],
    how="left",
    suffixes=("", "_data"),
  )
  matched = merged[merged["_matched"].eq(True)]
  unmatched = merged[~merged["_boundary_index"].isin(matched["_boundary_index"])]
  result = (
    pd
    .concat([matched, unmatched.drop_duplicates("_boundary_index")], ignore_index=True)
    .sort_values("_boundary_index")
    .drop_duplicates("_boundary_index")
    .drop(columns=["_boundary_index", "_il_key", "_ilce_key", "_matched"])
  )
  return gpd.GeoDataFrame(
    result,
    geometry=boundaries.geometry.name,
    crs=boundaries.crs,
  ).reset_index(drop=True)


def _ilce_alias_lookup(boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
  rows: list[dict[str, object]] = []
  for boundary_index, row in boundaries.reset_index(drop=True).iterrows():
    for il_alias in row["il_aliases"]:
      rows.extend(
        {
          "_boundary_index": boundary_index,
          "_il_key": alias_key(il_alias),
          "_ilce_key": alias_key(ilce_alias),
        }
        for ilce_alias in row["ilce_aliases"]
      )
  lookup = pd.DataFrame(rows).drop_duplicates([
    "_boundary_index",
    "_il_key",
    "_ilce_key",
  ])
  return lookup.merge(
    boundaries.reset_index(names="_boundary_index"),
    on="_boundary_index",
    how="left",
  )


def _plot_frame(
  gdf: gpd.GeoDataFrame,
  *,
  color: str,
  label: str | None,
  level: BoundaryLevel,
  tooltip_fn: HoverCallback | None,
) -> pd.DataFrame:
  columns = ["il", "adm1_tr"]
  if level == "ilce":
    columns.extend(["ilce", "adm2_tr"])
  if color not in columns:
    columns.append(color)
  result = pd.DataFrame(gdf[columns])
  result["hover_text"] = [
    _hover_text(row, color=color, label=label, level=level, tooltip_fn=tooltip_fn)
    for _, row in gdf.iterrows()
  ]
  return result


def _choropleth_figure(
  px: Any,
  gdf: gpd.GeoDataFrame,
  plot_frame: pd.DataFrame,
  options: _ChoroplethOptions,
) -> tuple[Any, list[str]]:
  prepared = plot_frame.reset_index(drop=True).copy()
  prepared["_map_id"] = [f"division-{index}" for index in prepared.index]
  geojson_frame = gdf[["geometry"]].copy()
  geojson_frame["_map_id"] = prepared["_map_id"]
  if _is_numeric_column(plot_frame, options.color):
    return px.choropleth(
      prepared,
      geojson=_geodataframe_json(geojson_frame),
      locations="_map_id",
      color=options.color,
      featureidkey="properties._map_id",
      custom_data=["hover_text"],
      projection="mercator",
      height=options.height,
      width=options.width,
      **options.plot_options,
    ), []
  categories = _legend_categories(prepared[options.color])
  fig = _categorical_choropleth_figure(
    gdf,
    prepared,
    geojson_frame,
    categories,
    options,
  )
  return fig, categories


def _categorical_choropleth_figure(
  _gdf: gpd.GeoDataFrame,
  prepared: pd.DataFrame,
  geojson_frame: gpd.GeoDataFrame,
  categories: list[str],
  options: _ChoroplethOptions,
) -> Any:
  go = _plotly_module()
  geojson = _geodataframe_json(geojson_frame)
  prepared["_category"] = [
    _category_label(str(value)) if pd.notna(value) else "No data"
    for value in prepared[options.color]
  ]
  category_orders = [*categories]
  if prepared["_category"].eq("No data").any():
    category_orders.append("No data")
  color_map = _resolved_color_map(
    prepared[options.color],
    color_map=options.color_map,
  )
  code_by_category = {category: index for index, category in enumerate(category_orders)}
  z_values = [code_by_category[category] for category in prepared["_category"]]
  colorscale = _discrete_colorscale([
    color_map[category] for category in category_orders
  ])
  fig = go.Figure(
    go.Choropleth(
      geojson=geojson,
      locations=prepared["_map_id"],
      z=z_values,
      featureidkey="properties._map_id",
      customdata=[[text] for text in prepared["hover_text"]],
      colorscale=colorscale,
      zmin=0,
      zmax=max(0, len(category_orders) - 1),
      showscale=False,
      hovertemplate="%{customdata[0]}<extra></extra>",
      **options.plot_options,
    )
  )
  for category in categories:
    fig.add_trace(
      go.Scattergeo(
        lon=[None],
        lat=[None],
        mode="markers",
        marker={"color": color_map[category], "size": 9},
        name=category,
        hoverinfo="skip",
        visible="legendonly",
        showlegend=True,
      )
    )
  fig.update_layout(height=options.height, width=options.width)
  return fig


def _discrete_colorscale(colors: list[str]) -> list[list[float | str]]:
  if len(colors) == 1:
    return [[0.0, colors[0]], [1.0, colors[0]]]
  return [[index / (len(colors) - 1), color] for index, color in enumerate(colors)]


def _style_interactive_map(
  fig: Any,
  style: _InteractiveMapStyle,
) -> None:
  fig.update_traces(
    hovertemplate="%{customdata[0]}<extra></extra>",
    marker_line_color="rgba(255,255,255,0.75)",
    marker_line_width=0.12,
    selector={"type": "choropleth"},
  )
  legend_columns = max(1, math.ceil(len(style.categories) / 2))
  for trace in fig.data:
    if trace.name == "No data":
      trace.showlegend = False
  fig.update_layout(
    height=style.height,
    width=style.width,
    margin={"l": 0, "r": 0, "t": 72 if style.legend else 0, "b": 0},
    dragmode="pan",
    showlegend=style.legend,
    legend_title_text="",
    legend={
      "orientation": "h",
      "x": 0,
      "xanchor": "left",
      "y": 1.08,
      "yanchor": "bottom",
      "entrywidth": 1 / legend_columns,
      "entrywidthmode": "fraction",
      "traceorder": "normal",
    },
  )
  fig.update_geos(
    visible=False,
    projection={"type": "mercator"},
    lonaxis={"range": style.lon_range},
    lataxis={"range": style.lat_range},
  )


def _add_province_borders(fig: Any) -> None:
  go = _plotly_module()
  province_boundaries = load_boundaries("il")
  province_boundaries = province_boundaries.set_geometry(
    province_boundaries.geometry.simplify(
      DEFAULT_SIMPLIFY_TOLERANCE,
      preserve_topology=True,
    )
  )
  lon, lat = _boundary_coordinates(province_boundaries.boundary)
  fig.add_trace(
    go.Scattergeo(
      lon=lon,
      lat=lat,
      mode="lines",
      line={"color": "rgba(255,255,255,0.9)", "width": 0.55},
      hoverinfo="skip",
      showlegend=False,
    )
  )


def _disable_scroll_zoom(fig: Any) -> None:
  fig._config = {**getattr(fig, "_config", {}), "scrollZoom": False}


def _hover_text(
  row: pd.Series,
  *,
  color: str,
  label: str | None,
  level: BoundaryLevel,
  tooltip_fn: HoverCallback | None,
) -> str:
  if tooltip_fn is not None:
    return tooltip_fn(row)
  place = _display_place(row, level=level)
  display_label = label if label is not None else color
  return "<br>".join([
    f"<b>{place}</b>",
    "",
    f"{display_label}: {_format_value(row[color])}",
  ])


def _display_place(row: pd.Series, *, level: BoundaryLevel) -> str:
  province = turkish_title(row.get("adm1_tr", row["il"]))
  if level == "il":
    return province
  district = turkish_title(row.get("adm2_tr", row["ilce"]))
  return f"{district}, {province}"


def _format_value(value: object) -> str:
  if _is_missing_value(value):
    return "No data"
  if isinstance(value, int | float):
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.3g}"
  return str(value)


def _is_missing_value(value: object) -> bool:
  return (
    value is None
    or value is pd.NA
    or value is pd.NaT
    or (isinstance(value, float) and math.isnan(value))
  )


def _legend_categories(values: pd.Series) -> list[str]:
  categories = {
    _category_label(str(value))
    for value in values.dropna().unique()
    if str(value) != "No data"
  }
  return sorted(categories, key=str.casefold)


def _category_label(category: str) -> str:
  if category == "No data":
    return category
  if category.isupper():
    return category
  return category.replace("_", " ").title()


def _raw_category_for_label(values: pd.Series, label: str) -> str:
  for value in values.dropna().unique():
    category = str(value)
    if _category_label(category) == label:
      return category
  return label


def _resolved_color_map(
  values: pd.Series,
  *,
  color_map: Mapping[str, str] | None,
) -> dict[str, str]:
  result = {"No data": "#dddddd"}
  for value in values.dropna().unique():
    category = str(value)
    label = _category_label(category)
    if color_map is not None and category in color_map:
      result[label] = color_map[category]
    elif color_map is not None and label in color_map:
      result[label] = color_map[label]
    else:
      result[label] = _stable_color(category)
  return result


def _stable_color(value: str) -> str:
  digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
  hue = int(digest[:8], 16) / 0xFFFFFFFF
  red, green, blue = colorsys.hsv_to_rgb(hue, 0.55, 0.75)
  return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def _add_categorical_legend(
  ax: Any,
  categories: list[str],
  color_map: Mapping[str, str],
  *,
  legend_kwargs: Mapping[str, Any] | None,
) -> None:
  handles = [
    Patch(facecolor=color_map[category], edgecolor="white", label=category)
    for category in categories
  ]
  if handles:
    options: dict[str, Any] = {
      "bbox_to_anchor": (1.02, 0.5),
      "borderaxespad": 0,
      "frameon": False,
      "loc": "center left",
    }
    options.update(dict(legend_kwargs or {}))
    ax.legend(handles=handles, **options)


def _numeric_legend_kwargs(legend_kwargs: Mapping[str, Any]) -> dict[str, Any]:
  options: dict[str, Any] = {
    "fraction": 0.035,
    "pad": 0.02,
    "shrink": 0.72,
  }
  options.update(dict(legend_kwargs))
  return options


def _geo_ranges(gdf: gpd.GeoDataFrame) -> tuple[list[float], list[float]]:
  min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
  lon_padding = (max_lon - min_lon) * 0.03
  lat_padding = (max_lat - min_lat) * 0.05
  return (
    [float(min_lon - lon_padding), float(max_lon + lon_padding)],
    [float(min_lat - lat_padding), float(max_lat + lat_padding)],
  )


def _geodataframe_json(gdf: gpd.GeoDataFrame) -> Any:
  to_json = cast("Callable[[], str]", gdf.to_json)
  return json.loads(to_json())


def _is_numeric_column(frame: pd.DataFrame, column: str) -> bool:
  return pd.api.types.is_numeric_dtype(frame[column])


def _require_column(frame: pd.DataFrame, column: str) -> None:
  if column in frame.columns:
    return
  msg = f"{column!r} column not found"
  raise ValueError(msg)


def _plotly_module() -> Any:
  return importlib.import_module("plotly.graph_objects")


def _plotly_express_module() -> Any:
  return importlib.import_module("plotly.express")


def _install_crosshair_css() -> None:
  global _crosshair_css_installed  # noqa: PLW0603
  if _crosshair_css_installed:
    return
  try:
    ipython = importlib.import_module("IPython")
    display_module = importlib.import_module("IPython.display")
  except ModuleNotFoundError:
    return
  get_ipython = getattr(ipython, "get_" + "ipython")
  shell = get_ipython()
  if shell is None or shell.__class__.__name__ != "ZMQInteractiveShell":
    return
  html = getattr(display_module, "HT" + "ML")
  display = getattr(display_module, "dis" + "play")
  display(
    html(
      """
      <style>
      .js-plotly-plot .plotly,
      .js-plotly-plot .plotly .drag,
      .js-plotly-plot .plotly .nsewdrag,
      .js-plotly-plot .plotly .cursor-crosshair {
        cursor: crosshair !important;
      }
      </style>
      """
    )
  )
  _crosshair_css_installed = True


def _boundary_coordinates(
  boundaries: gpd.GeoSeries,
) -> tuple[list[float | None], list[float | None]]:
  lon: list[float | None] = []
  lat: list[float | None] = []
  for boundary in boundaries:
    _append_geometry_coordinates(boundary, lon, lat)
  return lon, lat


def _append_geometry_coordinates(
  geometry: object,
  lon: list[float | None],
  lat: list[float | None],
) -> None:
  geoms = getattr(geometry, "geoms", None)
  if geoms is not None:
    for part in geoms:
      _append_geometry_coordinates(part, lon, lat)
    return
  coords = getattr(geometry, "coords", None)
  if coords is None:
    return
  for x, y in coords:
    lon.append(float(x))
    lat.append(float(y))
  lon.append(None)
  lat.append(None)

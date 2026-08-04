from __future__ import annotations

import json
from typing import Any

import geopandas as gpd


def project_geojson(
  geojson: dict[str, Any],
  *,
  crs: str = "EPSG:3857",
) -> dict[str, Any]:
  """Project a WGS84 GeoJSON feature collection into another CRS."""
  if geojson.get("type") != "FeatureCollection":
    msg = "geojson must be a FeatureCollection"
    raise ValueError(msg)
  if not geojson.get("features"):
    return {"type": "FeatureCollection", "features": []}
  frame = gpd.GeoDataFrame.from_features(geojson, crs="EPSG:4326").to_crs(crs)
  return json.loads(frame.to_json(drop_id=True))

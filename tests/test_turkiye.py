import matplotlib.axes
import pandas as pd
import pytest
from shapely.geometry import Point

from turkiye import (
  get_division,
  get_geometry_source,
  load_boundaries,
  map_path,
  plot,
  plot_interactive,
  plot_static,
  register_geometry_source,
)


def test_default_boundaries_load_district_and_province_levels() -> None:
  ilce = load_boundaries("ilce")
  il = load_boundaries("il")

  assert len(ilce) == 973
  assert len(il) == 81
  assert ilce.crs == "EPSG:4326"
  assert {"il", "ilce", "il_aliases", "ilce_aliases", "geometry"}.issubset(ilce.columns)
  canakkale = ilce[ilce["il"].eq("CANAKKALE")].iloc[0]
  sirnak_merkez = ilce[ilce["il"].eq("SIRNAK") & ilce["ilce"].eq("SIRNAK")].iloc[0]
  assert "ÇANAKKALE" in canakkale["il_aliases"]
  assert "CANAKKALE" in canakkale["il_aliases"]
  assert "ŞIRNAK" in sirnak_merkez["ilce_aliases"]
  assert "SIRNAK" in sirnak_merkez["ilce_aliases"]
  assert "MERKEZ" in sirnak_merkez["ilce_aliases"]


def test_simplified_map_path_works() -> None:
  path = map_path(simplified=True)

  assert path.name == "tur_polbna_adm2_simplified.shp"
  assert path.exists()
  assert path.with_suffix(".dbf").exists()
  assert path.with_suffix(".shx").exists()
  assert path.with_suffix(".cpg").exists()


def test_default_boundaries_exclude_major_lakes() -> None:
  ilce = load_boundaries("ilce")
  il = load_boundaries("il")

  lake_van = Point(42.82, 38.66)
  lake_tuz = Point(33.38, 38.75)

  assert not ilce.geometry.covers(lake_van).any()
  assert not ilce.geometry.covers(lake_tuz).any()
  assert not il.geometry.covers(lake_van).any()
  assert not il.geometry.covers(lake_tuz).any()


def test_plotly_returns_choropleth_with_hover_text() -> None:
  data = pd.DataFrame({
    "il": ["SIRNAK", "SIRNAK"],
    "ilce": ["GUCLUKONAK", "MERKEZ"],
    "winner": ["a", "b"],
  })

  fig = plot_interactive(
    data,
    color="winner",
    tooltip_fn=lambda row: f"{row['ilce']} winner {row['winner']}",
  )

  assert fig.data[0].type == "choropleth"
  assert any(trace.type == "scattergeo" and not trace.showlegend for trace in fig.data)
  hover_texts: list[str] = []
  for trace in fig.data:
    custom_data = getattr(trace, "customdata", None)
    if custom_data is None:
      continue
    hover_texts.extend(str(row[0]) for row in custom_data if row[0] is not None)
  assert any("winner" in text for text in hover_texts)
  assert fig._config["scrollZoom"] is False


def test_interactive_plot_matches_turkish_and_ascii_aliases() -> None:
  dotless_sirnak = "sırnak"  # noqa: RUF001
  data = pd.DataFrame({
    "il": ["ÇANAKKALE", "SIRNAK", "ŞIRNAK", "sirnak", dotless_sirnak],
    "ilce": ["MERKEZ", "SIRNAK", "MERKEZ", "sirnak", dotless_sirnak],
    "value": [1, 2, 3, 4, 5],
  })

  fig = plot_interactive(data, color="value")

  values = list(fig.data[0].z)
  assert 1 in values
  assert any(value in values for value in [2, 3, 4, 5])


def test_static_plot_returns_axes_for_numeric_and_categorical() -> None:
  numeric = pd.DataFrame(
    {"value": [1, 2]}, index=pd.Index(["ANKARA", "ISTANBUL"], name="il")
  )
  categorical = pd.DataFrame({
    "il": ["SIRNAK", "SIRNAK"],
    "ilce": ["GUCLUKONAK", "MERKEZ"],
    "winner": ["a", "b"],
  })

  ax_numeric = plot_static(numeric, color="value")
  ax_categorical = plot(categorical, backend="matplotlib", color="winner")

  assert isinstance(ax_numeric, matplotlib.axes.Axes)
  assert isinstance(ax_categorical, matplotlib.axes.Axes)
  assert not ax_numeric.axison


def test_static_plot_places_default_legends_outside_map() -> None:
  numeric = pd.DataFrame(
    {"value": [1, 2]}, index=pd.Index(["ANKARA", "ISTANBUL"], name="il")
  )
  categorical = pd.DataFrame({
    "il": ["SIRNAK", "SIRNAK"],
    "ilce": ["GUCLUKONAK", "MERKEZ"],
    "winner": ["a", "b"],
  })

  ax_numeric = plot_static(numeric, color="value")
  ax_categorical = plot_static(categorical, color="winner")

  numeric_axes = ax_numeric.figure.axes
  assert len(numeric_axes) > 1
  assert numeric_axes[-1].get_position().x0 > ax_numeric.get_position().x1

  legend = ax_categorical.get_legend()
  assert legend is not None
  assert legend.get_frame_on() is False
  assert legend.get_bbox_to_anchor()._bbox.x0 > 1


def test_named_multiindex_and_custom_index_names() -> None:
  data = pd.DataFrame(
    {"value": [1]},
    index=pd.MultiIndex.from_tuples(
      [("SIRNAK", "GUCLUKONAK")],
      names=["province", "district"],
    ),
  )

  fig = plot_interactive(
    data,
    color="value",
    index_names={"il": "province", "ilce": "district"},
  )

  assert fig.data[0].type == "choropleth"


def test_unnamed_index_errors_clearly() -> None:
  data = pd.DataFrame(
    {"value": [1]},
    index=pd.MultiIndex.from_tuples([("SIRNAK", "GUCLUKONAK")]),
  )

  with pytest.raises(ValueError, match="MultiIndex levels must be named"):
    plot_interactive(data, color="value")


def test_derecik_history_pre_and_post_district_creation() -> None:
  old = get_division(il="HAKKARI", ilce="DERECIK", year=2000)
  current = get_division(il="HAKKARI", ilce="DERECIK", year=2024)

  assert old.level == "belde"
  assert old.ilce == "SEMDINLI"
  assert old.geometry_key == ("HAKKARI", "SEMDINLI")
  assert current.level == "ilce"
  assert current.geometry_key == ("HAKKARI", "DERECIK")


def test_belde_plot_falls_back_to_parent_geometry() -> None:
  data = pd.DataFrame({
    "il": ["HAKKARI"],
    "ilce": ["SEMDINLI"],
    "belde": ["DERECIK"],
    "value": [1],
  })

  with pytest.warns(UserWarning, match="not bundled"):
    ax = plot_static(data, color="value")

  assert isinstance(ax, matplotlib.axes.Axes)


def test_source_metadata_and_registered_source_validation() -> None:
  default = get_geometry_source("default")
  hgm = get_geometry_source("hgm")
  arcgis = get_geometry_source("arcgis_michael_bauer_2024")

  assert default.access_mode == "bundled"
  assert "HDX" in default.source
  assert hgm.non_commercial is True
  assert arcgis.access_mode == "online-reference-only"
  assert arcgis.exportable is False

  registered = register_geometry_source(
    map_path(),
    level="ilce",
    source_name="test-local",
    license="Test license",
    attribution="Test attribution",
  )

  assert registered.access_mode == "registered-local"
  assert load_boundaries("ilce", source="test-local").shape[0] == 973

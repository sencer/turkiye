from __future__ import annotations

from dataclasses import dataclass

from turkiye._common import normalize_name


@dataclass(frozen=True, slots=True)
class Division:
  level: str
  il: str
  ilce: str | None = None
  belde: str | None = None
  mahalle: str | None = None
  requested_year: int | None = None
  valid_from: int | None = None
  valid_to: int | None = None
  parent_chain: tuple[str, ...] = ()
  geometry_level: str = "il"
  geometry_key: tuple[str, ...] = ()
  source_references: tuple[str, ...] = ()


def get_division(
  *,
  il: str | None = None,
  ilce: str | None = None,
  belde: str | None = None,
  mahalle: str | None = None,
  year: int | None = None,
) -> Division:
  if il is None:
    msg = "il is required"
    raise ValueError(msg)
  normalized_il = normalize_name(il)
  normalized_ilce = normalize_name(ilce) if ilce is not None else None
  normalized_belde = normalize_name(belde) if belde is not None else None
  normalized_mahalle = normalize_name(mahalle) if mahalle is not None else None

  if normalized_il == "HAKKARI" and normalized_ilce == "DERECIK":
    if year is not None and year < 2018:
      return Division(
        level="belde",
        il="HAKKARI",
        ilce="SEMDINLI",
        belde="DERECIK",
        requested_year=year,
        valid_to=2018,
        parent_chain=("HAKKARI", "SEMDINLI"),
        geometry_level="ilce",
        geometry_key=("HAKKARI", "SEMDINLI"),
        source_references=(
          "İçişleri İl ve İlçe Kuruluş Tarihleri",
          "Curated relationship: Derecik before district creation",
        ),
      )
    return Division(
      level="ilce",
      il="HAKKARI",
      ilce="DERECIK",
      requested_year=year,
      valid_from=2018,
      parent_chain=("HAKKARI",),
      geometry_level="ilce",
      geometry_key=("HAKKARI", "DERECIK"),
      source_references=("İçişleri İl ve İlçe Kuruluş Tarihleri",),
    )

  if normalized_mahalle is not None:
    level = "mahalle"
    geometry_level = "ilce" if normalized_ilce is not None else "il"
  elif normalized_belde is not None:
    level = "belde"
    geometry_level = "ilce" if normalized_ilce is not None else "il"
  elif normalized_ilce is not None:
    level = "ilce"
    geometry_level = "ilce"
  else:
    level = "il"
    geometry_level = "il"

  geometry_key = tuple(
    value
    for value in (normalized_il, normalized_ilce if geometry_level == "ilce" else None)
    if value is not None
  )
  parent_chain = tuple(
    value
    for value in (normalized_il, normalized_ilce, normalized_belde)
    if value is not None
  )
  return Division(
    level=level,
    il=normalized_il,
    ilce=normalized_ilce,
    belde=normalized_belde,
    mahalle=normalized_mahalle,
    requested_year=year,
    parent_chain=parent_chain[:-1] if len(parent_chain) > 1 else parent_chain,
    geometry_level=geometry_level,
    geometry_key=geometry_key,
    source_references=("User supplied current administrative identity",),
  )

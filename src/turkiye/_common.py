from __future__ import annotations

from pathlib import Path
import unicodedata

PACKAGE_ROOT = Path(__file__).resolve().parent

LEVELS = {"il", "ilce", "belde", "mahalle"}


def normalize_name(value: object) -> str:
  return " ".join(str(value).strip().upper().split())


def alias_key(value: object) -> str:
  decomposed = unicodedata.normalize("NFKD", normalize_name(value))
  ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
  return normalize_name(ascii_text)


def name_aliases(value: object) -> tuple[str, ...]:
  normalized = normalize_name(value)
  ascii_alias = alias_key(normalized)
  return tuple(dict.fromkeys((normalized, ascii_alias)))


def turkish_title(value: object) -> str:
  text = str(value).strip()
  lowered = text.translate(str.maketrans({"I": "ı", "İ": "i"})).lower()  # noqa: RUF001
  return " ".join(_capitalize_turkish_word(word) for word in lowered.split())


def _capitalize_turkish_word(word: str) -> str:
  if not word:
    return word
  first = "İ" if word[0] == "i" else word[0].upper()
  return first + word[1:]

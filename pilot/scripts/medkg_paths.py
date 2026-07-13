"""Centralized data path resolution from .env.

Usage:
    from medkg_paths import DATA_ROOT, MEDKG_ROOT, UMLS_DIR, ...

All paths are loaded from /home/max/Graph-DDXPlus/.env. If the env var is not set,
falls back to the legacy /home/max/Graph-DDXPlus/data/X path (which is now usually a
symlink to /windows/data/X).
"""
from __future__ import annotations
import os
from pathlib import Path

ENV_PATH = Path("/home/max/Graph-DDXPlus/.env")
PROJECT_ROOT = Path("/home/max/Graph-DDXPlus")


def _load_env():
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env()


def _path(env_key: str, fallback: str) -> Path:
    return Path(os.environ.get(env_key, fallback))


DATA_ROOT       = _path("DATA_ROOT",       "/windows/data")
MEDKG_ROOT      = _path("MEDKG_ROOT",      str(PROJECT_ROOT / "data" / "medkg"))
UMLS_DIR        = _path("UMLS_DIR",        str(PROJECT_ROOT / "data" / "umls_extracted"))
UMLS_SUBSET_DIR = _path("UMLS_SUBSET_DIR", str(PROJECT_ROOT / "data" / "umls_subset"))
UMLS_FULL_DIR   = _path("UMLS_FULL_DIR",   str(PROJECT_ROOT / "data" / "2025AB-full"))
SNOMED_DIR      = _path("SNOMED_DIR",      str(PROJECT_ROOT / "data" / "snomed_ct_2026"))
SNOMED_OLD_DIR  = _path("SNOMED_OLD_DIR",  str(PROJECT_ROOT / "data" / "snomed_ct"))
DDXPLUS_DIR     = _path("DDXPLUS_DIR",     str(PROJECT_ROOT / "data" / "ddxplus"))
SYMCAT_DIR      = _path("SYMCAT_DIR",      str(PROJECT_ROOT / "data" / "symcat"))
RAREBENCH_DIR   = _path("RAREBENCH_DIR",   str(PROJECT_ROOT / "data" / "rarebench"))
ERREASON_DIR    = _path("ERREASON_DIR",    str(PROJECT_ROOT / "data" / "er_reason"))
SEMMEDDB_DIR    = _path("SEMMEDDB_DIR",    str(PROJECT_ROOT / "data" / "semmeddb"))
EXTERNAL_KG_DIR = _path("EXTERNAL_KG_DIR", str(PROJECT_ROOT / "data" / "external_kg"))

# Subdirectories under MEDKG_ROOT
MEDKG_RAW       = MEDKG_ROOT / "raw"
MEDKG_PROCESSED = MEDKG_ROOT / "processed"
MEDKG_KG        = MEDKG_ROOT / "kg"
MEDKG_LOGS      = MEDKG_ROOT / "logs"


def ensure_dirs():
    """Create all medkg subdirectories if missing."""
    for d in [MEDKG_ROOT, MEDKG_RAW, MEDKG_PROCESSED, MEDKG_KG, MEDKG_LOGS]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("=== medkg_paths configuration ===")
    for var in ["DATA_ROOT", "MEDKG_ROOT", "UMLS_DIR", "UMLS_SUBSET_DIR", "SNOMED_DIR",
                "DDXPLUS_DIR", "SYMCAT_DIR", "RAREBENCH_DIR", "ERREASON_DIR",
                "SEMMEDDB_DIR", "EXTERNAL_KG_DIR"]:
        path = globals()[var]
        exists = "✓" if path.exists() else "✗"
        print(f"  {var:18s} = {path}  [{exists}]")

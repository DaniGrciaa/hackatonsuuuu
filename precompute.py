"""Precalcula las escenas de ejemplo de la app: ejecuta run_all_layers sobre cada escena de data/ y guarda
el resultado en examples/<escena>/ (imágenes base comprimidas, máscaras PNG y metrics.json).

    python precompute.py                 # todas las escenas de data/
    python precompute.py data/niebla     # solo una

examples/ se sube al repositorio para que la app abra los ejemplos al instante, sin reanalizar.
"""
from __future__ import annotations

import sys
from pathlib import Path

from core.ingestion import load_scene
from pipeline import run_all_layers
from results_io import date_from_names, save_analysis
from run import find_scenes

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "examples"

# Texto de las tarjetas "Probar con un incendio real" (lo demás sale del análisis)
INFO = {
    "niebla": {"titulo": "Niebla", "orden": 1},
    "avila_madrid": {"titulo": "Ávila y Madrid", "orden": 2},
    "extra_mierla": {"titulo": "La Mierla", "orden": 3},
}


def precompute(scene_dir: Path) -> Path:
    sc = load_scene(scene_dir)
    res = run_all_layers(sc["tc"], sc["swir"], sc["wildfire"], name=sc["name"])
    meta = {"titulo": sc["name"].replace("_", " ").capitalize(), "orden": 99, **INFO.get(sc["name"], {}),
            "fecha": date_from_names(Path(f).name for f in sc["files"].values())}
    out = save_analysis(res, OUT / sc["name"], meta)
    for lid, r in res["layers"].items():
        print(f"  {lid:12s} {r['status']:13s} {r['time_s']:6.2f} s  {r['error'] or ''}")
    print(f"{sc['name']}: {res['time_s']} s -> {out.relative_to(ROOT)}")
    return out


def main():
    srcs = [Path(a) for a in sys.argv[1:]] or find_scenes(str(ROOT / "data"))
    for s in srcs:
        precompute(s)


if __name__ == "__main__":
    main()

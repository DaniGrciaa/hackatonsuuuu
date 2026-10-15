"""Escenas SIN incendio activo (o previas): no debería salir cicatriz ni focos.

    python tools_c/negativas.py
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.fire import detect_fires, fire_details  # noqa: E402
from core.ingestion import load_scene  # noqa: E402
from core.segmentation import burn_mask  # noqa: E402
from core.spectral import bands, exclusion_masks, indices  # noqa: E402

DL = Path.home() / "Downloads"
CASES = {  # nombre -> carpeta, fecha
    "jun26 (Browser_images-9)": ("Browser_images-9", "2026-06-26"),
    "jun24 antes del incendio (-12)": ("Browser_images-12", "2026-06-24"),
    "jun24 antes del incendio (-14)": ("Browser_images-14", "2026-06-24"),
    "jun09 (Browser_images-10)": ("Browser_images-10", "2026-06-09"),
}


def files(folder: str, date: str) -> list[str]:
    return [f for k in ("SWIR", "True_color", "Wildfires") for f in glob.glob(f"{DL}/{folder}/{date}*_{k}.*")]


if __name__ == "__main__":
    print(f"{'escena':34s} {'% quemado':>9s} {'px fuego':>9s} {'focos':>6s}")
    for name, (d, date) in CASES.items():
        fl = files(d, date)
        if not fl:
            print(f"{name:34s} (no encontrada en {DL / d})")
            continue
        sc = load_scene(fl, name=name)
        b = bands(sc)
        ind = indices(b)
        ms = exclusion_masks(b, ind)
        bm = burn_mask(ind, ms)
        f = detect_fires(b, ind, bm, ms)
        print(f"{name:34s} {bm.mean() * 100:9.2f} {int(f.sum()):9d} {len(fire_details(f, bm, ms)):6d}")

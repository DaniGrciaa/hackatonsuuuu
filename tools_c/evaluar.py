"""Compara variantes de golpe: IoU contra dNBR (práctica 26-07) y % marcado en las demás escenas.

    python tools_c/evaluar.py v3 v4
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.ingestion import load_scene                       # noqa: E402
from core.segmentation import burn_details                  # noqa: E402
from core.spectral import bands, exclusion_masks, indices   # noqa: E402
from medir_dnbr import DEF, POST, THRESH, reference, scores  # noqa: E402

SCENES = ["data/niebla", "data/avila_madrid", "data/extra_mierla", "data_test/sin_fuego"]


def prep(src, name=None):
    sc = load_scene(src, name=name)
    b = bands(sc)
    ind = indices(b)
    return sc, b, ind, exclusion_masks(b, ind)


if __name__ == "__main__":
    variants = sys.argv[1:] or ["v3"]
    R = reference(DEF)
    vis = [str(next(DEF.glob(f"{POST}*_{k}.tiff"))) for k in ("SWIR", "True_color", "Wildfires")]
    cases = {"práctica": prep(vis, "dnbr_0726")} | {Path(s).name: prep(s) for s in SCENES}
    head = "variante | IoU/prec/rec >0.27 | IoU/prec/rec >0.10 | " + " | ".join(f"{k[:10]:>10s}" for k in cases if k != "práctica")
    print(head)
    for v in variants:
        cells = []
        sc, b, ind, ms = cases["práctica"]
        m = burn_details(b, ind, ms, v)["mask"]
        ok = R["ok"] & sc["valid"]
        for t in THRESH:
            s = scores(m, R["refs"][t], ok)
            cells.append(f"{s['IoU']:.2f}/{s['prec']:.2f}/{s['recall']:.2f}    ")
        for k, (sc, b, ind, ms) in cases.items():
            if k == "práctica":
                continue
            m = burn_details(b, ind, ms, v)["mask"]
            cells.append(f"{100 * m.sum() / sc['valid'].sum():9.1f}%")
        print(f"{v:8s} | " + " | ".join(cells))

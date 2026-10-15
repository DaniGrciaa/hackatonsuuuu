"""Banco de medida del puesto C: IoU de la cicatriz contra una referencia dNBR (solo práctica).

En el reto real NO habrá foto previa; aquí usamos las bandas crudas de dos fechas (24-06 antes, 26-07 durante)
solo para MEDIR la solución, que sigue usando únicamente SWIR/True color del 26-07.

    python tools_c/medir_dnbr.py [carpeta_datos_16bit] [--solo-referencia]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.ingestion import load_scene                       # noqa: E402
from core.segmentation import VARIANTS, burn_details        # noqa: E402
from core.spectral import bands, exclusion_masks, indices   # noqa: E402

DEF = Path.home() / "Documents/hackathon/datos_16bit"
PRE, POST = "2026-06-24", "2026-07-26"
PX_M2 = 34.95 * 33.45                                        # tamaño de píxel de este lote (UTM 30N)
THRESH = (0.27, 0.10)                                        # severidad moderada / generosa (USGS)


def band(folder: Path, date: str, name: str) -> np.ndarray:
    f = next(folder.glob(f"{date}*_{name}_(Raw).tiff"))
    return tifffile.imread(f).astype(np.float32) / 65535.0


def reference(folder: Path) -> dict:
    nbr = {}
    nodata = np.zeros((1128, 2500), bool)
    water = np.zeros_like(nodata)
    for d in (PRE, POST):
        b8a, b12 = band(folder, d, "B8A"), band(folder, d, "B12")
        nbr[d] = (b8a - b12) / (b8a + b12 + 1e-6)
        nodata |= (b8a == 0) & (b12 == 0)
        water |= b8a < 0.05                                  # agua: muy poco infrarrojo cercano
    dnbr = nbr[PRE] - nbr[POST]
    ok = ~nodata & ~water
    return {"dnbr": dnbr, "ok": ok, "refs": {t: (dnbr > t) & ok for t in THRESH}}


def scores(pred: np.ndarray, ref: np.ndarray, ok: np.ndarray) -> dict:
    p, r = pred & ok, ref & ok
    tp = (p & r).sum()
    return {"IoU": tp / max((p | r).sum(), 1), "prec": tp / max(p.sum(), 1), "recall": tp / max(r.sum(), 1),
            "area_ratio": p.sum() / max(r.sum(), 1)}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    folder = Path(args[0]) if args else DEF
    R = reference(folder)
    print("== Referencia dNBR (control: ~348 km² para >0,27 y ~620 km² para >0,10 según el lote de práctica)")
    for t, m in R["refs"].items():
        print(f"   dNBR > {t:.2f}: {m.sum():8d} px = {m.sum() * PX_M2 / 1e6:6.0f} km²  ({m.mean() * 100:.1f}% de la imagen)")
    if "--solo-referencia" in sys.argv:
        raise SystemExit

    vis = [str(next(folder.glob(f"{POST}*_{k}.tiff"))) for k in ("SWIR", "True_color", "Wildfires")]
    sc = load_scene(vis, name="dnbr_0726")
    assert sc["size"] == R["ok"].shape, (sc["size"], R["ok"].shape)
    b = bands(sc)
    ind = indices(b)
    ms = exclusion_masks(b, ind)
    ok = R["ok"] & sc["valid"]
    print(f"\n== Solución (solo SWIR/True color del {POST}) contra la referencia")
    print(f"{'variante':8s} {'%img':>6s} | " + " | ".join(f"dNBR>{t:.2f}: IoU  prec  recall  área/ref" for t in THRESH))
    res = {}
    for v in VARIANTS:
        m = burn_details(b, ind, ms, v)["mask"]
        res[v] = m
        row = " | ".join(f"{'':9s}{(sc_ := scores(m, R['refs'][t], ok))['IoU']:.2f}  {sc_['prec']:.2f}  {sc_['recall']:.2f}    {sc_['area_ratio']:.2f}" for t in THRESH)
        print(f"{v:8s} {m.mean() * 100:6.1f} | {row}")

    # Mapa de errores de la variante por defecto contra la referencia principal
    m, ref = res["v3"] & ok, R["refs"][THRESH[0]] & ok
    img = sc["swir"][..., ::-1].copy()
    img = (img * 0.6).astype(np.uint8)
    img[m & ref] = (0, 200, 0)         # verde: acierto
    img[m & ~ref] = (0, 0, 255)        # rojo: falso positivo (marco cicatriz, la referencia no)
    img[~m & ref] = (255, 120, 0)      # azul: omisión (la referencia dice quemado, yo no)
    out = Path("debug/dnbr_0726"); out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "errores_v3.png"), img)
    print(f"\nMapa de errores -> {out / 'errores_v3.png'}  (verde acierto, rojo falso positivo, azul omisión)")

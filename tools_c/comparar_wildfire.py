"""Compara los focos de core/fire.py con la capa Wildfire. SOLO PARA VALIDAR: core/ no lee la Wildfire.

La Wildfire (script QuickFire de Copernicus) marca como "fuego" cualquier cosa muy brillante en el infrarrojo de onda
corta, así que también pinta de rosa el suelo pálido, las minas y las parcelas. Por eso el recall contra ella SUBESTIMA
al detector. Este script lo hace visible: separa los objetos de referencia que son naranjas en la SWIR (fuego probable)
de los que son pálidos (suelo probable) y dice, para cada foco perdido, qué condición del detector falla.

    python tools_c/comparar_wildfire.py                  # niebla y extra_mierla
    python tools_c/comparar_wildfire.py data/niebla --montaje
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.fire import fire_conditions  # noqa: E402
from core.ingestion import load_scene  # noqa: E402
from core.spectral import bands  # noqa: E402

TOL_PX = 5          # tolerancia al emparejar objetos
MIN_PX = 6          # los objetos de referencia más pequeños se tratan como ruido
CAUSAS = ("sin brillo suficiente", "brillante pero no naranja (suelo pálido probable)", "naranja pero sin contraste")


def wildfire_ref(sc: dict) -> np.ndarray:
    """Píxeles naranja muy saturados de la Wildfire (definición estricta, la misma que en el README)."""
    w = sc["wildfire"].astype(int)
    r, g, b = w[..., 0], w[..., 1], w[..., 2]
    return (r > 200) & (r - b > 120) & (r - g > 40) & sc["valid"]


def objetos(mask: np.ndarray, min_px: int = 1):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return lab, [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= min_px], st, cen


def comparar(sc: dict) -> dict:
    b = bands(sc)
    cond = fire_conditions(b)
    det, ref = cond["mask"], wildfire_ref(sc)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TOL_PX + 1, 2 * TOL_PX + 1))
    det_d = cv2.dilate(det.astype(np.uint8), k).astype(bool)
    ref_d = cv2.dilate(ref.astype(np.uint8), k).astype(bool)

    lab_r, ids_r, st_r, cen_r = objetos(ref, MIN_PX)
    lab_d, ids_d, _, _ = objetos(det)
    ok_det = sum(ref_d[lab_d == i].any() for i in ids_d)
    perdidos, hallados = [], []
    for i in ids_r:
        c = lab_r == i
        if det_d[c].any():
            hallados.append(i)
        else:
            if not cond["hot"][c].any():
                causa = CAUSAS[0]
            elif not cond["cand"][c].any():
                causa = CAUSAS[1]
            else:
                causa = CAUSAS[2]
            perdidos.append((i, causa))
    pale = [i for i, c in perdidos if c == CAUSAS[1]]
    ref_fuego = len(ids_r) - len(pale)          # objetos de referencia que no son "pálidos"
    return {
        "ref": len(ids_r), "detectados_en_ref": len(hallados), "perdidos": perdidos,
        "det": len(ids_d), "det_confirmados": int(ok_det),
        "precision": ok_det / max(len(ids_d), 1), "recall": len(hallados) / max(len(ids_r), 1),
        "recall_sin_palidos": len(hallados) / max(ref_fuego, 1),
        "lab_r": lab_r, "st_r": st_r, "cen_r": cen_r, "hallados": hallados,
    }


def montaje(sc: dict, r: dict, out: Path, n: int = 6) -> Path:
    """Recortes (SWIR arriba, Wildfire abajo): focos hallados vs perdidos, para juzgarlos a ojo."""
    swir, wf = sc["swir"][..., ::-1], sc["wildfire"][..., ::-1]
    st, cen = r["st_r"], r["cen_r"]
    perdidos = [i for i, _ in r["perdidos"]]

    def fila(ids, titulo, color):
        ids = sorted(ids, key=lambda i: -st[i, cv2.CC_STAT_AREA])[:n]
        h = 70
        tiles = []
        for i in ids:
            x, y = int(cen[i][0]), int(cen[i][1])
            x0 = min(max(x - h, 0), swir.shape[1] - 2 * h)
            y0 = min(max(y - h, 0), swir.shape[0] - 2 * h)
            t = np.vstack([cv2.resize(im[y0:y0 + 2 * h, x0:x0 + 2 * h], (210, 210), interpolation=cv2.INTER_NEAREST) for im in (swir, wf)])
            cv2.putText(t, f"{st[i, cv2.CC_STAT_AREA]}px", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(t)
        if not tiles:
            return np.zeros((26, 210, 3), np.uint8)
        banda = np.zeros((26, 210 * len(tiles), 3), np.uint8)
        cv2.putText(banda, titulo, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return np.vstack([banda, np.hstack(tiles)])

    a, b = fila(r["hallados"], "HALLADOS (arriba SWIR, abajo Wildfire)", (0, 255, 0)), fila(perdidos, "PERDIDOS", (0, 0, 255))
    w = max(a.shape[1], b.shape[1])
    pad = lambda x: np.pad(x, ((0, 0), (0, w - x.shape[1]), (0, 0)))  # noqa: E731
    out.mkdir(parents=True, exist_ok=True)
    f = out / "wildfire_montaje.png"
    cv2.imwrite(str(f), np.vstack([pad(a), pad(b)]))
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("escenas", nargs="*", default=["data/niebla", "data/extra_mierla"])
    ap.add_argument("--montaje", action="store_true", help="guarda debug/<escena>/wildfire_montaje.png")
    a = ap.parse_args()
    for src in a.escenas:
        sc = load_scene(src)
        if sc["wildfire"] is None:
            print(f"{sc['name']}: no tiene Wildfire, nada que comparar")
            continue
        r = comparar(sc)
        print(f"== {sc['name']}")
        print(f"   objetos de referencia (Wildfire): {r['ref']}  | detectados por nosotros: {r['det']}")
        print(f"   precisión {r['precision']:.2f} ({r['det_confirmados']}/{r['det']} de lo que marcamos coincide con la Wildfire)")
        print(f"   recall    {r['recall']:.2f} ({len(r['hallados'])}/{r['ref']} objetos de la Wildfire hallados)")
        causas = {c: sum(1 for _, x in r["perdidos"] if x == c) for c in CAUSAS}
        for c, nn in causas.items():
            if nn:
                print(f"      perdidos por «{c}»: {nn}")
        print(f"   recall sin los que son pálidos en la SWIR: {r['recall_sin_palidos']:.2f}  "
              f"(estimación: suponemos que los pálidos NO son fuego; se comprueba con el montaje, no es una medida)")
        if a.montaje:
            print("   montaje ->", montaje(sc, r, Path("debug") / sc["name"]))

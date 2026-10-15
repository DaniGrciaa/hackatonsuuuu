"""T3 - Máscara de área quemada y perímetro.

Variantes (cada una añade un filtro a la anterior, para poder medirlas por separado):
  v0  dos pasos: fuera el agua -> Otsu del brillo (lo oscuro) -> Otsu del NBR (lo bajo) -> morfología
  v1  v0 + tono rojizo (Otsu de RED_DOM entre lo oscuro) y fuera tierra clara / nubes
  v2  v1 + prior espacial: se queda la mancha grande y lo que está pegado al humo o al fuego

    python core/segmentation.py data/niebla     # PNG en debug/<escena>/seg_*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import threshold_otsu

VARIANTS = ("v0", "v1", "v2")
DEFAULT_VARIANT = "v2"
KEEP_FRAC = 0.25        # v2: se conservan manchas con al menos esta fracción del área de la mayor
NEAR_FRAC = 0.03        # v2: "pegado al humo/fuego" = a menos de esta fracción de la diagonal


def _ellipse(k: int) -> np.ndarray:
    k = max(3, int(k) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _otsu(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(threshold_otsu(x[:: max(1, x.size // 500_000)]))


def _components(m: np.ndarray):
    n, lab, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    return n, lab, st, cen


def fire_like(ind: dict, b: dict) -> np.ndarray:
    """Candidatos a fuego activo en la SWIR: B12 casi saturado y claramente por encima del resto (naranja/rojo vivo)."""
    b12 = b["B12"]
    return (b12 > np.percentile(b12[b["valid"]], 99.5)) & (ind["RED_DOM"] > 0.25) & b["valid"]


def burn_details(b: dict, ind: dict, masks: dict, variant: str = DEFAULT_VARIANT) -> dict:
    """Devuelve {mask, variant, steps (máscaras intermedias), discarded (componentes quitados), thresholds}."""
    valid = b["valid"]
    d = float(np.hypot(*valid.shape))
    k_open, k_close = max(3, int(d / 900)), max(5, int(d / 450))
    min_area = int(0.0009 * valid.size)            # ~4000 px en 2500 px de ancho
    steps, th = {}, {}

    # --- v0 ---
    cand = valid & ~masks["water"]
    th["bright"] = _otsu(ind["BRIGHT"][cand])
    dark = cand & (ind["BRIGHT"] < th["bright"])
    th["nbr"] = _otsu(ind["NBR"][dark])
    m = dark & (ind["NBR"] < th["nbr"])
    steps["dark"], steps["v0_raw"] = dark, m.copy()

    if variant in ("v1", "v2"):
        th["red_dom"] = _otsu(ind["RED_DOM"][m])
        m &= ind["RED_DOM"] > th["red_dom"]
        m &= ~masks["bare"] & ~masks["cloud"]
        steps["v1_raw"] = m.copy()

    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, _ellipse(k_open))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse(k_close)).astype(bool)

    discarded = []
    n, lab, st, cen = _components(m)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    for i in range(1, n):
        if not keep[i]:
            discarded.append({"reason": "pequeño", "area": int(st[i, 4]), "centroid": [float(cen[i][0]), float(cen[i][1])]})

    if variant == "v2" and keep.any():
        areas = np.where(keep, st[:, cv2.CC_STAT_AREA], 0)
        big = areas.max()
        near = np.zeros(valid.shape, np.uint8)
        near[masks["smoke"] | fire_like(ind, b)] = 1
        near = cv2.dilate(near, _ellipse(int(NEAR_FRAC * d) * 2 + 1)).astype(bool)
        for i in range(1, n):
            if keep[i] and areas[i] < KEEP_FRAC * big and not near[lab == i].any():
                keep[i] = False
                discarded.append({"reason": "suelto, lejos del humo/fuego", "area": int(st[i, 4]),
                                  "centroid": [float(cen[i][0]), float(cen[i][1])]})
    m = keep[lab] if n > 1 else np.zeros_like(m)

    # Huecos: los pequeños se rellenan (ceniza clara, claros); el agua grande de dentro se vuelve a restar
    inv = (~m).astype(np.uint8)
    n2, lab2, st2, _ = _components(inv)
    for i in range(1, n2):
        touches = st2[i, 0] == 0 or st2[i, 1] == 0 or st2[i, 0] + st2[i, 2] == m.shape[1] or st2[i, 1] + st2[i, 3] == m.shape[0]
        if not touches and st2[i, 4] < 4 * min_area:
            m[lab2 == i] = True
    m &= ~masks["water"] & valid
    return {"mask": m, "variant": variant, "steps": steps, "discarded": discarded, "thresholds": th}


def burn_mask(ind: dict, masks: dict, b: dict | None = None, variant: str = DEFAULT_VARIANT) -> np.ndarray:
    """Interfaz acordada: bool HxW. 'b' (bandas) es opcional pero necesario para v2 (fuego)."""
    if b is None:
        raise ValueError("burn_mask necesita las bandas (b=...) para localizar el fuego activo y la validez")
    return burn_details(b, ind, masks, variant)["mask"]


def perimeter(mask: np.ndarray, scale: float = 1.0) -> list[dict]:
    """Polígonos [{'exterior': [[x,y],...], 'holes': [[[x,y],...], ...]}] con coordenadas / scale
    (scale = trabajo/original, así salen en píxeles de la imagen original)."""
    cnts, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    eps = 0.0007 * float(np.hypot(*mask.shape))
    def poly(c):
        return (cv2.approxPolyDP(c, eps, True).reshape(-1, 2) / scale).round(1).tolist()
    out = []
    for i, c in enumerate(cnts):
        if hier[0][i][3] == -1 and len(c) >= 3:
            holes = [poly(cnts[j]) for j in range(len(cnts)) if hier[0][j][3] == i and len(cnts[j]) >= 3]
            out.append({"exterior": poly(c), "holes": holes})
    return out


def draw_variants(scene: dict, results: dict, root: str = "debug") -> str:
    out = Path(root) / scene["name"]
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for v, r in results.items():
        im = scene["swir"][..., ::-1].copy()
        cnts, _ = cv2.findContours(r["mask"].astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cnts, -1, (0, 255, 255), max(2, im.shape[1] // 900))
        pct = 100 * r["mask"].sum() / scene["valid"].sum()
        cv2.putText(im, f"{v}: {pct:.1f}%", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, im.shape[1] / 1300, (255, 255, 255), 4)
        tiles.append(cv2.resize(im, (1300, int(im.shape[0] * 1300 / im.shape[1]))))
        cv2.imwrite(str(out / f"seg_{v}.png"), r["mask"].astype(np.uint8) * 255)
    cv2.imwrite(str(out / "seg_variants.png"), np.vstack(tiles))
    return str(out)


if __name__ == "__main__":
    from core.ingestion import load_scene
    from core.spectral import bands, exclusion_masks, indices

    for src in sys.argv[1:] or ["data"]:
        sc = load_scene(src)
        b = bands(sc)
        ind = indices(b)
        ms = exclusion_masks(b, ind)
        res = {v: burn_details(b, ind, ms, v) for v in VARIANTS}
        print(sc["name"])
        for v, r in res.items():
            print(f"  {v}: {100 * r['mask'].sum() / sc['valid'].sum():5.1f}% quemado, "
                  f"{len(perimeter(r['mask']))} manchas, {len(r['discarded'])} descartadas")
        print("  umbrales:", {k: round(x, 3) for k, x in res['v2']['thresholds'].items()})
        print("  debug ->", draw_variants(sc, res))

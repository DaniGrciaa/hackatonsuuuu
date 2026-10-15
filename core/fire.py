"""Focos de fuego activo (nivel avanzado del reto).

Idea (la de los algoritmos de satélite tipo MODIS/VIIRS, adaptada a la imagen SWIR de Copernicus,
R=B12, G=B8A, B=B04): un píxel es foco si
  1. está entre los más calientes de la escena en B12 (percentil 99,5 del canal rojo de la SWIR, sin recortar),
  2. es claramente naranja/rojo: (R - max(G, B)) / R >= 0,40 (descarta suelo claro, mina y nube, que son pálidos,
     y campos pardos, que tienen más verde: G~160 frente a 106-139 en los focos reales),
  3. destaca sobre su entorno: R - media_fondo > 3 * desviación_fondo en una ventana de 61 px, calculando el fondo
     SIN los píxeles calientes (si no, un frente largo se compara consigo mismo y no destaca).

Medido (por focos, con 5 px de tolerancia; ver README): Niebla vs Wildfire P 0,75 R 0,54 | Extra vs Wildfire P 0,64 R 0,88 | Extra vs marcas a mano P 1,00 R 0,82 |
práctica 26-07 vs regla física QuickFire (B11+B12>1) P 0,82 R 0,29 | escenas sin incendio: 0, 1, 1 y 9 px.

    python -m core.fire data/niebla      # debug/<escena>/fire_*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HOT_PCT = 99.5      # percentil de B12 para ser candidato
RED_RATIO = 0.30    # (R - max(G, B)) / R mínimo: naranja/rojo vivo, no pardo ni pálido (ver README: barrido)
Z_CTX = 2.5         # cuántas desviaciones por encima del fondo local
WIN = 61            # ventana del fondo (px a resolución de trabajo)
NEAR_SCAR_PX = 15   # foco "en el frente" si está a menos de esto de la cicatriz


def _swir_raw(b: dict) -> np.ndarray:
    sw = b.get("_swir")
    if sw is None:
        raise ValueError("detect_fires necesita las bandas de core.spectral.bands (clave interna '_swir')")
    return sw.astype(np.float32)


def fire_mask(b: dict) -> np.ndarray:
    sw = _swir_raw(b)
    valid = b["valid"]
    if not valid.any():
        return np.zeros(valid.shape, bool)
    R, G, B = sw[..., 0], sw[..., 1], sw[..., 2]
    red = (R - np.maximum(G, B)) / (R + 1e-6)
    hot = (R >= np.percentile(R[valid], HOT_PCT)) & valid
    cand = hot & (red >= RED_RATIO)
    if not cand.any():
        return np.zeros(valid.shape, bool)
    bg = (valid & ~cv2.dilate(hot.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)).astype(np.float32)
    k = (WIN, WIN)
    support = cv2.blur(bg, k)
    n = support + 1e-6
    mu = cv2.blur(R * bg, k) / n
    sd = np.sqrt(np.clip(cv2.blur(R * R * bg, k) / n - mu * mu, 0, None))
    return cand & (support > 0) & ((R - mu) > Z_CTX * np.maximum(sd, 1.0))


def detect_fires(b: dict, ind: dict, burn: np.ndarray, masks: dict) -> np.ndarray:
    """Interfaz acordada: bool HxW con los píxeles de fuego activo."""
    return fire_mask(b)


def fire_details(fires: np.ndarray, burn: np.ndarray, masks: dict, scale: float = 1.0) -> list[dict]:
    """Lista de focos: centroide (px de la imagen original), área, y si está en el frente de la cicatriz,
    bajo/junto al humo o aislado (aislado = menos confianza: puede ser un foco nuevo o un falso positivo)."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(fires.astype(np.uint8), connectivity=8)
    if n <= 1:
        return []
    dist = cv2.distanceTransform((~burn).astype(np.uint8), cv2.DIST_L2, 5)
    smoke = cv2.dilate(masks["smoke"].astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    out = []
    for i in range(1, n):
        c = lab == i
        d = float(dist[c].min())
        if d <= NEAR_SCAR_PX:
            where, conf = "frente de la cicatriz", "alta"
        elif smoke[c].any():
            where, conf = "junto al humo", "media"
        else:
            where, conf = "aislado", "baja"
        out.append({"centro": [round(cen[i][0] / scale, 1), round(cen[i][1] / scale, 1)],
                    "area_px": int(st[i, cv2.CC_STAT_AREA]), "dist_cicatriz_px": round(d, 1),
                    "ubicacion": where, "confianza": conf})
    return out


if __name__ == "__main__":
    from core.ingestion import load_scene
    from core.segmentation import burn_mask
    from core.spectral import bands, exclusion_masks, indices

    for src in sys.argv[1:] or ["data"]:
        sc = load_scene(src)
        b = bands(sc)
        ind = indices(b)
        ms = exclusion_masks(b, ind)
        burn = burn_mask(ind, ms)
        f = detect_fires(b, ind, burn, ms)
        det = fire_details(f, burn, ms, sc["scale"])
        print(f"{sc['name']}: {int(f.sum())} px de fuego, {len(det)} focos "
              + str({k: sum(d['ubicacion'] == k for d in det) for k in ('frente de la cicatriz', 'junto al humo', 'aislado')}))
        out = Path("debug") / sc["name"]
        out.mkdir(parents=True, exist_ok=True)
        im = sc["swir"][..., ::-1].copy()
        for d in det:
            x, y = int(d["centro"][0] * sc["scale"]), int(d["centro"][1] * sc["scale"])
            cv2.circle(im, (x, y), 12, (255, 255, 255) if d["confianza"] == "alta" else (0, 255, 255), 2)
        cv2.imwrite(str(out / "fire_focos.png"), im)

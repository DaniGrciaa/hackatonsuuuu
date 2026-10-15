"""T2 - Bandas, normalización e índices.

SWIR (Copernicus): R=B12, G=B8A, B=B04.   True Color: R=B04, G=B03, B=B02.
Todo se normaliza por percentiles de cada escena y banda (nada de umbrales sobre 0..255).

    python core/spectral.py data/niebla     # PNG de depuración en debug/<escena>/spec_*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

P_LO, P_HI = 1.0, 99.0
EPS = 1e-6


def pnorm(x: np.ndarray, valid: np.ndarray | None = None, lo: float = P_LO, hi: float = P_HI) -> np.ndarray:
    """Lleva x a 0..1 usando los percentiles lo/hi de los píxeles válidos."""
    v = x[valid] if valid is not None and valid.any() else x
    a, b = np.percentile(v[:: max(1, v.size // 400_000)], [lo, hi])
    return np.clip((x - a) / max(b - a, EPS), 0, 1).astype(np.float32)


def calibrate(tc: np.ndarray, swir: np.ndarray, valid: np.ndarray) -> dict:
    """Recta robusta azul_SWIR ≈ a * rojo_TC + b (ambos son B04), sobre píxeles válidos.
    Sirve para comprobar que las dos fotos están alineadas y para pasar una a la escala de la otra."""
    step = max(1, int(valid.sum() // 200_000))
    x = tc[..., 0][valid][::step].astype(np.float32)
    y = swir[..., 2][valid][::step].astype(np.float32)
    lo, hi = np.percentile(x, [2, 98])
    m = (x > lo) & (x < hi)
    a, b = np.polyfit(x[m], y[m], 1)
    return {"a": float(a), "b": float(b), "corr": float(np.corrcoef(x, y)[0, 1])}


def bands(scene: dict) -> dict:
    """Devuelve B02, B03, B04, B8A, B12 (float32, 0..1). Las que no se puedan obtener, ausentes.
    B12/B8A/B04 salen de la SWIR; B02/B03 solo existen si hay True Color.
    Claves extra: 'valid' (máscara), 'B04_tc' (B04 desde la TC), 'calib'."""
    valid = scene["valid"]
    swir, tc = scene["swir"], scene["tc"]
    out = {
        "B12": pnorm(swir[..., 0].astype(np.float32), valid),
        "B8A": pnorm(swir[..., 1].astype(np.float32), valid),
        "B04": pnorm(swir[..., 2].astype(np.float32), valid),
        "valid": valid,
    }
    if tc is not None:
        out["B04_tc"] = pnorm(tc[..., 0].astype(np.float32), valid)
        out["B03"] = pnorm(tc[..., 1].astype(np.float32), valid)
        out["B02"] = pnorm(tc[..., 2].astype(np.float32), valid)
        out["calib"] = calibrate(tc, swir, valid)
    return out


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - b) / (a + b + EPS)).astype(np.float32)


def indices(b: dict) -> dict:
    """NBR, NDVI, WATER, BRIGHT (+ RED_DOM y SAT, que usa la segmentación).
    NBR  = (B8A - B12)/(B8A + B12): bajo en lo quemado.
    NDVI = (B8A - B04)/(B8A + B04): vegetación sana.
    WATER  = 1 - máximo de los tres canales SWIR: alto en el agua (negra).
    BRIGHT = media de los tres canales SWIR.
    RED_DOM = B12 - max(B8A, B04): positivo si domina el rojo (cicatriz marrón rojiza)."""
    b12, b8a, b04 = b["B12"], b["B8A"], b["B04"]
    mx = np.maximum(np.maximum(b12, b8a), b04)
    mn = np.minimum(np.minimum(b12, b8a), b04)
    return {
        "NBR": _nd(b8a, b12),
        "NDVI": _nd(b8a, b04),
        "WATER": (1 - mx).astype(np.float32),
        "BRIGHT": ((b12 + b8a + b04) / 3).astype(np.float32),
        "RED_DOM": (b12 - np.maximum(b8a, b04)).astype(np.float32),
        "SAT": ((mx - mn) / (mx + EPS)).astype(np.float32),
    }


def _gray_png(x: np.ndarray, path: Path, lo=1, hi=99):
    a, b = np.percentile(x, [lo, hi])
    cv2.imwrite(str(path), (np.clip((x - a) / max(b - a, EPS), 0, 1) * 255).astype(np.uint8))


def save_debug(scene: dict, b: dict, ind: dict, root: str = "debug") -> str:
    out = Path(root) / scene["name"]
    out.mkdir(parents=True, exist_ok=True)
    for k, v in ind.items():
        cm = cv2.applyColorMap((np.clip((v - np.percentile(v, 1)) / max(np.ptp(np.percentile(v, [1, 99])), EPS), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(str(out / f"spec_{k}.png"), cm)
    return str(out)


if __name__ == "__main__":
    from core.ingestion import load_scene

    for src in sys.argv[1:] or ["data"]:
        sc = load_scene(src)
        b = bands(sc)
        ind = indices(b)
        print(sc["name"], "bandas:", [k for k in b if k.startswith("B")])
        if "calib" in b:
            c = b["calib"]
            print(f"  azul SWIR ≈ {c['a']:.2f} x rojo TC {c['b']:+.0f}  (corr {c['corr']:.3f})")
            # Hipótesis: la misma recta lleva el verde y el rojo de la SWIR a la escala de la TC
            tc, sw = sc["tc"].astype(float), sc["swir"].astype(float)
            v = sc["valid"]
            for i, n in ((0, "B12 (R)"), (1, "B8A (G)")):
                pred = (sw[..., i] - c["b"]) / c["a"]
                print(f"  SWIR {n} -> escala TC: mediana {np.median(pred[v]):.0f} (TC R/G/B medianas "
                      f"{np.median(tc[..., 0][v]):.0f}/{np.median(tc[..., 1][v]):.0f}/{np.median(tc[..., 2][v]):.0f})")
        for k, v in ind.items():
            print(f"  {k:8s} p1/p50/p99 = {np.percentile(v[sc['valid']], [1, 50, 99]).round(2)}")
        print("  debug ->", save_debug(sc, b, ind))

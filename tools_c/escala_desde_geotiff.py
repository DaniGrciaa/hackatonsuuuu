"""Mide los metros por píxel de una JPG sin georreferencia alineándola con un GeoTIFF georreferenciado de la misma zona.

    python tools_c/escala_desde_geotiff.py <True_color georreferenciado.tiff> <carpeta de la escena JPG> [--guardar]

Empareja puntos característicos (SIFT + RANSAC, transformación afín) entre las dos True Color. La escala del GeoTIFF
(ModelPixelScaleTag) dividida por la escala de la transformación da los metros por píxel de la JPG original.
Con --guardar escribe <carpeta>/escala.json (lo leen run.py y la app).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile


def main(tiff: str, folder: str, save: bool) -> None:
    with tifffile.TiffFile(tiff) as t:
        sx_m, sy_m = t.pages[0].tags["ModelPixelScaleTag"].value[:2]
        geo = t.pages[0].asarray()
    geo = (geo / 257).astype(np.uint8) if geo.dtype == np.uint16 else geo
    geo = cv2.cvtColor(geo[..., :3], cv2.COLOR_RGB2GRAY)
    jpg = next(p for p in Path(folder).iterdir() if "true" in p.name.lower())
    img = cv2.imread(str(jpg), cv2.IMREAD_GRAYSCALE)
    f = min(1.0, 3300 / img.shape[1])
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(8000)
    k1, d1 = sift.detectAndCompute(geo, None)
    k2, d2 = sift.detectAndCompute(small, None)
    good = [a for a, b in cv2.BFMatcher().knnMatch(d1, d2, k=2) if a.distance < 0.75 * b.distance]
    src = np.float32([k1[g.queryIdx].pt for g in good])
    dst = np.float32([k2[g.trainIdx].pt for g in good])
    A, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=20000, confidence=0.999)
    if A is None or inl.sum() < 50:
        raise SystemExit("No se pudo alinear (pocos puntos coherentes): ¿son la misma zona?")
    ax, ay = np.hypot(A[0, 0], A[1, 0]), np.hypot(A[0, 1], A[1, 1])
    mx, my = sx_m / ax * f, sy_m / ay * f
    res = np.linalg.norm(src[inl.ravel() == 1] @ A[:, :2].T + A[:, 2] - dst[inl.ravel() == 1], axis=1)
    print(f"{int(inl.sum())} puntos coherentes, error medio {res.mean():.2f} px, giro {np.degrees(np.arctan2(A[1, 0], A[0, 0])):.2f}°")
    print(f"{jpg.name}: {mx:.2f} x {my:.2f} m/px  ->  {mx * my / 1e4:.5f} ha por píxel")
    if save:
        out = {"m_por_px_x": round(float(mx), 3), "m_por_px_y": round(float(my), 3),
               "fuente": f"alineación SIFT+RANSAC con {Path(tiff).name} (UTM, {sx_m:.2f} x {sy_m:.2f} m/px); "
                         f"{int(inl.sum())} puntos, error medio {res.mean():.2f} px"}
        (Path(folder) / "escala.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print("guardado", Path(folder) / "escala.json")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        raise SystemExit(__doc__)
    main(args[0], args[1], "--guardar" in sys.argv)

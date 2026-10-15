"""Visor interactivo (viewer.html) incrustado en Streamlit.

    from components.viewer import build_payload, render
    render(build_payload(analysis, title="niebla"))

`analysis` tiene la forma que devuelve pipeline.run_all_layers (o la que carga un ejemplo precalculado):
{"scene": {...}, "bases": {"tc", "swir", "wildfire": RGB uint8 | None}, "layers": {id: {status, mask, metrics, ...}}}.
Todo lo interactivo ocurre dentro del visor: Streamlit no se recarga al cambiar capas, opacidad o modo.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from layers import LAYERS_BY_ID

HTML_PATH = Path(__file__).with_name("viewer.html")
MAX_SIDE = 2048
BASE_NAMES = {"tc": "True Color", "swir": "SWIR", "wildfire": "Wildfires"}


def viewer_size(h: int, w: int, max_side: int = MAX_SIDE) -> tuple[int, int]:
    s = min(1.0, max_side / max(h, w))
    return max(1, int(round(h * s))), max(1, int(round(w * s)))


def jpeg_b64(rgb: np.ndarray, size: tuple[int, int] | None = None, quality: int = 84) -> str:
    if size is not None and rgb.shape[:2] != size:
        rgb = cv2.resize(rgb, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def mask_png_bytes(mask: np.ndarray) -> bytes:
    """PNG de 1 bit (blanco = capa)."""
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(mask.astype(bool))).save(buf, "PNG", optimize=True)
    return buf.getvalue()


def resize_mask(mask: np.ndarray, size: tuple[int, int], keep_small: bool = False) -> np.ndarray:
    """Reescala una máscara. keep_small: un píxel marcado basta (para que no desaparezcan focos de 1-2 px)."""
    if mask.shape == size:
        return mask.astype(bool)
    area = cv2.resize(mask.astype(np.float32), (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return area > (0.0 if keep_small else 0.5)


def contours(mask: np.ndarray) -> list[list[int]]:
    """Bordes exteriores y huecos como listas planas [x0, y0, x1, y1, ...] en píxeles del visor."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    eps = max(0.6, 0.0004 * float(np.hypot(*mask.shape)))
    out = []
    for c in cnts:
        if len(c) < 3:
            continue
        poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(poly) >= 3:
            out.append(poly.ravel().astype(int).tolist())
    return out


def points(mask: np.ndarray) -> list[list[float]]:
    """Centro de cada elemento (componente conexa) en píxeles del visor."""
    n, _, _, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [[round(float(x), 1), round(float(y), 1)] for x, y in cen[1:n]]


def build_payload(analysis: dict, title: str = "", max_side: int = MAX_SIDE) -> dict:
    swir = analysis["bases"]["swir"]
    ah, aw = swir.shape[:2]
    vh, vw = viewer_size(ah, aw, max_side)
    bases = [{"id": k, "nombre": BASE_NAMES[k], "src": jpeg_b64(analysis["bases"][k], (vh, vw))}
             for k in ("tc", "swir", "wildfire") if analysis["bases"].get(k) is not None]

    capas = []
    for lid, res in analysis["layers"].items():
        layer = LAYERS_BY_ID.get(lid)
        if layer is None:
            continue
        item = {"id": lid, "nombre": layer.nombre, "descripcion": layer.descripcion, "color": layer.color,
                "tipo": layer.tipo, "patron": layer.patron, "visible": layer.visible,
                "estado": res["status"], "error": res.get("error"), "tiempo_s": res.get("time_s", 0.0),
                "metricas": res.get("metrics")}
        if res["status"] == "ok" and res.get("mask") is not None:
            m = resize_mask(res["mask"], (vh, vw), keep_small=layer.tipo == "puntos")
            item["mascara"] = "data:image/png;base64," + base64.b64encode(mask_png_bytes(m)).decode()
            if layer.tipo == "contorno":
                item["contornos"] = contours(m)
            if layer.tipo == "puntos":
                item["puntos"] = points(m)
        if "focos_por_ubicacion" in res:
            item["focos_por_ubicacion"] = res["focos_por_ubicacion"]
        capas.append(item)

    return {
        "titulo": title or analysis.get("scene", {}).get("name", ""),
        "ancho": vw, "alto": vh,
        "tamano_analisis": [aw, ah],
        "factor_analisis": aw / vw,
        "bases": bases,
        "capas": capas,
        "avisos": analysis.get("scene", {}).get("warnings", []),
    }


def to_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return HTML_PATH.read_text(encoding="utf-8").replace("__PAYLOAD__", data, 1)


def render(payload: dict, height: int = 820) -> None:
    import streamlit.components.v1 as components
    components.html(to_html(payload), height=height, scrolling=False)

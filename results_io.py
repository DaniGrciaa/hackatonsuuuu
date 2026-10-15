"""Guardar y cargar un análisis (escenas de ejemplo y descarga en zip).

Carpeta de un análisis guardado (examples/<escena>/):
    tc.jpg, swir.jpg, wildfire.jpg   imágenes base comprimidas (lado máximo 2048 px)
    thumb.jpg                        miniatura para la tarjeta de la app
    masks/<capa>.png                 máscara de 1 bit a la resolución de análisis
    metrics.json                     cifras por capa en píxeles, estado, tiempo y error
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from components.viewer import mask_png_bytes, viewer_size
from layers import LAYERS_BY_ID

BASE_KINDS = ("tc", "swir", "wildfire")
MONTHS = "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split()
UNITS_NOTE = "Píxeles de la imagen analizada. La imagen no está georreferenciada: no hay hectáreas ni km."


def date_from_names(names) -> str | None:
    """Fecha legible a partir de nombres tipo 2026-07-26-..._Sentinel-2_L2A_SWIR.jpg."""
    for n in names:
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(n))
        if m:
            y, mo, d = map(int, m.groups())
            if 1 <= mo <= 12:
                return f"{d} de {MONTHS[mo - 1]} de {y}"
    return None


def metrics_dict(analysis: dict, meta: dict | None = None) -> dict:
    sc = analysis["scene"]
    capas = {}
    for lid, r in analysis["layers"].items():
        layer = LAYERS_BY_ID.get(lid)
        item = {"nombre": layer.nombre if layer else lid, "estado": r["status"], "error": r.get("error"),
                "tiempo_s": r.get("time_s"), "metricas": r.get("metrics"),
                "mascara": f"masks/{lid}.png" if r.get("mask") is not None else None}
        if "focos_por_ubicacion" in r:
            item["focos_por_ubicacion"] = r["focos_por_ubicacion"]
        capas[lid] = item
    return {
        **(meta or {}),
        "escena": sc["name"],
        "tamano_analisis": [sc["size"][1], sc["size"][0]],
        "tamano_original": [sc["orig_size"][1], sc["orig_size"][0]],
        "unidades": UNITS_NOTE,
        "tiempo_total_s": analysis.get("time_s"),
        "avisos": sc.get("warnings", []),
        "capas": capas,
    }


def _jpeg(rgb: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=quality, optimize=True, progressive=True)
    return buf.getvalue()


def save_analysis(analysis: dict, out_dir: str | Path, meta: dict | None = None, quality: int = 84) -> Path:
    out = Path(out_dir)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    for old in (out / "masks").glob("*.png"):
        old.unlink()
    h, w = analysis["bases"]["swir"].shape[:2]
    vh, vw = viewer_size(h, w)
    for k in BASE_KINDS:
        im = analysis["bases"].get(k)
        path = out / f"{k}.jpg"
        if im is None:
            path.unlink(missing_ok=True)
            continue
        path.write_bytes(_jpeg(cv2.resize(im, (vw, vh), interpolation=cv2.INTER_AREA), quality))
    thumb_src = analysis["bases"].get("tc")
    thumb_src = thumb_src if thumb_src is not None else analysis["bases"]["swir"]
    th, tw = viewer_size(h, w, 640)
    (out / "thumb.jpg").write_bytes(_jpeg(cv2.resize(thumb_src, (tw, th), interpolation=cv2.INTER_AREA), 80))
    for lid, r in analysis["layers"].items():
        if r.get("mask") is not None:
            (out / "masks" / f"{lid}.png").write_bytes(mask_png_bytes(r["mask"]))
    (out / "metrics.json").write_text(json.dumps(metrics_dict(analysis, meta), ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    return out


def load_analysis(folder: str | Path) -> dict:
    """Devuelve la misma forma que pipeline.run_all_layers (bases a <= 2048 px, máscaras a resolución de análisis)."""
    folder = Path(folder)
    m = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    bases = {}
    for k in BASE_KINDS:
        p = folder / f"{k}.jpg"
        bases[k] = np.asarray(Image.open(p).convert("RGB")) if p.exists() else None
    layers = {}
    for lid, c in m["capas"].items():
        mask = None
        if c.get("mascara") and (folder / c["mascara"]).exists():
            mask = np.asarray(Image.open(folder / c["mascara"]).convert("L")) > 127
        r = {"status": c["estado"], "mask": mask, "metrics": c.get("metricas"), "time_s": c.get("tiempo_s"),
             "error": c.get("error")}
        if "focos_por_ubicacion" in c:
            r["focos_por_ubicacion"] = c["focos_por_ubicacion"]
        layers[lid] = r
    w, h = m["tamano_analisis"]
    ow, oh = m["tamano_original"]
    return {"scene": {"name": m["escena"], "size": [h, w], "orig_size": [oh, ow], "warnings": m.get("avisos", [])},
            "bases": bases, "layers": layers, "time_s": m.get("tiempo_total_s"), "meta": m}


def zip_bytes(analysis: dict, meta: dict | None = None) -> bytes:
    """Zip con masks/<capa>.png (1 bit, resolución de análisis) y metrics.json."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for lid, r in analysis["layers"].items():
            if r.get("mask") is not None:
                z.writestr(f"masks/{lid}.png", mask_png_bytes(r["mask"]))
        z.writestr("metrics.json", json.dumps(metrics_dict(analysis, meta), ensure_ascii=False, indent=1))
    return buf.getvalue()

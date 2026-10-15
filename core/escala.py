"""Escala del terreno (metros por píxel) de una escena, si se conoce.

Las JPG de Copernicus Browser no llevan escala ni coordenadas. Si se conoce, se guarda junto a las imágenes en
<carpeta de la escena>/escala.json:
    {"m_por_px_x": 12.38, "m_por_px_y": 12.32, "fuente": "cómo se obtuvo"}
en metros por píxel de la imagen ORIGINAL. Sin ese archivo todo se da en píxeles: no se inventa una escala.
"""
from __future__ import annotations

import json
from pathlib import Path

FILE = "escala.json"


def scale_for(src) -> dict | None:
    """src: carpeta de la escena o lista de archivos (se mira la carpeta del primero)."""
    if isinstance(src, (list, tuple)):
        if not src:
            return None
        src = Path(src[0]).parent
    p = Path(src)
    p = p if p.is_dir() else p.parent
    f = p / FILE
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    mx, my = float(d["m_por_px_x"]), float(d["m_por_px_y"])
    if not (0 < mx < 10_000 and 0 < my < 10_000):
        raise ValueError(f"{f}: escala fuera de rango")
    return {"m_por_px_x": mx, "m_por_px_y": my, "fuente": d.get("fuente", "")}


def ha_per_px(scale: dict, work_scale: float = 1.0) -> float:
    """Hectáreas que mide un píxel de la imagen de TRABAJO (reducida por work_scale = trabajo/original)."""
    return scale["m_por_px_x"] * scale["m_por_px_y"] / 1e4 / work_scale ** 2


def km_per_px(scale: dict, work_scale: float = 1.0) -> float:
    """Kilómetros por píxel de la imagen de trabajo (media de los dos ejes: los bordes van en todas direcciones)."""
    return (scale["m_por_px_x"] + scale["m_por_px_y"]) / 2 / 1000 / work_scale

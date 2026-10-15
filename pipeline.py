"""Ejecuta todas las capas del visor sobre una escena (reutiliza core/ sin cambiar sus algoritmos).

    from pipeline import run_all_layers
    res = run_all_layers(true_color_rgb, swir_rgb, wildfires_rgb)   # arrays RGB uint8; TC y Wildfires pueden ser None
    res["layers"]["burn"] -> {"status", "mask", "metrics", "time_s", "error", ...}

Todas las cifras van en píxeles de la imagen analizada: sin georreferenciación no hay hectáreas ni km.
"""
from __future__ import annotations

import importlib.util
import time
from typing import Callable

import cv2
import numpy as np

from core.ingestion import scene_from_arrays
from layers import LAYERS, Layer

STAGE_NAMES = {
    "bands": "bandas de la imagen",
    "indices": "índices espectrales",
    "masks": "máscaras de agua, humo, nubes y suelo",
    "burn": "zona quemada",
    "fires": "focos de fuego activo",
}


class StageError(RuntimeError):
    """Un paso común falló; las capas que dependen de él lo heredan como error."""


class Context:
    """Calcula bajo demanda (y una sola vez) los pasos comunes que comparten las capas.
    Si un paso falla se recuerda el error y cada capa que lo pida falla con el mismo mensaje."""

    def __init__(self, scene: dict):
        self.scene = scene
        self._cache: dict[str, tuple[object, str | None]] = {}

    def _stage(self, key: str, fn: Callable[[], object]):
        if key not in self._cache:
            try:
                self._cache[key] = (fn(), None)
            except StageError as e:
                self._cache[key] = (None, str(e))
            except Exception as e:  # noqa: BLE001 - se informa a la capa, no se propaga
                self._cache[key] = (None, f"No se pudieron calcular las {STAGE_NAMES[key]}: {e}"
                                    if key in ("bands", "indices", "masks") else
                                    f"No se pudo calcular la capa base «{STAGE_NAMES[key]}»: {e}")
        value, err = self._cache[key]
        if err is not None:
            raise StageError(err)
        return value

    @property
    def bands(self) -> dict:
        from core.spectral import bands
        return self._stage("bands", lambda: bands(self.scene))

    @property
    def indices(self) -> dict:
        from core.spectral import indices
        return self._stage("indices", lambda: indices(self.bands))

    @property
    def masks(self) -> dict:
        from core.spectral import exclusion_masks
        return self._stage("masks", lambda: exclusion_masks(self.bands, self.indices))

    @property
    def burn(self) -> np.ndarray:
        return self.burn_details["mask"]

    @property
    def burn_details(self) -> dict:
        from core.segmentation import burn_details
        return self._stage("burn", lambda: burn_details(self.bands, self.indices, self.masks))

    @property
    def fires(self) -> np.ndarray:
        from core.fire import fire_mask
        return self._stage("fires", lambda: fire_mask(self.bands))


def mask_metrics(mask: np.ndarray) -> dict:
    """Área, % de la imagen, número de regiones y perímetro (borde exterior + huecos), todo en píxeles."""
    m = mask.astype(np.uint8)
    area = int(m.sum())
    n, _ = cv2.connectedComponents(m, connectivity=8)
    cnts, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return {
        "area_px": area,
        "pct_imagen": round(100 * area / m.size, 3) if m.size else 0.0,
        "regiones": int(n - 1),
        "perimetro_px": round(float(sum(cv2.arcLength(c, True) for c in cnts)), 1),
    }


def missing_requirements(layer: Layer) -> list[str]:
    return [mod for mod in layer.requiere if importlib.util.find_spec(mod) is None]


def _check_mask(mask, shape) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.shape != shape:
        raise ValueError(f"la máscara mide {mask.shape} y la imagen {shape}")
    return mask.astype(bool)


def _fire_extra(ctx: Context) -> dict:
    """Detalle de focos (frente de la cicatriz / junto al humo / aislado); opcional, nunca rompe la capa."""
    try:
        from core.fire import fire_details
        foci = fire_details(ctx.fires, ctx.burn, ctx.masks, 1.0)
    except Exception:  # noqa: BLE001
        return {}
    return {"focos": foci, "focos_por_ubicacion": {k: sum(f["ubicacion"] == k for f in foci)
                                                   for k in ("frente de la cicatriz", "junto al humo", "aislado")}}


def run_all_layers(true_color: np.ndarray | None, swir: np.ndarray, wildfires: np.ndarray | None = None, *,
                   layers: list[Layer] | None = None, name: str = "escena",
                   on_layer: Callable[[Layer, dict], None] | None = None) -> dict:
    """Ejecuta todas las capas, cada una aislada en su try/except y cronometrada.

    Devuelve {"scene": {...}, "bases": {"tc","swir","wildfire"}, "layers": {id: resultado}} donde resultado es
    {"status": "ok" | "error" | "no_disponible", "mask": bool HxW | None, "metrics": dict | None,
     "time_s": float, "error": str | None}. El tiempo de un paso común se suma a la primera capa que lo pide.
    on_layer(capa, resultado) se llama al terminar cada capa (para mostrar progreso)."""
    layers = LAYERS if layers is None else layers
    t0 = time.perf_counter()
    scene = scene_from_arrays(swir, true_color, wildfires, name=name)
    scene_time = time.perf_counter() - t0
    ctx = Context(scene)
    shape = scene["valid"].shape

    out: dict[str, dict] = {}
    for i, layer in enumerate(layers):
        t = time.perf_counter()
        res = {"status": "ok", "mask": None, "metrics": None, "time_s": 0.0, "error": None}
        missing = missing_requirements(layer)
        if missing:
            res["status"] = "no_disponible"
            res["error"] = f"Necesita {', '.join(missing)}, que no está instalado en este equipo."
        else:
            try:
                mask = _check_mask(layer.calcular(ctx), shape)
                res["mask"], res["metrics"] = mask, mask_metrics(mask)
                if layer.id == "fire":
                    res.update(_fire_extra(ctx))
                if layer.id == "perimeter":
                    # Perímetro del incendio = borde EXTERIOR suavizado, la misma medida que run.py
                    # (perimetro_px suma además los bordes escalonados de las islas)
                    from core.segmentation import perimeter
                    polys = perimeter(mask)
                    res["metrics"]["perimetro_exterior_px"] = round(float(sum(
                        cv2.arcLength(np.array(p["exterior"], np.float32).reshape(-1, 1, 2), True) for p in polys)), 1)
            except Exception as e:  # noqa: BLE001 - una capa rota no tumba las demás
                res["status"], res["error"] = "error", str(e) or repr(e)
        res["time_s"] = round(time.perf_counter() - t + (scene_time if i == 0 else 0.0), 3)
        out[layer.id] = res
        if on_layer is not None:
            on_layer(layer, res)

    return {
        "scene": {"name": scene["name"], "size": list(scene["size"]), "orig_size": list(scene["orig_size"]),
                  "scale": scene["scale"], "warnings": list(scene["warnings"])},
        "bases": {"tc": scene["tc"], "swir": scene["swir"], "wildfire": scene["wildfire"]},
        "layers": out,
        "time_s": round(time.perf_counter() - t0, 3),
    }

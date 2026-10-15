"""Registro de capas del visor: qué se calcula, cómo se llama para un usuario y cómo se dibuja.

Cada capa recibe un contexto (pipeline.Context) que calcula bajo demanda, y una sola vez, los pasos
comunes de core/: bandas, índices, máscaras de descarte, cicatriz y focos. La capa solo extrae su máscara.

tipo:   "relleno"  (superficie), "contorno" (se dibuja el borde), "puntos" (elementos pequeños con marcador)
patron: trama que acompaña al color para que las capas no se distingan solo por él
        ("liso", "diagonal", "cruzado", "puntos", "horizontal")
requiere: módulos de Python opcionales; si falta alguno la capa sale como "no disponible"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.ndimage import binary_fill_holes


@dataclass(frozen=True)
class Layer:
    id: str
    nombre: str
    descripcion: str
    color: str
    tipo: str
    calcular: Callable[[object], np.ndarray]
    patron: str = "liso"
    visible: bool = False
    requiere: tuple[str, ...] = field(default_factory=tuple)


def _burn(ctx) -> np.ndarray:
    return ctx.burn


def _perimeter(ctx) -> np.ndarray:
    # Lo que queda dentro del perímetro exterior: la cicatriz con sus islas sin quemar incluidas
    return binary_fill_holes(ctx.burn)


def _fire(ctx) -> np.ndarray:
    return ctx.fires


def _burn_active(ctx) -> np.ndarray:
    # Mismo criterio que run.py (core.fire.activity_zone): cicatriz a menos de ~1 % de la diagonal de un foco
    from core.fire import activity_zone
    return ctx.burn & activity_zone(ctx.burn, ctx.fires)


def _exclusion(key: str) -> Callable[[object], np.ndarray]:
    return lambda ctx: ctx.masks[key]


def _nodata(ctx) -> np.ndarray:
    return ~ctx.scene["valid"]


LAYERS: list[Layer] = [
    Layer("burn", "Zona quemada", "Terreno que muestra la huella oscura y rojiza del fuego.",
          "#E8590C", "relleno", _burn, patron="diagonal", visible=True),
    Layer("perimeter", "Perímetro del incendio", "Línea que rodea toda la zona afectada, incluidas las islas sin quemar.",
          "#FFB000", "contorno", _perimeter, visible=True),
    Layer("fire", "Fuego activo", "Puntos muy calientes donde el fuego parece seguir encendido.",
          "#FF2E2E", "puntos", _fire, visible=True),
    Layer("burn_active", "Zona quemada que aún arde", "Parte de la zona quemada que tiene fuego activo muy cerca.",
          "#FF8A3D", "relleno", _burn_active, patron="cruzado"),
    Layer("smoke", "Humo", "Columnas de humo que tapan parcialmente el suelo.",
          "#7FD1C7", "relleno", _exclusion("smoke"), patron="horizontal"),
    Layer("water", "Agua", "Embalses, ríos y lagos; nunca se cuentan como quemados.",
          "#3D8BFD", "relleno", _exclusion("water")),
    Layer("cloud", "Nubes", "Nubes blancas que impiden ver el suelo.",
          "#E6ECF0", "relleno", _exclusion("cloud"), patron="puntos"),
    Layer("bare", "Suelo claro o campos secos", "Tierra desnuda o cultivos secos que se parecen a lo quemado y se descartan.",
          "#C9B38A", "relleno", _exclusion("bare"), patron="puntos"),
    Layer("urban", "Posible zona urbana (poco fiable)", "Zonas con mucho detalle fino, como pueblos; a veces marca parcelas.",
          "#B48EAD", "relleno", _exclusion("urban"), patron="cruzado"),
    Layer("nodata", "Sin datos", "Bordes sin imagen y el logotipo de Copernicus; no se analizan.",
          "#5B6168", "relleno", _nodata, patron="horizontal"),
]

LAYERS_BY_ID = {layer.id: layer for layer in LAYERS}

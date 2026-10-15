"""T1 - Carga de una escena Sentinel-2 (SWIR, True Color y Wildfire).

Uso como módulo:
    from core.ingestion import load_scene
    scene = load_scene("data/niebla")          # carpeta
    scene = load_scene(["a_SWIR.jpg", ...])    # lista de archivos

Uso suelto (guarda PNG de depuración en debug/<escena>/):
    python core/ingestion.py data/niebla
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MAX_SIDE = 2600            # lado mayor de la resolución de trabajo
LOGO_BOX = (0.07, 0.075)   # caja del logo de Copernicus: fracción de (ancho, alto), abajo a la derecha
KINDS = ("swir", "tc", "wildfire")


# ----------------------------------------------------------------------------- lectura
def _read_rgb(path: str) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError(f"No se puede leer la imagen: {path}")
    if im.dtype == np.uint16:                     # GeoTIFF visual de 16 bits -> 8 bits
        im = (im / 257).round().astype(np.uint8)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4:
        im = im[..., :3]
    return np.ascontiguousarray(im[..., ::-1])    # BGR -> RGB


def _list_images(src) -> list[str]:
    if isinstance(src, (str, os.PathLike)):
        p = Path(src)
        if p.is_dir():
            return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
        return [str(p)]
    return [str(f) for f in src]


# ----------------------------------------------------------------------------- identificación
def kind_from_name(path: str) -> str | None:
    """Tipo por nombre de archivo: 'swir', 'tc', 'wildfire' o None."""
    n = Path(path).stem.lower()
    if "swir" in n:
        return "swir"
    if "wildfire" in n or re.search(r"(^|[^a-z])fire", n):
        return "wildfire"
    if "true" in n or re.search(r"(^|[^a-z])tc([^a-z]|$)", n):
        return "tc"
    return None


def content_features(rgb: np.ndarray) -> dict:
    """Rasgos de color usados para adivinar el tipo de imagen cuando el nombre no ayuda."""
    sub = rgb[::4, ::4].reshape(-1, 3).astype(np.int16)
    r, g, b = sub[:, 0], sub[:, 1], sub[:, 2]
    return {
        # La SWIR tiene mucha vegetación verde saturada (G muy por encima de R y B)
        "green_sat": float(np.mean((g - r > 40) & (g - b > 40))),
        # La Wildfire tiene píxeles naranja saturados y el azul "apagado"
        "orange": float(np.mean((r > 200) & (r - b > 120) & (r - g > 40))),
        "blue_p99": float(np.percentile(b, 99)),
        "red_p99": float(np.percentile(r, 99)),
    }


def kind_from_content(rgb: np.ndarray) -> str:
    """Heurística observada en las imágenes de ejemplo (ver README)."""
    f = content_features(rgb)
    if f["green_sat"] > 0.05:
        return "swir"
    # En las Wildfire el azul (p99) queda muy por debajo del rojo (p99); en las TC están cerca
    if f["orange"] > 1e-4 or f["red_p99"] - f["blue_p99"] > 50:
        return "wildfire"
    return "tc"


# ----------------------------------------------------------------------------- máscara de validez
def logo_box(h: int, w: int) -> tuple[slice, slice]:
    bw, bh = int(round(w * LOGO_BOX[0])), int(round(h * LOGO_BOX[1]))
    return slice(h - bh, h), slice(w - bw, w)


def has_logo(rgb: np.ndarray) -> bool:
    """El logo es blanco y poco saturado. Se compara la esquina con el resto de la imagen."""
    h, w = rgb.shape[:2]
    ys, xs = logo_box(h, w)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    white = (hsv[..., 2] > 225) & (hsv[..., 1] < 40)
    corner = white[ys, xs].mean()
    rest = white[::3, ::3].mean()
    return bool(corner > 0.01 and corner > 8 * rest)  # medido: mierla TC 0.0265, el resto <= 0.0011


def nodata_mask(images: list[np.ndarray]) -> np.ndarray:
    """Sin datos = negro puro en TODAS las imágenes y pegado al borde (el agua no es negra en la TC)."""
    black = np.ones(images[0].shape[:2], bool)
    for im in images:
        black &= im.max(axis=2) <= 3
    if not black.any():
        return black
    n, lab, stats, _ = cv2.connectedComponentsWithStats(black.astype(np.uint8), 8)
    h, w = black.shape
    out = np.zeros_like(black)
    min_area = 0.001 * h * w
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        touches = x == 0 or y == 0 or x + bw == w or y + bh == h
        if touches and a >= min_area:
            out |= lab == i
    return out


# ----------------------------------------------------------------------------- carga
def load_scene(src, name: str | None = None, max_side: int = MAX_SIDE) -> dict:
    """Carga una escena y devuelve:
    {tc, swir, wildfire: RGB uint8 a resolución de trabajo o None,
     name, size (H, W) de trabajo, orig_size (H, W), scale (trabajo/original),
     valid: bool HxW (sin logo ni píxeles sin datos), files, warnings}
    """
    paths = _list_images(src)
    if not paths:
        raise ValueError(f"No hay imágenes en {src}")
    warnings: list[str] = []
    imgs: dict[str, np.ndarray] = {}
    files: dict[str, str] = {}
    for p in paths:
        rgb = _read_rgb(p)
        k = kind_from_name(p)
        k_c = kind_from_content(rgb)
        if k is None:
            k = k_c
            warnings.append(f"{Path(p).name}: tipo '{k}' deducido por contenido")
        elif k != k_c:
            warnings.append(f"{Path(p).name}: el nombre dice '{k}' y el contenido parece '{k_c}' (se usa el nombre)")
        if k in imgs:
            warnings.append(f"{Path(p).name}: ya había una imagen '{k}', se ignora")
            continue
        imgs[k], files[k] = rgb, p

    if "swir" not in imgs:
        raise ValueError("La escena no tiene imagen SWIR: es imprescindible para detectar la cicatriz")

    orig_size = imgs["swir"].shape[:2]
    for k, im in list(imgs.items()):
        if im.shape[:2] != orig_size:
            warnings.append(f"{k}: tamaño {im.shape[:2]} distinto de la SWIR {orig_size}; se reescala")
            imgs[k] = cv2.resize(im, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_AREA)

    # Validez a resolución original (el logo es pequeño)
    valid = ~nodata_mask(list(imgs.values()))
    # El logo solo se busca en TC/Wildfire: en la SWIR los campos claros parecen blanco.
    logos = [k for k, im in imgs.items() if k != "swir" and has_logo(im)]
    if logos:
        ys, xs = logo_box(*orig_size)
        valid[ys, xs] = False
        warnings.append(f"logo de Copernicus detectado en {logos}; se enmascara la esquina")

    scale = min(1.0, max_side / max(orig_size))
    if scale < 1.0:
        size = (int(round(orig_size[0] * scale)), int(round(orig_size[1] * scale)))
        for k in imgs:
            imgs[k] = cv2.resize(imgs[k], (size[1], size[0]), interpolation=cv2.INTER_AREA)
        valid = cv2.resize(valid.astype(np.uint8), (size[1], size[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        size = orig_size

    if name is None:
        first = Path(paths[0])
        name = Path(src).name if isinstance(src, (str, os.PathLike)) and Path(src).is_dir() else first.stem

    return {
        "tc": imgs.get("tc"), "swir": imgs["swir"], "wildfire": imgs.get("wildfire"),
        "name": name, "size": size, "orig_size": orig_size, "scale": scale,
        "valid": valid, "files": files, "warnings": warnings, "logo_in": logos,
    }


def group_files(paths: list[str]) -> list[list[str]]:
    """Agrupa archivos sueltos por escena: mismo tamaño y, entre tamaños iguales,
    correlación alta entre el rojo de la TC y el azul de la SWIR."""
    by_size: dict[tuple, list[str]] = {}
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_4)
        by_size.setdefault(im.shape[:2], []).append(p)
    groups = []
    for ps in by_size.values():
        swirs = [p for p in ps if kind_from_name(p) == "swir"]
        if len(swirs) <= 1:
            groups.append(ps)
            continue
        # Varias escenas del mismo tamaño: cada no-SWIR va con la SWIR más correlacionada
        sw_b = {s: cv2.imread(s, cv2.IMREAD_REDUCED_COLOR_4)[..., 0].ravel().astype(float) for s in swirs}
        g = {s: [s] for s in swirs}
        for p in ps:
            if p in swirs:
                continue
            red = cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_4)[..., 2].ravel().astype(float)
            best = max(swirs, key=lambda s: np.corrcoef(red, sw_b[s])[0, 1])
            g[best].append(p)
        groups.extend(g.values())
    return groups


# ----------------------------------------------------------------------------- depuración
def save_debug(scene: dict, root: str = "debug") -> str:
    out = Path(root) / scene["name"]
    out.mkdir(parents=True, exist_ok=True)
    for k in KINDS:
        im = scene[k]
        if im is not None:
            cv2.imwrite(str(out / f"ingest_{k}.png"), im[..., ::-1])
    cv2.imwrite(str(out / "ingest_valid.png"), scene["valid"].astype(np.uint8) * 255)
    return str(out)


if __name__ == "__main__":
    for src in sys.argv[1:] or ["data"]:
        sc = load_scene(src)
        print(f"{sc['name']}: trabajo {sc['size']} original {sc['orig_size']} escala {sc['scale']:.3f}")
        for k in KINDS:
            f = sc["files"].get(k)
            if f:
                feats = content_features(_read_rgb(f)[::2, ::2])
                print(f"  {k:9s} {Path(f).name}  contenido->{kind_from_content(_read_rgb(f))}  "
                      + " ".join(f"{a}={b:.4g}" for a, b in feats.items()))
            else:
                print(f"  {k:9s} (no hay)")
        print(f"  válidos: {sc['valid'].mean():.4f}")
        for w in sc["warnings"]:
            print("  AVISO:", w)
        print("  debug ->", save_debug(sc))

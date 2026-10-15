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


def calibrate(tc: np.ndarray, swir: np.ndarray, valid: np.ndarray) -> dict | None:
    """Recta robusta azul_SWIR ≈ a * rojo_TC + b (ambos son B04), sobre píxeles válidos.
    Sirve para comprobar que las dos fotos están alineadas y para pasar una a la escala de la otra."""
    if valid.sum() < 3:
        return None
    step = max(1, int(valid.sum() // 200_000))
    x = tc[..., 0][valid][::step].astype(np.float32)
    y = swir[..., 2][valid][::step].astype(np.float32)
    lo, hi = np.percentile(x, [2, 98])
    m = (x > lo) & (x < hi)
    if m.sum() < 3 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return None
    a, b = np.polyfit(x[m], y[m], 1)
    return {"a": float(a), "b": float(b), "corr": float(np.corrcoef(x, y)[0, 1])}


def water_raw(swir: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float]:
    """Agua con los valores ORIGINALES de la SWIR: el agua absorbe el infrarrojo cercano (B8A) y el de onda corta (B12),
    así que max(B12, B8A) forma un grupo en lo más oscuro, separado de la tierra por un hueco en el histograma.
    El umbral es ese hueco, buscado en cada escena: sirve igual con un embalse pequeño que con media imagen de mar.
    (La versión anterior suponía que el agua era como mucho ~10 % de la escena y fallaba con mucho mar.)"""
    if not valid.any():
        return np.zeros(valid.shape, bool), 0.0
    mx = swir[..., :2].max(axis=2)
    v = mx[valid]
    med = float(np.median(v))
    hist = np.bincount(v.ravel(), minlength=256)[:256].astype(np.float64)
    hist = np.convolve(hist, np.ones(5) / 5, mode="same")                  # suaviza el histograma
    lo, hi = max(1, int(0.25 * med)), max(2, int(0.5 * med))
    peak = int(np.argmax(hist[: lo + 1]))                                   # modo oscuro (agua, si la hay)
    t = float(peak + int(np.argmin(hist[peak: hi + 1])))                    # hueco entre el agua y la tierra
    water = (mx <= t) & valid
    water = cv2.morphologyEx(water.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    return water, t


def bands(scene: dict) -> dict:
    """Devuelve B02, B03, B04, B8A, B12 (float32, 0..1). Las que no se puedan obtener, ausentes.
    B12/B8A/B04 salen de la SWIR; B02/B03 solo existen si hay True Color.
    Claves extra: 'valid' (máscara), 'B04_tc' (B04 desde la TC), 'calib'."""
    valid = scene["valid"]
    swir, tc = scene["swir"], scene["tc"]
    water, t_water = water_raw(swir, valid)
    land = valid & ~water
    stats = land if land.sum() > 0.05 * max(int(valid.sum()), 1) else valid   # percentiles sobre TIERRA: el mar no los arrastra
    out = {
        "B12": pnorm(swir[..., 0].astype(np.float32), stats),
        "B8A": pnorm(swir[..., 1].astype(np.float32), stats),
        "B04": pnorm(swir[..., 2].astype(np.float32), stats),
        "valid": valid,
        "land": stats,
        "_water": water,
        "_water_t": t_water,
        "_swir": swir,   # SWIR original sin recortar: los focos saturan B12 y el percentil 99 los aplanaría
    }
    if tc is not None:
        out["B04_tc"] = pnorm(tc[..., 0].astype(np.float32), stats)
        out["B03"] = pnorm(tc[..., 1].astype(np.float32), stats)
        out["B02"] = pnorm(tc[..., 2].astype(np.float32), stats)
        calibration = calibrate(tc, swir, valid)
        if calibration is not None:
            out["calib"] = calibration
        else:
            scene["warnings"].append("calibracion visual no disponible: contraste o muestras insuficientes")
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
        # NBR con los valores ORIGINALES de la SWIR (G=B8A, R=B12; misma ganancia en los dos canales). El NBR normalizado
        # por percentiles de arriba pierde el signo físico: > 0 = más infrarrojo cercano que de onda corta (vegetación o
        # suelo seco), < 0 = quemado. Se usa para decidir si una mancha tiene evidencia de quemado.
        "NBR_RAW": _nd(b["_swir"][..., 1].astype(np.float32), b["_swir"][..., 0].astype(np.float32)),
        "_bands": b,   # referencia interna: permite burn_mask(indices, masks) sin pasar las bandas aparte
    }


def _gray_png(x: np.ndarray, path: Path, lo=1, hi=99):
    a, b = np.percentile(x, [lo, hi])
    cv2.imwrite(str(path), (np.clip((x - a) / max(b - a, EPS), 0, 1) * 255).astype(np.uint8))


def save_debug(scene: dict, b: dict, ind: dict, root: str = "debug") -> str:
    out = Path(root) / scene["name"]
    out.mkdir(parents=True, exist_ok=True)
    for k, v in ind.items():
        if k.startswith("_"):
            continue
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
            if k.startswith("_"):
                continue
            print(f"  {k:8s} p1/p50/p99 = {np.percentile(v[sc['valid']], [1, 50, 99]).round(2)}")
        print("  debug ->", save_debug(sc, b, ind))


# ----------------------------------------------------------------------------- máscaras de descarte (T6)
from skimage.filters import threshold_otsu  # noqa: E402


def _diag(shape) -> float:
    return float(np.hypot(*shape))


def _remove_small(m: np.ndarray, min_px: int) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, 4] >= min_px
    return keep[lab]


def _otsu_tail(x: np.ndarray, valid: np.ndarray, q: float) -> float:
    """Otsu aplicado solo a la cola baja (percentil q) de los valores: separa 'muy oscuro' de 'oscuro'."""
    v = x[valid]
    if v.size == 0:
        return 0.0
    v = v[v <= np.percentile(v, q)]
    return float(threshold_otsu(v))


def exclusion_masks(b: dict, ind: dict) -> dict:
    """Máscaras booleanas HxW: water, cloud, bare, urban, smoke (todas adaptativas, por percentiles/Otsu)."""
    valid = b["valid"]
    shape = valid.shape
    if not valid.any():
        return {key: np.zeros(shape, bool) for key in ("water", "cloud", "bare", "urban", "smoke")}
    d = _diag(shape)
    mx = 1 - ind["WATER"]                       # máximo de los 3 canales SWIR
    bright, sat, nbr, red = ind["BRIGHT"], ind["SAT"], ind["NBR"], ind["RED_DOM"]
    b12, b8a, b04 = b["B12"], b["B8A"], b["B04"]
    k = max(3, int(d / 400) | 1)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # AGUA: ver water_raw() (hueco del histograma de max(B12, B8A) originales, calculado en bands()).
    if "_water" in b:
        water = _remove_small(b["_water"], max(4, int(d / 700) ** 2))
    else:
        water = np.zeros(shape, bool)
    land = b.get("land", valid)          # los percentiles de las demás máscaras, solo sobre tierra

    # HUMO: turquesa = B8A y B04 claramente por encima de B12 (el rojo baja, azul y verde suben)
    cyan = np.minimum(b8a, b04) - b12
    t_s = float(np.percentile(cyan[land], 90))
    smoke = (cyan > max(t_s, 0.15)) & (sat > 0.15) & valid & ~water
    smoke = cv2.morphologyEx(smoke.astype(np.uint8), cv2.MORPH_OPEN, k_open).astype(bool)
    smoke = _remove_small(smoke, int(0.0002 * valid.size))

    # TIERRA CLARA / CAMPOS SECOS: muy brillante en SWIR (lo quemado y el matorral son oscuros)
    t_b = float(np.percentile(bright[land], 75))
    bare = (bright > max(t_b, float(threshold_otsu(bright[land][::13])))) & valid & ~water & ~smoke
    bare = cv2.morphologyEx(bare.astype(np.uint8), cv2.MORPH_OPEN, k_open).astype(bool)

    # NUBE: blanco (poco saturado) y muy brillante; con True Color se exige también que sea claro y gris allí
    cloud = (bright > np.percentile(bright[land], 97)) & (sat < np.percentile(sat[land], 15)) & valid
    if "B03" in b:
        grey_tc = (b["B03"] > np.percentile(b["B03"][land], 90)) & (b["B02"] > np.percentile(b["B02"][land], 85))
        cloud &= grey_tc
    cloud = _remove_small(cv2.morphologyEx(cloud.astype(np.uint8), cv2.MORPH_OPEN, k_open).astype(bool),
                          int(0.0003 * valid.size))

    # CIUDAD (EXPERIMENTAL, poco fiable: también marca parcelas): contraste fino denso + poca vegetación.
    hf = np.abs(bright - cv2.GaussianBlur(bright, (0, 0), 1.5))
    dens = cv2.blur((hf > np.percentile(hf[land], 70)).astype(np.float32), (k * 7, k * 7))
    urban = (dens > np.percentile(dens[land], 93)) & valid & ~water & ~smoke
    urban = cv2.morphologyEx(urban.astype(np.uint8), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 8 + 1, k * 8 + 1))).astype(bool)
    urban = _remove_small(urban, int(0.002 * valid.size))

    return {"water": water, "cloud": cloud, "bare": bare, "urban": urban, "smoke": smoke}


def save_mask_debug(scene: dict, masks: dict, root: str = "debug") -> str:
    out = Path(root) / scene["name"]
    out.mkdir(parents=True, exist_ok=True)
    for k, m in masks.items():
        cv2.imwrite(str(out / f"mask_{k}.png"), m.astype(np.uint8) * 255)
    # montaje: SWIR con cada máscara pintada de un color
    base = scene["swir"][..., ::-1].copy()
    colors = {"water": (255, 0, 0), "smoke": (255, 255, 0), "bare": (0, 255, 255), "urban": (255, 0, 255), "cloud": (255, 255, 255)}
    for k, m in masks.items():
        base[m] = (0.4 * base[m] + 0.6 * np.array(colors[k])).astype(np.uint8)
    cv2.imwrite(str(out / "mask_overview.png"), base)
    return str(out)

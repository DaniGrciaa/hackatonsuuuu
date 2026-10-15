"""T4/T5 - Procesa todas las escenas de una carpeta sin tocar nada.

    python run.py data/ --out salida/ [--m-per-px 10] [--variant v2]

Si <carpeta> contiene imágenes directamente se trata como una sola escena; si no, cada
subcarpeta con imágenes es una escena. Por escena guarda en <out>/<escena>/:
  perimetro_swir.png, perimetro_tc.png, mascara_quemado.png, capa_focos.png (RGBA),
  mascaras_descarte/*.png, resultado.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from core.fire import fire_details
from core.ingestion import IMG_EXT, load_scene
from core.segmentation import DEFAULT_VARIANT, VARIANTS, burn_details, perimeter
from core.spectral import bands, exclusion_masks, indices

COL_PERIM = (0, 255, 255)   # amarillo (BGR) para el perímetro
COL_HOLE = (255, 200, 0)    # azul claro para huecos (embalses)
COL_FIRE = (0, 0, 255)      # rojo para los focos


def find_scenes(root: str) -> list[Path]:
    p = Path(root)
    if any(f.suffix.lower() in IMG_EXT for f in p.iterdir()):
        return [p]
    return sorted(d for d in p.iterdir() if d.is_dir() and any(f.suffix.lower() in IMG_EXT for f in d.iterdir()))


def detect_fires(b: dict, ind: dict, burn: np.ndarray, masks: dict):
    """Capa de focos: usa core/fire.py (puesto A) si existe; si no, capa vacía.
    Firma esperada (a confirmar con A): fire.detect_fires(bands, indices, burn_mask, masks) -> bool HxW."""
    try:
        from core import fire  # noqa: PLC0415
        out = np.asarray(fire.detect_fires(b, ind, burn, masks))
        return out.astype(bool), None
    except ImportError:
        return np.zeros(burn.shape, bool), "core/fire.py no existe todavía: capa de focos vacía"
    except Exception as e:  # el fallo de A no debe tumbar mi salida
        return np.zeros(burn.shape, bool), f"fire.detect_fires falló: {e!r}"


def draw_overlay(base_rgb: np.ndarray, burn: np.ndarray, fires: np.ndarray, water: np.ndarray,
                 foci: list | None = None, scale: float = 1.0) -> np.ndarray:
    im = base_rgb[..., ::-1].copy()
    th = max(2, im.shape[1] // 800)
    tint = im.copy()
    tint[burn] = (0.55 * tint[burn] + 0.45 * np.array(COL_PERIM)).astype(np.uint8)
    im = cv2.addWeighted(tint, 0.6, im, 0.4, 0)
    cnts, hier = cv2.findContours(burn.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(cnts):
        cv2.drawContours(im, [c], -1, COL_HOLE if hier[0][i][3] != -1 else COL_PERIM, th)
    if fires.any():
        fc, _ = cv2.findContours(fires.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, fc, -1, COL_FIRE, th + 1)
    # Círculos para que los focos (a menudo de 1-10 px) se vean; más finos si la confianza es baja
    r = max(8, im.shape[1] // 180)
    for f in foci or []:
        x, y = int(f["centro"][0] * scale), int(f["centro"][1] * scale)
        cv2.circle(im, (x, y), r, COL_FIRE, th + 1 if f["confianza"] != "baja" else th)
    _legend(im, th)
    return im


def _legend(im: np.ndarray, th: int):
    s = im.shape[1] / 2500
    x0, y0, dy = int(20 * s), int(20 * s), int(42 * s)
    cv2.rectangle(im, (x0 - 10, y0 - 10), (x0 + int(470 * s), y0 + 3 * dy), (0, 0, 0), -1)
    items = [(COL_PERIM, "Perimetro area quemada"), (COL_HOLE, "Hueco: agua o isla no quemada"), (COL_FIRE, "Foco de fuego activo")]
    for k, (col, txt) in enumerate(items):
        y = y0 + k * dy + dy // 2
        cv2.line(im, (x0, y), (x0 + int(50 * s), y), col, th + 2)
        cv2.putText(im, txt, (x0 + int(65 * s), y + int(10 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, (255, 255, 255), max(1, th // 2))


def process(scene_dir: Path, out_root: Path, m_per_px: float | None, variant: str) -> dict:
    t0 = time.time()
    sc = load_scene(scene_dir)
    b = bands(sc)
    ind = indices(b)
    masks = exclusion_masks(b, ind)
    det = burn_details(b, ind, masks, variant)
    burn = det["mask"]
    fires, fire_warn = detect_fires(b, ind, burn, masks)
    polys = perimeter(burn, sc["scale"])
    foci = fire_details(fires, burn, masks, sc["scale"])
    active = burn & cv2.dilate(fires.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)

    out = out_root / sc["name"]
    (out / "mascaras_descarte").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "perimetro_swir.png"), draw_overlay(sc["swir"], burn, fires, masks["water"], foci, sc["scale"]))
    if sc["tc"] is not None:
        cv2.imwrite(str(out / "perimetro_tc.png"), draw_overlay(sc["tc"], burn, fires, masks["water"], foci, sc["scale"]))
    cv2.imwrite(str(out / "mascara_quemado.png"), burn.astype(np.uint8) * 255)
    rgba = np.zeros((*burn.shape, 4), np.uint8)
    rgba[fires] = (0, 0, 255, 255)
    cv2.imwrite(str(out / "capa_focos.png"), rgba)
    for k, m in masks.items():
        cv2.imwrite(str(out / "mascaras_descarte" / f"{k}.png"), m.astype(np.uint8) * 255)

    n_valid = int(sc["valid"].sum())
    n_burn = int(burn.sum())
    perim_px = float(sum(cv2.arcLength(np.array(p["exterior"], np.float32).reshape(-1, 1, 2), True) for p in polys))
    orig_px = n_burn / sc["scale"] ** 2           # píxeles equivalentes en la imagen original
    res = {
        "escena": sc["name"], "variante": variant,
        "tamano_trabajo": list(sc["size"]), "tamano_original": list(sc["orig_size"]), "escala": round(sc["scale"], 4),
        "pixeles_quemados_trabajo": n_burn, "pixeles_quemados_original_equiv": int(round(orig_px)),
        "porcentaje_quemado": round(100 * n_burn / n_valid, 2),
        "num_componentes": len(polys), "num_huecos": int(sum(len(p["holes"]) for p in polys)),
        "perimetro_px_original": round(perim_px, 1),
        "porcentaje_agua": round(100 * masks["water"].sum() / n_valid, 2),
        "porcentaje_humo": round(100 * masks["smoke"].sum() / n_valid, 2),
        "pixeles_focos": int(fires.sum()),
        "num_focos": len(foci),
        "focos_por_ubicacion": {k: sum(f["ubicacion"] == k for f in foci) for k in ("frente de la cicatriz", "junto al humo", "aislado")},
        "porcentaje_quemado_con_actividad": round(100 * active.sum() / max(n_burn, 1), 2),
        "porcentaje_quemado_sin_actividad": round(100 * (n_burn - active.sum()) / max(n_burn, 1), 2),
        "focos": foci,
        "descartados": det["discarded"][:200], "umbrales": {k: round(v, 4) for k, v in det["thresholds"].items()},
        "poligonos": polys, "avisos": sc["warnings"] + ([fire_warn] if fire_warn else []),
    }
    if sc["tc"] is None:
        res["avisos"].append("sin True Color: solo SWIR")
    if m_per_px:
        res["m_por_px"] = m_per_px
        res["hectareas"] = round(orig_px * m_per_px ** 2 / 1e4, 1)
        res["perimetro_km"] = round(perim_px * m_per_px / 1000, 2)
    res["tiempo_s"] = round(time.time() - t0, 2)
    (out / "resultado.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta")
    ap.add_argument("--out", default="salida")
    ap.add_argument("--m-per-px", type=float, default=None, help="metros por píxel de la imagen ORIGINAL (opcional)")
    ap.add_argument("--variant", default=DEFAULT_VARIANT, choices=VARIANTS)
    a = ap.parse_args()
    scenes = find_scenes(a.carpeta)
    if not scenes:
        raise SystemExit(f"No se encontraron escenas en {a.carpeta}")
    for s in scenes:
        try:
            r = process(s, Path(a.out), a.m_per_px, a.variant)
            extra = f", {r['hectareas']} ha" if "hectareas" in r else ""
            print(f"{r['escena']}: {r['porcentaje_quemado']}% quemado, {r['num_componentes']} manchas{extra}, {r['tiempo_s']} s")
            for w in r["avisos"]:
                print("   aviso:", w)
        except Exception as e:  # una escena rota no debe parar el resto
            print(f"{s.name}: ERROR {e!r}")


if __name__ == "__main__":
    main()

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

from core.escala import ha_per_px, km_per_px, scale_for
from core.fire import activity_radius_px, activity_zone
from core.ingestion import IMG_EXT, load_scene
from core.segmentation import DEFAULT_VARIANT, VARIANTS, burn_details, perimeter
from core.spectral import bands, exclusion_masks, indices

COL_PERIM = (0, 255, 255)   # amarillo (BGR): perímetro y relleno de lo quemado SIN actividad
COL_ACTIVE = (0, 120, 255)  # naranja: superficie quemada CON actividad (junto a un foco)
COL_HOLE = (255, 200, 0)    # azul claro para huecos (embalses, islas)
COL_SMOKE_BURN = (255, 120, 200)  # violeta: POSIBLE zona quemada bajo el humo (capa aparte, menos fiable)
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
        if out.ndim != 2 or out.shape != burn.shape or out.dtype != np.bool_:
            raise ValueError("detect_fires debe devolver bool HxW a resolucion de trabajo")
        return out, None
    except ImportError:
        return np.zeros(burn.shape, bool), "core/fire.py no existe todavía: capa de focos vacía"
    except Exception as e:  # el fallo de A no debe tumbar mi salida
        return np.zeros(burn.shape, bool), f"fire.detect_fires falló: {e!r}"


def draw_overlay(base_rgb: np.ndarray, burn: np.ndarray, fires: np.ndarray, active: np.ndarray,
                 foci: list | None = None, scale: float = 1.0, footer: str = "",
                 under_smoke: np.ndarray | None = None) -> np.ndarray:
    im = base_rgb[..., ::-1].copy()
    th = max(2, im.shape[1] // 800)
    tint = im.copy()
    calm = burn & ~active
    tint[calm] = (0.55 * tint[calm] + 0.45 * np.array(COL_PERIM)).astype(np.uint8)
    tint[active] = (0.2 * tint[active] + 0.8 * np.array(COL_ACTIVE)).astype(np.uint8)
    tint[fires] = COL_FIRE
    im = cv2.addWeighted(tint, 0.7, im, 0.3, 0)
    if under_smoke is not None and under_smoke.any():
        # trama de rayas (no relleno liso) y borde discontinuo: se ve que es otra cosa, menos segura
        yy, xx = np.mgrid[0:im.shape[0], 0:im.shape[1]]
        hatch = under_smoke & (((xx + yy) // max(3, th * 2)) % 3 == 0)
        im[hatch] = (0.25 * im[hatch] + 0.75 * np.array(COL_SMOKE_BURN)).astype(np.uint8)
        cs, _ = cv2.findContours(under_smoke.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            for j in range(0, len(c), 8):
                cv2.polylines(im, [c[j:j + 4]], False, COL_SMOKE_BURN, max(1, th - 1))
    cnts, hier = cv2.findContours(burn.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(cnts):
        cv2.drawContours(im, [c], -1, COL_HOLE if hier[0][i][3] != -1 else COL_PERIM, th)
    # Círculos para que los focos (a menudo de 1-10 px) se vean; más finos si la confianza es baja
    r = max(8, im.shape[1] // 180)
    for f in foci or []:
        x, y = int(f["centro"][0] * scale), int(f["centro"][1] * scale)
        cv2.circle(im, (x, y), r, COL_FIRE, th + 1 if f["confianza"] != "baja" else th)
    _legend(im, th, footer)
    return im


def _legend(im: np.ndarray, th: int, footer: str = ""):
    """Leyenda (sin tildes: la fuente de OpenCV no las tiene). Rellenos como cuadrados, líneas como líneas."""
    s = im.shape[1] / 2500
    x0, y0, dy = int(20 * s), int(20 * s), int(42 * s)
    items = [("line", COL_PERIM, "Perimetro del area afectada"),
             ("fill", COL_PERIM, "Afectada SIN actividad aparente"),
             ("fill", COL_ACTIVE, "Afectada CON posible fuego activo cerca"),
             ("circle", COL_FIRE, "Posible foco activo"),
             ("line", COL_HOLE, "Hueco: agua o isla no quemada"),
             ("fill", COL_SMOKE_BURN, "Posible afectada bajo el humo (menos fiable)")]
    cv2.rectangle(im, (x0 - 10, y0 - 10), (x0 + int(640 * s), y0 + len(items) * dy), (0, 0, 0), -1)
    for k, (kind, col, txt) in enumerate(items):
        y = y0 + k * dy + dy // 2
        if kind == "line":
            cv2.line(im, (x0, y), (x0 + int(50 * s), y), col, th + 2)
        elif kind == "fill":
            cv2.rectangle(im, (x0, y - int(12 * s)), (x0 + int(50 * s), y + int(12 * s)), col, -1)
        else:
            cv2.circle(im, (x0 + int(25 * s), y), int(13 * s), col, th + 1)
        cv2.putText(im, txt, (x0 + int(65 * s), y + int(10 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, (255, 255, 255), max(1, th // 2))
    if footer:
        h = im.shape[0]
        (tw, tht), _ = cv2.getTextSize(footer, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, max(1, th // 2))
        cv2.rectangle(im, (x0 - 10, h - y0 - tht - int(20 * s)), (x0 + tw + 10, h - y0 + int(5 * s)), (0, 0, 0), -1)
        cv2.putText(im, footer, (x0, h - y0 - int(8 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, (255, 255, 255), max(1, th // 2))


def process(scene_dir: Path, out_root: Path, m_per_px: float | None, variant: str) -> dict:
    t0 = time.time()
    sc = load_scene(scene_dir)
    # Escala: --m-per-px manda; si no, <escena>/escala.json; si no hay ninguna, todo en píxeles
    escala = ({"m_por_px_x": m_per_px, "m_por_px_y": m_per_px, "fuente": "--m-per-px"} if m_per_px
              else scale_for(scene_dir))
    if not sc["valid"].any():
        raise ValueError("escena sin pixeles validos")
    b = bands(sc)
    ind = indices(b)
    masks = exclusion_masks(b, ind)
    det = burn_details(b, ind, masks, variant)
    burn = det["mask"]
    under = det.get("under_smoke", np.zeros_like(burn))
    fires, fire_warn = detect_fires(b, ind, burn, masks)
    polys = perimeter(burn, sc["scale"])
    foci = []
    if fire_warn is None:
        from core.fire import fire_details
        foci = fire_details(fires, burn, masks, sc["scale"])
    active = activity_zone(burn, fires)
    n_valid = int(sc["valid"].sum())
    n_burn = int(burn.sum())
    n_act = int((active & burn).sum())
    pct_act = round(100 * n_act / max(n_burn, 1), 1)
    area_txt = f"{n_burn * ha_per_px(escala, sc['scale']):,.0f} ha".replace(",", ".") if escala \
        else f"{100 * n_burn / n_valid:.1f}% de la imagen"
    footer = (f"Afectada: {area_txt}  |  con actividad {pct_act}%  |  "
              f"sin actividad {100 - pct_act if n_burn else 0:.1f}%  |  {len(foci)} posibles focos")

    out = out_root / sc["name"]
    (out / "mascaras_descarte").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "perimetro_swir.png"), draw_overlay(sc["swir"], burn, fires, active, foci, sc["scale"], footer, under))
    if sc["tc"] is not None:
        cv2.imwrite(str(out / "perimetro_tc.png"), draw_overlay(sc["tc"], burn, fires, active, foci, sc["scale"], footer, under))
    cv2.imwrite(str(out / "mascara_quemado.png"), burn.astype(np.uint8) * 255)
    cv2.imwrite(str(out / "mascara_actividad.png"), active.astype(np.uint8) * 255)   # 1 canal, 0/255, resolución de trabajo
    cv2.imwrite(str(out / "mascara_posible_bajo_humo.png"), under.astype(np.uint8) * 255)
    # Canonical evaluation masks use original image coordinates. Existing
    # masks and overlays retain the working resolution for compatibility.
    height, width = sc["orig_size"]
    valid_original = sc["valid_original"]
    def original_mask(mask):
        if mask.shape != sc["orig_size"]:
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        return mask & valid_original
    for name, mask in (("affected", burn), ("active", fires), ("valid", valid_original)):
        cv2.imwrite(str(out / f"{name}.png"), original_mask(mask).astype(np.uint8) * 255)
    rgba = np.zeros((*burn.shape, 4), np.uint8)
    rgba[fires] = (0, 0, 255, 255)
    cv2.imwrite(str(out / "capa_focos.png"), rgba)
    for k, m in masks.items():
        cv2.imwrite(str(out / "mascaras_descarte" / f"{k}.png"), m.astype(np.uint8) * 255)

    perim_px = float(sum(cv2.arcLength(np.array(p["exterior"], np.float32).reshape(-1, 1, 2), True) for p in polys))
    orig_px = n_burn / sc["scale"] ** 2           # píxeles equivalentes en la imagen original
    res = {
        "escena": sc["name"], "variante": variant,
        "archivos_entrada": {key: Path(value).name for key, value in sc["files"].items()},
        "mascaras_evaluacion": {key: f"{key}.png" for key in ("affected", "active", "valid")},
        "coordenadas_mascaras": "original", "coordenadas_overlays": "trabajo",
        "orden_dimensiones": "alto, ancho",
        "estado_afectado": "disponible",
        "estado_activo": "disponible" if fire_warn is None else "no_disponible",
        "tamano_trabajo": list(sc["size"]), "tamano_original": list(sc["orig_size"]), "escala": round(sc["scale"], 4),
        "pixeles_quemados_trabajo": n_burn, "pixeles_quemados_original_equiv": int(round(orig_px)),
        "porcentaje_quemado": round(100 * n_burn / n_valid, 2),
        "num_componentes": len(polys), "num_huecos": int(sum(len(p["holes"]) for p in polys)),
        "perimetro_px_original": round(perim_px, 1),
        "porcentaje_agua": round(100 * masks["water"].sum() / n_valid, 2),
        "porcentaje_humo": round(100 * masks["smoke"].sum() / n_valid, 2),
        "porcentaje_posible_bajo_humo": round(100 * under.sum() / n_valid, 2),   # aparte: NO incluido en porcentaje_quemado
        "pixeles_focos": int(fires.sum()),
        "num_focos": len(foci),
        "focos_por_ubicacion": {k: sum(f["ubicacion"] == k for f in foci) for k in ("frente de la cicatriz", "junto al humo", "aislado")},
        "porcentaje_quemado_con_actividad": round(100 * n_act / max(n_burn, 1), 2),
        "porcentaje_quemado_sin_actividad": round(100 * (n_burn - n_act) / max(n_burn, 1), 2) if n_burn else 0.0,
        "radio_actividad_px": activity_radius_px(burn.shape),
        "focos": foci,
        "descartados": det["discarded"][:200], "umbrales": {k: round(v, 4) for k, v in det["thresholds"].items()},
        "poligonos": polys, "avisos": sc["warnings"] + ([fire_warn] if fire_warn else []),
    }
    if sc["tc"] is None:
        res["avisos"].append("sin True Color: solo SWIR")
    if escala:
        hpp, kpp = ha_per_px(escala, sc["scale"]), km_per_px(escala, sc["scale"])
        res["escala"] = escala
        res["hectareas"] = round(n_burn * hpp, 1)
        res["hectareas_con_actividad"] = round(n_act * hpp, 1)
        res["hectareas_posible_bajo_humo"] = round(int(under.sum()) * hpp, 1)
        res["perimetro_km"] = round(perim_px * km_per_px(escala, 1.0), 2)   # perim_px ya está en píxeles originales
    else:
        res["avisos"].append("sin escala conocida: cifras en píxeles (añade escala.json o --m-per-px para hectáreas)")
    res["tiempo_s"] = round(time.time() - t0, 2)
    (out / "resultado.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
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
    failures = []
    for s in scenes:
        try:
            r = process(s, Path(a.out), a.m_per_px, a.variant)
            extra = f", {r['hectareas']} ha" if "hectareas" in r else ""
            print(f"{r['escena']}: {r['porcentaje_quemado']}% quemado, {r['num_componentes']} manchas{extra}, {r['tiempo_s']} s")
            for w in r["avisos"]:
                print("   aviso:", w)
        except Exception as e:  # una escena rota no debe parar el resto
            print(f"{s.name}: ERROR {e!r}")
            failures.append(s.name)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

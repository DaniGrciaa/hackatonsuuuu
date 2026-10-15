"""Métricas del apartado 2.1 de la evaluación frente a una referencia anotada a mano (si existe).

Zona afectada: IoU, Dice (= F1 por píxel), precisión y recall.
Puntos calientes: precisión, recall y F1 por FOCO, con un criterio de coincidencia fijo (un foco detectado acierta si
hay fuego anotado a menos de TOL_PX píxeles de análisis, y un foco anotado se encuentra si hay detección a esa distancia).
Todo se mide solo donde hay anotación (valid.png de la referencia): la cobertura dice qué parte de la imagen es.

Las referencias son las de Jesús (referencias/desarrollo/<nombre del True Color>/{affected,active,valid}.png).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from evaluation import evaluate_mask
from evaluar_proyecto import canonical_name

ROOT = Path(__file__).resolve().parent
REF_ROOT = ROOT / "referencias" / "desarrollo"
TOL_PX = 5          # tolerancia de coincidencia de focos, en píxeles de la imagen analizada


def find_reference(tc_name: str | None, ref_root: Path = REF_ROOT) -> Path | None:
    if not tc_name or not ref_root.exists():
        return None
    hits = [p for p in ref_root.iterdir() if p.is_dir() and canonical_name(p.name) == canonical_name(tc_name)]
    return hits[0] if len(hits) == 1 else None


def _load(p: Path) -> np.ndarray:
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127


def _to(mask: np.ndarray, shape) -> np.ndarray:
    if mask.shape == tuple(shape):
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def _focos(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray, tol: int) -> dict:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    pred, ref = pred & valid, ref & valid
    ref_d = cv2.dilate(ref.astype(np.uint8), k).astype(bool)
    pred_d = cv2.dilate(pred.astype(np.uint8), k).astype(bool)
    n_p, lab_p = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    n_r, lab_r = cv2.connectedComponents(ref.astype(np.uint8), connectivity=8)
    tp_p = sum(bool(ref_d[lab_p == i].any()) for i in range(1, n_p))
    tp_r = sum(bool(pred_d[lab_r == i].any()) for i in range(1, n_r))
    p = tp_p / (n_p - 1) if n_p > 1 else None
    r = tp_r / (n_r - 1) if n_r > 1 else None
    f1 = 2 * p * r / (p + r) if p is not None and r is not None and (p + r) > 0 else None
    return {"precision": p, "recall": r, "f1": f1, "detectados": n_p - 1, "anotados": n_r - 1, "tolerancia_px": TOL_PX}


def evaluate(burn: np.ndarray, fire: np.ndarray, image_valid: np.ndarray, tc_name: str | None) -> dict | None:
    """burn, fire, image_valid: bool a resolución de análisis. Devuelve None si no hay referencia para esa imagen."""
    ref = find_reference(tc_name)
    if ref is None:
        return None
    r_valid = _load(ref / "valid.png")
    shape = r_valid.shape
    valid = r_valid & _to(image_valid, shape)
    tol = max(1, round(TOL_PX * shape[1] / burn.shape[1]))
    afe = evaluate_mask(_to(burn, shape), _load(ref / "affected.png"), valid)
    return {
        "referencia": "anotación manual parcial (Jesús)",
        "cobertura_pct": round(afe["coverage_pct"], 2),
        "afectada": {"iou": afe["iou"], "dice": afe["f1"], "precision": afe["precision"], "recall": afe["recall"]},
        "focos": _focos(_to(fire, shape), _load(ref / "active.png"), valid, tol),
    }

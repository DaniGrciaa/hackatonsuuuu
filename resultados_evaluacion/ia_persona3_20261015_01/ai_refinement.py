"""Random Forest experimental para JPG alineados; no interpreta bandas fisicas."""
from pathlib import Path
import time

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

SEED = 20261015
TILE = 512
HALO = 15  # radio maximo de las ventanas de textura (31 x 31)
THRESHOLD = 0.5
RF_CONFIG = dict(n_estimators=96, max_depth=16, min_samples_leaf=3,
                 max_features="sqrt", n_jobs=2, random_state=SEED)
FEATURE_NAMES = [f"{source}_{name}" for source in ("tc", "swir") for name in
                 ("r", "g", "b", "h", "s", "v", "lab_l", "lab_a", "lab_b",
                  "gray_mean7", "gray_std7", "gray_mean31", "gray_std31",
                  "chromatic_r", "chromatic_g")]


def validate_images(tc, swir):
    if tc.shape != swir.shape or tc.ndim != 3 or tc.shape[2] != 3:
        raise ValueError("Se necesitan dos imagenes RGB alineadas del mismo tamano.")
    if tc.dtype != np.uint8 or swir.dtype != np.uint8:
        raise ValueError("Las imagenes deben ser RGB uint8.")


def feature_tile(tc, swir, box):
    """Caracteristicas en una ventana original, con halo para no crear costuras."""
    validate_images(tc, swir)
    x0, y0, x1, y1 = box
    height, width = tc.shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Ventana fuera de imagen.")
    left, top = max(0, x0 - HALO), max(0, y0 - HALO)
    right, bottom = min(width, x1 + HALO), min(height, y1 + HALO)
    features = []
    for image in (tc, swir):
        patch = np.ascontiguousarray(image[top:bottom, left:right])
        rgb = patch.astype(np.float32) / 255
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv /= np.array([180, 255, 255], dtype=np.float32)
        lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32) / 255
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        channels = [rgb[..., i] for i in range(3)]
        channels += [hsv[..., i] for i in range(3)]
        channels += [lab[..., i] for i in range(3)]
        for size in (7, 31):
            mean = cv2.boxFilter(gray, -1, (size, size), borderType=cv2.BORDER_REFLECT_101)
            mean2 = cv2.boxFilter(gray * gray, -1, (size, size), borderType=cv2.BORDER_REFLECT_101)
            channels += [mean, np.sqrt(np.maximum(0, mean2 - mean * mean))]
        total = np.maximum(rgb.sum(axis=2), 1 / 255)
        channels += [rgb[..., 0] / total, rgb[..., 1] / total]
        ys, xs = slice(y0 - top, y1 - top), slice(x0 - left, x1 - left)
        features.extend(channel[ys, xs] for channel in channels)
    return np.stack(features, axis=-1).astype(np.float32, copy=False)


def tiles(height, width, tile=TILE):
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            yield (x, y, min(width, x + tile), min(height, y + tile))


def sample_scene(tc, swir, affected, active, valid, limit=8000, seed=SEED):
    """Muestrea SOLO regiones anotadas; no etiqueta el resto como negativo."""
    validate_images(tc, swir)
    height, width = tc.shape[:2]
    if any(mask.shape != (height, width) or mask.dtype != np.bool_
           for mask in (affected, active, valid)):
        raise ValueError("Referencias booleanas de tamano original requeridas.")
    rng = np.random.default_rng(seed)
    coordinates, classes, counts = [], [], {}
    for label, mask in ((0, valid & ~affected),
                        (1, valid & affected & ~active), (2, valid & active)):
        indices = np.flatnonzero(mask)
        counts[str(label)] = int(indices.size)
        if indices.size > limit:
            indices = rng.choice(indices, limit, replace=False)
        coordinates.append(indices)
        classes.append(np.full(indices.size, label, dtype=np.uint8))
    indices, labels = np.concatenate(coordinates), np.concatenate(classes)
    if not len(indices):
        raise ValueError("No hay muestras anotadas.")
    ys, xs = indices // width, indices % width
    columns = (width + TILE - 1) // TILE
    tile_ids = (ys // TILE) * columns + xs // TILE
    features = np.empty((len(indices), len(FEATURE_NAMES)), dtype=np.float32)
    for tile_id in np.unique(tile_ids):
        selected = np.flatnonzero(tile_ids == tile_id)
        x, y = int(tile_id % columns) * TILE, int(tile_id // columns) * TILE
        box = (x, y, min(width, x + TILE), min(height, y + TILE))
        feature_map = feature_tile(tc, swir, box)
        features[selected] = feature_map[ys[selected] - y, xs[selected] - x]
    return {"x": features, "classes": labels, "available": counts,
            "sampled": {str(i): int(np.sum(labels == i)) for i in (0, 1, 2)}}


def fit_heads(records):
    """Equilibra positivos/negativos dentro de cada escena de entrenamiento."""
    x = np.concatenate([record["x"] for record in records])
    models, seconds = {}, {}
    for head in ("afectado", "activo"):
        labels, weights = [], []
        for record in records:
            y = record["classes"] > 0 if head == "afectado" else record["classes"] == 2
            if np.unique(y).size != 2:
                raise ValueError(f"Faltan positivos o negativos de {head} en una escena de entrenamiento.")
            counts = np.bincount(y.astype(np.int32), minlength=2)
            labels.append(y)
            weights.append(1.0 / counts[y.astype(np.int32)])
        y, weight = np.concatenate(labels), np.concatenate(weights)
        weight *= len(weight) / weight.sum()
        start = time.perf_counter()
        model = RandomForestClassifier(**RF_CONFIG)
        model.fit(x, y, sample_weight=weight)
        models[head] = model
        seconds[head] = time.perf_counter() - start
    return models, seconds


def positive_probability(model, features):
    index = np.flatnonzero(model.classes_ == True)
    if len(index) != 1:
        raise ValueError("El clasificador no tiene ambas clases.")
    return model.predict_proba(features)[:, int(index[0])]


class JpegRefiner:
    """Inferencia opcional a resolucion original; carga solo modelos propios."""
    def __init__(self, model_path):
        bundle = joblib.load(Path(model_path))
        if bundle["feature_names"] != FEATURE_NAMES or bundle["schema"] != 1:
            raise ValueError("Modelo incompatible con las caracteristicas actuales.")
        self.models = bundle["models"]
        self.threshold = bundle["threshold"]

    def detect(self, true_color, swir, valid=None):
        validate_images(true_color, swir)
        height, width = true_color.shape[:2]
        if valid is None:
            valid = np.ones((height, width), dtype=bool)
        if valid.shape != (height, width) or valid.dtype != np.bool_:
            raise ValueError("Mascara de validez incompatible.")
        result = {head: np.zeros((height, width), dtype=bool) for head in self.models}
        for box in tiles(height, width):
            x0, y0, x1, y1 = box
            region = valid[y0:y1, x0:x1]
            if not region.any():
                continue
            features = feature_tile(true_color, swir, box)[region]
            for head, model in self.models.items():
                result[head][y0:y1, x0:x1][region] = (
                    positive_probability(model, features) >= self.threshold)
        return result

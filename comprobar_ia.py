"""Comprobaciones pequenas del experimento, sin leer ni modificar el proyecto."""
from pathlib import Path
import tempfile

import joblib
import numpy as np

from ai_refinement import (FEATURE_NAMES, JpegRefiner, feature_tile, fit_heads,
                           positive_probability, sample_scene)
from comparar_ia import confusion, metric


def main():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (112, 141, 3), dtype=np.uint8)
    whole = feature_tile(image, image, (0, 0, 141, 112))
    part = feature_tile(image, image, (47, 32, 104, 92))
    np.testing.assert_allclose(part, whole[32:92, 47:104], atol=1e-5)
    assert whole.shape == (112, 141, 30)
    assert np.isfinite(whole).all()
    gray = np.zeros_like(image) + 128
    assert np.isfinite(feature_tile(gray, gray, (0, 0, 141, 112))).all()

    # El resto sin anotar no puede entrar como negativo ni positivo.
    valid = np.zeros(image.shape[:2], dtype=bool)
    valid[20:90, 25:115] = True
    affected = np.zeros_like(valid)
    affected[20:90, 55:115] = True
    active = np.zeros_like(valid)
    active[20:90, 90:115] = True
    record = sample_scene(image, image, affected, active, valid, limit=100, seed=9)
    assert record['available'] == {'0': 2100, '1': 2450, '2': 1750}
    assert record['sampled'] == {'0': 100, '1': 100, '2': 100}
    outside = affected.copy()
    outside[~valid] = True
    again = sample_scene(image, image, outside, active, valid, limit=100, seed=9)
    np.testing.assert_array_equal(record['x'], again['x'])
    np.testing.assert_array_equal(record['classes'], again['classes'])

    counts = confusion(np.array([True, True, False]), np.array([True, False, True]))
    assert counts.tolist() == [1, 1, 1, 0]
    result = metric(counts, 4)
    assert result['f1'] == 0.5 and result['iou'] == 1 / 3
    assert result['coverage_pct'] == 75

    # Aprendizaje / serializacion / inferencia de dos cabezas booleanas.
    synthetic = image.copy()
    synthetic[~affected] = [0, 220, 20]
    synthetic[affected & ~active] = [100, 50, 30]
    synthetic[active] = [255, 70, 10]
    training = sample_scene(synthetic, synthetic, affected, active, valid, limit=100)
    models, _ = fit_heads([training, training])
    for head, target in [('afectado', training['classes'] > 0), ('activo', training['classes'] == 2)]:
        pred = positive_probability(models[head], training['x']) >= 0.5
        assert np.mean(pred == target) > 0.95
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
        path = Path(folder) / 'model.joblib'
        joblib.dump(dict(schema=1, models=models, feature_names=FEATURE_NAMES, threshold=0.5), path)
        result = JpegRefiner(path).detect(synthetic, synthetic, valid)
        assert set(result) == {'afectado', 'activo'}
        for pred in result.values():
            assert pred.dtype == np.bool_ and pred.shape == valid.shape
            assert not pred[~valid].any()
    print('Correcto: ventanas con halo, regiones ignoradas, confusion, modelo e inferencia.')


if __name__ == '__main__':
    main()

import json
import numpy as np


def evaluate_mask(prediction, reference, valid):
    """Compara máscaras booleanas H×W solo donde valid=True."""
    arrays = [
        np.asarray(prediction),
        np.asarray(reference),
        np.asarray(valid),
    ]
    prediction, reference, valid = arrays

    if reference.ndim != 2:
        raise ValueError("Las máscaras deben tener dos dimensiones.")

    if not all(a.shape == reference.shape for a in arrays):
        raise ValueError("Las tres máscaras deben tener el mismo tamaño.")

    if not all(a.dtype == np.bool_ for a in arrays):
        raise ValueError("Las tres máscaras deben ser booleanas.")

    valid_pixels = int(valid.sum())
    if valid_pixels == 0:
        raise ValueError("No hay píxeles válidos para evaluar.")

    p = prediction[valid]
    r = reference[valid]

    tp = int(np.count_nonzero(p & r))
    fp = int(np.count_nonzero(p & ~r))
    fn = int(np.count_nonzero(~p & r))
    tn = int(np.count_nonzero(~p & ~r))

    def divide(numerator, denominator):
        return numerator / denominator if denominator else None

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "iou": divide(tp, tp + fp + fn),
        "valid_pixels": valid_pixels,
        "coverage_pct": 100 * valid_pixels / valid.size,
    }


if __name__ == "__main__":
    # Ejemplo pequeño con resultado conocido.
    reference = np.array([
        [True, False],
        [False, True],
    ])

    prediction = np.array([
        [True, True],
        [True, False],
    ])

    valid = np.array([
        [True, True],
        [False, True],
    ])

    result = evaluate_mask(prediction, reference, valid)

    # Debe haber un acierto positivo, una falsa alarma y una omisión.
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["iou"] == 1 / 3

    # Cambiar un píxel ignorado NO debe alterar las métricas.
    modified = prediction.copy()
    modified[1, 0] = False
    assert evaluate_mask(modified, reference, valid) == result

    # No detectar nada debe dar recall y F1 cero en este ejemplo.
    empty = evaluate_mask(
        np.zeros_like(prediction), reference, valid
    )
    assert empty["recall"] == 0
    assert empty["f1"] == 0

    print("Comprobaciones del evaluador superadas.")
    print(json.dumps(result, indent=2))
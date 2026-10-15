import io
import json
import zipfile

import numpy as np

from pipeline import run_all_layers
from results_io import date_from_names, load_analysis, save_analysis, zip_bytes


def test_guardar_y_cargar(tmp_path, synthetic_scene):
    tc, swir = synthetic_scene
    res = run_all_layers(tc, swir, name="sintetica")
    save_analysis(res, tmp_path, {"titulo": "Sintética"})
    back = load_analysis(tmp_path)
    assert back["meta"]["titulo"] == "Sintética"
    assert back["bases"]["wildfire"] is None and back["bases"]["tc"] is not None
    for lid, r in res["layers"].items():
        assert np.array_equal(back["layers"][lid]["mask"], r["mask"])
        assert back["layers"][lid]["metrics"] == r["metrics"]


def test_zip(synthetic_scene):
    tc, swir = synthetic_scene
    res = run_all_layers(tc, swir)
    z = zipfile.ZipFile(io.BytesIO(zip_bytes(res)))
    names = set(z.namelist())
    assert "metrics.json" in names and "masks/burn.png" in names
    m = json.loads(z.read("metrics.json"))
    assert "hectareas" not in json.dumps(m) and m["capas"]["burn"]["metricas"]["area_px"] >= 0


def test_fecha():
    assert date_from_names(["2026-08-08-00_00_x_SWIR.jpg"]) == "8 de agosto de 2026"
    assert date_from_names(["sin_fecha.jpg"]) is None

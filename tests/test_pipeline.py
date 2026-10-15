import glob

import numpy as np
import pytest

from layers import LAYERS, Layer
from pipeline import mask_metrics, run_all_layers


def test_sin_wildfires_ejecuta_todas_las_capas(synthetic_scene):
    tc, swir = synthetic_scene
    seen = []
    res = run_all_layers(tc, swir, on_layer=lambda layer, r: seen.append(layer.id))
    assert seen == [layer.id for layer in LAYERS]
    assert res["bases"]["wildfire"] is None
    for layer in LAYERS:
        r = res["layers"][layer.id]
        assert r["status"] == "ok", (layer.id, r["error"])
        assert r["mask"].dtype == np.bool_ and r["mask"].shape == swir.shape[:2]
        extra = {"perimetro_exterior_px"} if layer.id == "perimeter" else set()
        assert set(r["metrics"]) == {"area_px", "pct_imagen", "regiones", "perimetro_px"} | extra
        assert r["time_s"] >= 0
    # sin hectáreas ni km en ningún sitio
    assert not any("ha" in k or "km" in k for r in res["layers"].values() for k in r["metrics"])


def test_sin_true_color(synthetic_scene):
    _, swir = synthetic_scene
    res = run_all_layers(None, swir)
    assert res["layers"]["burn"]["status"] == "ok"


def test_una_capa_que_falla_no_tumba_las_demas(synthetic_scene):
    tc, swir = synthetic_scene

    def boom(ctx):
        raise RuntimeError("fallo provocado")

    layers = [LAYERS[0], Layer("rota", "Rota", "Siempre falla.", "#000000", "relleno", boom), *LAYERS[1:]]
    res = run_all_layers(tc, swir, layers=layers)
    assert res["layers"]["rota"]["status"] == "error"
    assert "fallo provocado" in res["layers"]["rota"]["error"]
    assert res["layers"]["rota"]["mask"] is None
    assert all(res["layers"][layer.id]["status"] == "ok" for layer in LAYERS)


def test_paso_comun_roto_solo_afecta_a_sus_dependientes(synthetic_scene, monkeypatch):
    import core.segmentation
    tc, swir = synthetic_scene
    monkeypatch.setattr(core.segmentation, "burn_details", lambda *a, **k: 1 / 0)
    res = run_all_layers(tc, swir)
    for lid in ("burn", "perimeter", "burn_active"):
        assert res["layers"][lid]["status"] == "error"
        assert "zona quemada" in res["layers"][lid]["error"]
    for lid in ("fire", "smoke", "water", "nodata"):
        assert res["layers"][lid]["status"] == "ok"


def test_dependencia_pesada_ausente_marca_no_disponible(synthetic_scene):
    tc, swir = synthetic_scene
    called = []
    heavy = Layer("ia", "IA", "Necesita un paquete que no existe.", "#000000", "relleno",
                  lambda ctx: called.append(1), requiere=("paquete_que_no_existe_xyz",))
    res = run_all_layers(tc, swir, layers=[heavy, LAYERS[0]])
    assert res["layers"]["ia"]["status"] == "no_disponible"
    assert "paquete_que_no_existe_xyz" in res["layers"]["ia"]["error"]
    assert not called
    assert res["layers"]["burn"]["status"] == "ok"


def test_metricas_en_pixeles():
    m = np.zeros((10, 10), bool)
    m[2:4, 2:4] = True
    m[7, 7] = True
    met = mask_metrics(m)
    assert met["area_px"] == 5 and met["regiones"] == 2 and met["pct_imagen"] == 5.0


@pytest.mark.skipif(not glob.glob("data/niebla/*SWIR*"), reason="sin data/niebla")
def test_misma_cicatriz_que_run_py():
    from core.ingestion import load_scene
    from core.segmentation import burn_details
    from core.spectral import bands, exclusion_masks, indices
    sc = load_scene("data/niebla")
    b = bands(sc)
    ind = indices(b)
    ref = burn_details(b, ind, exclusion_masks(b, ind))["mask"]
    res = run_all_layers(sc["tc"], sc["swir"], sc["wildfire"])
    assert np.array_equal(res["layers"]["burn"]["mask"], ref)

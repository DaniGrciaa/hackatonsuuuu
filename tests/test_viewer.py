import base64
import io
import json
import re

import numpy as np
from PIL import Image

from components.viewer import build_payload, to_html
from pipeline import run_all_layers


def _decode(src: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(src.split(",", 1)[1])))


def test_payload(synthetic_scene):
    tc, swir = synthetic_scene
    res = run_all_layers(tc, swir)
    res["layers"]["water"].update(status="error", mask=None, metrics=None, error="roto")
    p = build_payload(res, "prueba")
    assert [b["id"] for b in p["bases"]] == ["tc", "swir"]
    assert (p["ancho"], p["alto"]) == (swir.shape[1], swir.shape[0])
    by = {c["id"]: c for c in p["capas"]}
    assert by["water"]["estado"] == "error" and "mascara" not in by["water"]
    m = _decode(by["burn"]["mascara"])
    assert m.mode == "1" and m.size == (p["ancho"], p["alto"])
    assert "contornos" in by["perimeter"] and "puntos" in by["fire"]
    assert _decode(p["bases"][0]["src"]).format == "JPEG"


def test_lado_maximo_2048():
    big = np.zeros((1300, 2600, 3), np.uint8)
    res = {"scene": {"name": "x", "warnings": []}, "bases": {"tc": None, "swir": big, "wildfire": None},
           "layers": {"fire": {"status": "ok", "mask": np.zeros((1300, 2600), bool), "metrics": {}, "time_s": 0}}}
    res["layers"]["fire"]["mask"][5, 5] = True          # un foco de 1 px no debe perderse al reducir
    p = build_payload(res)
    assert max(p["ancho"], p["alto"]) == 2048
    assert np.asarray(_decode(p["capas"][0]["mascara"])).any()


def test_html_incrusta_json_sin_romper_el_script():
    p = {"titulo": "</script><b>x", "ancho": 1, "alto": 1, "bases": [], "capas": [], "avisos": []}
    html = to_html(p)
    assert "__PAYLOAD__" not in html
    raw = re.search(r'<script type="application/json" id="payload">(.*?)</script>', html, re.S).group(1)
    assert json.loads(raw)["titulo"] == "</script><b>x"
    assert "<script src" not in html and "http" not in raw  # sin dependencias externas

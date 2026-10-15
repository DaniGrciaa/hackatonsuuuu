import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def synthetic_scene():
    """Escena pequeña con vegetación, una cicatriz rojiza oscura, un embalse y unos focos."""
    rng = np.random.default_rng(0)
    h, w = 240, 320
    noise = lambda: rng.integers(-8, 9, (h, w, 3))  # noqa: E731
    swir = np.clip(np.array([70, 150, 60]) + noise(), 0, 255)       # R=B12, G=B8A, B=B04: vegetación verde
    yy, xx = np.mgrid[:h, :w]
    scar = (yy - 120) ** 2 / 60 ** 2 + (xx - 150) ** 2 / 90 ** 2 < 1
    swir[scar] = np.clip(np.array([115, 55, 45]) + noise()[scar], 0, 255)
    lake = (yy - 40) ** 2 + (xx - 280) ** 2 < 18 ** 2
    swir[lake] = (4, 6, 8)
    for y, x in ((120, 60), (100, 70), (140, 65)):                  # focos en el borde de la cicatriz
        swir[y - 1:y + 2, x - 1:x + 2] = (255, 70, 30)
    tc = np.clip(np.array([60, 90, 50]) + noise(), 0, 255)
    tc[scar] = np.clip(np.array([50, 40, 35]) + noise()[scar], 0, 255)
    tc[lake] = (20, 30, 50)
    return tc.astype(np.uint8), swir.astype(np.uint8)

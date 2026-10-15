import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation import evaluate_mask


def cargar_mascara(ruta):
    with Image.open(ruta) as imagen:
        datos = np.array(imagen)

    if datos.ndim != 2:
        raise ValueError(
            f"{ruta}: se necesita una máscara de un canal, "
            "no una imagen RGB ni un overlay."
        )

    valores = set(np.unique(datos).tolist())
    if not valores.issubset({0, 1, 255}):
        raise ValueError(
            f"{ruta}: máscara no binaria. Valores: {sorted(valores)[:10]}"
        )

    return datos > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--referencias", required=True)
    parser.add_argument("--pred-afectado", required=True)
    parser.add_argument("--pred-activo", required=True)
    parser.add_argument("--salida", required=True)
    args = parser.parse_args()

    referencias = Path(args.referencias)

    valid = cargar_mascara(referencias / "valid.png")
    ref_afectado = cargar_mascara(referencias / "affected.png")
    ref_activo = cargar_mascara(referencias / "active.png")

    pred_afectado = cargar_mascara(args.pred_afectado)
    pred_activo = cargar_mascara(args.pred_activo)

    resultados = {
        "escena": referencias.name,
        "alcance": "Solo regiones anotadas y válidas",
        "afectado": evaluate_mask(
            pred_afectado, ref_afectado, valid
        ),
        "activo": evaluate_mask(
            pred_activo, ref_activo, valid
        ),
    }

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    contenido = json.dumps(resultados, indent=2, ensure_ascii=False)
    salida.write_text(contenido, encoding="utf-8")

    print(contenido)
    print(f"\nInforme guardado en: {salida}")


if __name__ == "__main__":
    main()
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
    parser.add_argument("--pred-afectado")
    parser.add_argument("--pred-activo")
    parser.add_argument("--validez-imagen")
    parser.add_argument("--salida", required=True)
    args = parser.parse_args()
    if not args.pred_afectado and not args.pred_activo:
        parser.error("Indica al menos --pred-afectado o --pred-activo")

    referencias = Path(args.referencias)

    valid = cargar_mascara(referencias / "valid.png")
    if args.validez_imagen:
        image_valid = cargar_mascara(args.validez_imagen)
        if image_valid.shape != valid.shape:
            raise ValueError("Validez de imagen y referencia tienen dimensiones diferentes")
        valid &= image_valid

    resultados = {
        "escena": referencias.name,
        "alcance": "Solo regiones anotadas y válidas",
        "afectado": None,
        "activo": None,
    }
    for head, argument, filename in (
        ("afectado", args.pred_afectado, "affected.png"),
        ("activo", args.pred_activo, "active.png"),
    ):
        if argument:
            resultados[head] = evaluate_mask(
                cargar_mascara(argument), cargar_mascara(referencias / filename), valid
            )

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    contenido = json.dumps(resultados, indent=2, ensure_ascii=False)
    salida.write_text(contenido, encoding="utf-8")

    print(contenido)
    print(f"\nInforme guardado en: {salida}")


if __name__ == "__main__":
    main()

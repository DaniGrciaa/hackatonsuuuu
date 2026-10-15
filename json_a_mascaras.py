import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ETIQUETAS = {
    "fondo": 0,
    "superficie_afectada": 1,
    "fuego_activo": 2,
    "dudoso": 255,
}


def convertir(archivo_json, carpeta_salida):
    datos = json.loads(archivo_json.read_text(encoding="utf-8"))

    ancho = int(datos["imageWidth"])
    alto = int(datos["imageHeight"])
    formas = datos["shapes"]

    # Comprobamos las anotaciones antes de generar archivos.
    for forma in formas:
        etiqueta = forma["label"]
        if etiqueta not in ETIQUETAS:
            raise ValueError(f"Etiqueta desconocida: {etiqueta!r}")
        if forma.get("shape_type", "polygon") != "polygon":
            raise ValueError("Este conversor espera polígonos.")
        if len(forma["points"]) < 3:
            raise ValueError("Hay un polígono con menos de tres puntos.")

    # Todo comienza como desconocido, NO como fondo.
    clases = Image.new("L", (ancho, alto), 255)
    dibujo = ImageDraw.Draw(clases)

    # En caso de solapamiento: activo prevalece sobre afectado;
    # dudoso prevalece sobre cualquier otra etiqueta.
    for etiqueta, valor in ETIQUETAS.items():
        for forma in formas:
            if forma["label"] == etiqueta:
                puntos = [tuple(punto) for punto in forma["points"]]
                dibujo.polygon(puntos, fill=valor)

    clases = np.asarray(clases)

    mascaras = {
        "affected": (clases == 1) | (clases == 2),
        "active": clases == 2,
        "valid": clases != 255,
    }

    destino = carpeta_salida / archivo_json.stem
    destino.mkdir(parents=True, exist_ok=True)

    for nombre, mascara in mascaras.items():
        imagen = Image.fromarray(mascara.astype(np.uint8) * 255)
        imagen.save(destino / f"{nombre}.png")

    print(f"\n{archivo_json.name}: {ancho} x {alto}")
    for etiqueta, valor in ETIQUETAS.items():
        print(f"  {etiqueta}: {int((clases == valor).sum())} píxeles")
    print(f"Guardado en: {destino}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Uso: python json_a_mascaras.py CARPETA_JSON CARPETA_SALIDA"
        )

    origen = Path(sys.argv[1])
    salida = Path(sys.argv[2])
    archivos = sorted(origen.glob("*.json"))

    if not archivos:
        raise SystemExit(f"No hay archivos JSON en {origen}")

    for archivo in archivos:
        convertir(archivo, salida)
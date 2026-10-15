"""Regenera predicciones y referencias, y evalua las escenas anotadas."""
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    commands = [
        ["run.py", "data", "--out", "salida"],
        ["json_a_mascaras.py", "anotaciones/desarrollo", "referencias/desarrollo"],
        ["evaluar_proyecto.py"],
    ]
    for arguments in commands:
        print("Ejecutando:", " ".join(arguments), flush=True)
        subprocess.run([sys.executable, "-B", *arguments], cwd=root, check=True)
    print("Listo: resultados_evaluacion/resumen_desarrollo.json")


if __name__ == "__main__":
    main()

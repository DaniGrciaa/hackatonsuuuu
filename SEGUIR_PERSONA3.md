# Continuar como persona 3

Desde la carpeta principal del proyecto, en PowerShell:

```powershell
.\.venv_persona3\Scripts\python.exe preparar_persona3.py
```

Este comando ejecuta los dos detectores, convierte los JSON de desarrollo y evalua solo las escenas que tienen referencia humana. No entrena ni ajusta parametros.

## Carpetas

| Contenido | Carpeta |
|---|---|
| JPG originales del 21 de julio | `data/extra_mierla` |
| JPG originales del 26 de julio | `data/avila_madrid` |
| JPG originales del 8 de agosto | `data/niebla` |
| Poligonos LabelMe de desarrollo | `anotaciones/desarrollo` |
| Referencias PNG derivadas de esos poligonos | `referencias/desarrollo` |
| Predicciones y visualizaciones | `salida/<escena>` |
| Informe conjunto | `resultados_evaluacion/resumen_desarrollo.json` |

Las tres escenas ya se usaron para desarrollar los detectores. No son un test independiente. Una escena nueva debe tener otro grupo y split en `escenas.json`.

## Formato comun

`salida/<escena>/affected.png`, `active.png` y `valid.png` son PNG monocanal 0/255 a resolucion original. `active.png` contiene candidatos activos, incluidos los que aparecen fuera de la cicatriz. `valid.png` de prediccion excluye solo logo/nodata; `valid.png` de referencia delimita lo anotado con seguridad. El evaluador combina ambas.

`mascara_quemado.png`, `perimetro_*.png` y `capa_focos.png` se conservan a resolucion de trabajo para compatibilidad y visualizacion. No se usan como mascaras de evaluacion.

El resultado declara dimensiones como `[alto, ancho]` y estado de cada detector. Si falla o falta actividad, su estado es `no_disponible`; una mascara vacia con estado `disponible` es un resultado valido sin candidatos.

## Siguiente tarea

1. Ya hay anotaciones del 21 de julio. Revisa el informe de ambas cabezas.
2. Abre en LabelMe el True Color de `data/avila_madrid`, con el SWIR como apoyo.
3. Usa `fondo`, `superficie_afectada`, `fuego_activo` y `dudoso`. Guarda el JSON en `anotaciones/desarrollo`.
4. Ejecuta de nuevo `preparar_persona3.py`. La segunda escena pasara de `sin_referencia` a `evaluado`.

No etiquetar equivale a desconocido, nunca a fondo. Las metricas actuales cubren solo poligonos anotados; no prueban calidad de todo el perimetro. Prioriza tambien bordes y negativos dificiles. Wildfires es apoyo visual, no verificacion independiente.

Los nombres internos heredados NBR/B12 son rasgos de color de JPEG, no bandas recuperadas ni medidas fisicas. Los scripts `tools_c` son experimentos historicos con TIFF y no forman parte de esta rutina JPG.

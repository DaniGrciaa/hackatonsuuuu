# Entrega de Persona 3

Estado: anotaciones y evaluacion terminadas para las tres escenas de desarrollo. Experimento de Random Forest completado. Se conservan los detectores actuales de Personas 1 y 2.

## Resultado de los detectores actuales

| Escena | F1 afectado | Precision afectado | Recall afectado | F1 activo | Precision activo | Recall activo | Cobertura anotada |
|---|---:|---:|---:|---:|---:|---:|---:|
| 21 julio / extra_mierla | 99.71% | 100.00% | 99.42% | 42.20% | 100.00% | 26.74% | 9.71% |
| 26 julio / avila_madrid | 93.36% | 92.01% | 94.75% | 47.70% | 100.00% | 31.32% | 2.98% |
| 8 agosto / niebla | 98.84% | 99.86% | 97.84% | 65.27% | 95.72% | 49.52% | 5.28% |

Las metricas se calculan por pixel. No son porcentajes de incendios detectados ni de confianza. La referencia afectada incluye la referencia activa.

## Experimento de IA

Dos Random Forest con caracteristicas de color y textura de TC/SWIR JPG, sin bandas originales ni Wildfires como entrada. Tres rondas: entrenar con dos grupos de adquisicion y evaluar en el tercero, a resolucion original y solo en regiones anotadas. Configuracion y umbral identicos en las tres rondas. No hubo particion aleatoria de pixeles de una misma escena entre entrenamiento y evaluacion.

| Cabeza | F1 medio base | F1 medio RF | Conclusion |
|---|---:|---:|---|
| Afectado | 97.30% | 92.98% | Empeora 4.33 puntos; mantener Persona 1. |
| Activo | 51.72% | 65.97% | Mejora F1 14.24 puntos, con mas falsas alarmas; mantener Persona 2 como salida actual. |

Para activo, el RF obtiene precision de 68.71%, 43.04% y 61.64% por escena. Sus falsos positivos anotados son 163, 4581 y 840 pixeles, frente a 0, 0 y 37 del detector base. Por ello no supera la regla previa de precision minima del 90% por escena. El experimento muestra un intercambio entre detectar mas fuego y producir mas falsas alarmas, no una mejora en todos los criterios.

No se ha conectado la IA al pipeline ni exportado un modelo final para sustituir a los detectores. El modulo de inferencia queda disponible para futuros experimentos; no se necesita para la demo actual. No se probo SamGeo: se priorizo un clasificador ligero que permitiera cerrar la comparacion reproducible dentro del tiempo disponible.

## Evidencias de errores

- Base activo: 326, 3303 y 844 pixeles de fuego anotado no detectados en las tres escenas, respectivamente. La prioridad de Persona 2 es investigar esas omisiones.
- Base afectado en avila_madrid: 14564 pixeles de falsas alarmas y 9283 omisiones dentro de referencias validas. Persona 1 debe revisar esos errores antes de cambiar reglas generales.
- RF afectado: recortes con terreno y vegetacion marcados incorrectamente.
- RF activo: recortes con falsas alarmas, incluidas superficies y bordes sin fuego anotado.

Los recortes RF de la carpeta `errores` son ejemplos de ventanas con errores, no un ranking de los mayores errores ni evidencia sobre zonas sin anotar. Magenta = FP; cian = FN. El JSON registra cajas en coordenadas originales.

## Archivos y repeticion

- `anotaciones/desarrollo`: JSON humanos con las cuatro etiquetas exactas.
- `referencias/desarrollo`: PNG binarios de afectado, activo y validez humana.
- `evaluation.py`, `evaluar_png.py`, `evaluar_proyecto.py`: evaluadores de las mascaras.
- `resultados_evaluacion/resumen_desarrollo.json`: resultados actuales; puede actualizarse al ejecutar el pipeline.
- `resultados_evaluacion/ia_persona3_20261015_01`: comparacion congelada, CSV, informe, recortes, copias del punto de partida y hashes.
- `comparar_ia.py`, `ai_refinement.py`, `PERSONA3_IA.md`: experimento reproducible e instrucciones.

Para reevaluar una nueva version de los detectores manteniendo las referencias:

```powershell
.\.venv_persona3\Scripts\python.exe preparar_persona3.py
```

Para repetir el experimento RF sobre esas nuevas predicciones:

```powershell
.\.venv_persona3\Scripts\python.exe -B comparar_ia.py
```

Esto crea una carpeta nueva y conserva la comparacion anterior. No mezclar una revision de etiquetas con una revision del detector al afirmar que el detector ha mejorado.

## Limites para la demo y el jurado

Las referencias cubren solo 2.98%-9.71% de cada escena y son regiones seleccionadas manualmente. Las tres escenas ya se han usado en el desarrollo del sistema: no existe un test independiente. Las etiquetas de actividad son interpretaciones visuales de composiciones JPG, no confirmaciones independientes en campo. La validacion por grupos del RF reduce la fuga de etiquetas, pero no elimina el sesgo de desarrollo previo.

La comparacion es entre sistemas completos: el RF calcula caracteristicas a resolucion original; el detector base reduce la escena grande antes de recuperar la resolucion de salida. El experimento no permite atribuir las diferencias exclusivamente al clasificador.

Los tiempos del experimento (aproximadamente 50 segundos de entrenamiento total de las seis cabezas y 54 segundos de evaluacion de las regiones anotadas, aparte de preparacion) no representan la inferencia RF de una imagen completa. No afirmar que la IA sea mas rapida que los detectores actuales.

Persona 4 puede continuar utilizando las mascaras `salida/<escena>/affected.png`, `active.png`, `valid.png`, booleanas en origen y exportadas monocanal 0/255 a resolucion original. Los overlays y `capa_focos.png` son visualizaciones, no las entradas de la evaluacion.

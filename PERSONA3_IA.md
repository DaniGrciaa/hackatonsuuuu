# Experimento de IA de Persona 3

Entrada: JPG/JPEG RGB True Color y SWIR, alineados y del mismo tamano. Wildfires no se usa como entrada. Las caracteristicas son de color y textura de composiciones visuales, no bandas ni indices fisicos.

## Ejecutar

Desde la carpeta del proyecto, con scikit-learn instalado en el entorno de Persona 3:

```powershell
.\.venv_persona3\Scripts\python.exe -B comparar_ia.py
```

Se crea una carpeta nueva `resultados_evaluacion/ia_FECHA_HORA`. No se modifican los detectores, anotaciones ni predicciones existentes.

## Que hace

1. Comprueba que cada escena tenga TC, SWIR, referencias humanas y ambas predicciones disponibles.
2. Guarda copias de anotaciones, referencias, predicciones base y codigo de los detectores; registra SHA256 de esos archivos y de los JPG de entrada.
3. Extrae hasta 8.000 muestras de cada clase (fondo, afectado sin activo, activo) por escena. Las regiones desconocidas/dudosas y logo/nodata se excluyen.
4. Usa 30 caracteristicas: RGB, HSV, Lab, medias y desviaciones de gris en ventanas de 7 y 31 pixeles, y cromaticidad de ambos JPG. Calcula todo a resolucion original por ventanas con halo de 15 pixeles; no pierde frentes finos al reducir la escena completa.
5. Entrena dos Random Forest independientes con 96 arboles, profundidad maxima 16, hoja minima de 3 y umbral fijo 0.5. Equilibra positivos/negativos y contribucion de cada escena mediante pesos de entrenamiento.
6. Deja fuera un grupo de adquisicion completo en cada ronda. Evalua todos los pixeles humanos validos de ese grupo, no solo las muestras elegidas para entrenamiento. Nunca divide aleatoriamente los pixeles de una misma escena entre entrenamiento y evaluacion.
7. Compara cada modelo con las mascaras base guardadas, en exactamente los mismos pixeles validos y la misma resolucion.
8. Comprueba al terminar que las entradas no hayan cambiado durante el experimento. Si cambian, la comparacion falla y debe repetirse con entradas estables.

## Resultados

- `COMPARACION.md`: tabla y decision por cabeza.
- `comparacion.json`: TP, FP, FN, TN, precision, recall, F1, IoU, cobertura, tiempos, configuracion, versiones y escenas de entrenamiento de cada ronda.
- `comparacion.csv`: tabla para compartir.
- `errores/`: recortes de errores del RF dentro de regiones anotadas. Magenta = falsa alarma; cian = omision. Coordenadas originales en el JSON.
- `punto_partida/` y `huellas_entrada.json`: entradas congeladas y hashes. Los JPG se identifican por hash; no se duplican.
- `candidato_rf.joblib`: solo si alguna cabeza supera la regla de seleccion. Se entrena entonces con todas las escenas y requiere revision visual antes de integrarlo. Las metricas del informe corresponden a las rondas por grupos, NO a ese ajuste final con todas las escenas.

Regla fijada antes del experimento: subir al menos 1 punto porcentual de F1 medio por escena, no perder mas de 5 puntos en ninguna escena y mantener precision de al menos 90% en cada escena. Es una regla para seleccionar un candidato, no una garantia de funcionamiento con imagenes nuevas.

## Integracion opcional

`ai_refinement.py` contiene `JpegRefiner`. Solo cargar un modelo generado por el propio equipo:

```python
from ai_refinement import JpegRefiner

refiner = JpegRefiner("ruta/al/candidato_rf.joblib")
mascaras = refiner.detect(true_color_rgb, swir_rgb, valid=validez_imagen)
# Salidas bool H x W, a resolucion original, solo de las cabezas seleccionadas.
# Conservar el detector base para cualquier cabeza ausente.
```

Esta funcion recorre toda la imagen. Su tiempo debe medirse por separado antes de incorporarla a la demo. Los tiempos del informe recorren solo ventanas que contienen regiones anotadas y no permiten comparar velocidad de inferencia completa.

## Limites

Las referencias son parciales y de pocas escenas. Las tres escenas ya se usaron durante el desarrollo de los detectores y las etiquetas pudieron apoyarse en Wildfires. Esta validacion por grupos evita entrenar el RF con la escena evaluada, pero no convierte esas escenas en un test independiente ni prueba el perimetro completo. Mantener las etiquetas fijas durante la comparacion y documentar cualquier revision posterior.

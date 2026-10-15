# Comparacion Persona 3

Validacion de desarrollo por grupos de adquisicion, sobre referencias humanas parciales a resolucion original.

Umbral fijo 0.5. Random Forest: 96 arboles, profundidad maxima 16, hoja minima 3.

| Escena | Cabeza | F1 base | F1 RF | Precision RF | Recall RF |
|---|---|---:|---:|---:|---:|
| extra_mierla | afectado | 99.71% | 93.74% | 88.85% | 99.21% |
| extra_mierla | activo | 42.20% | 74.12% | 68.71% | 80.45% |
| avila_madrid | afectado | 93.36% | 89.28% | 81.73% | 98.37% |
| avila_madrid | activo | 47.70% | 53.86% | 43.04% | 71.97% |
| niebla | afectado | 98.84% | 95.91% | 95.10% | 96.73% |
| niebla | activo | 65.27% | 69.91% | 61.64% | 80.74% |

## Decision del experimento

- afectado: mantener detector base. F1 medio por escena: 97.30% -> 92.98%.
- activo: mantener detector base. F1 medio por escena: 51.72% -> 65.97%.

La regla previa exige subir al menos 1 punto de F1 medio, no perder mas de 5 puntos en ninguna escena y mantener precision >=90% en cada escena.
Los tiempos de evaluacion corresponden a pixeles anotados; no son tiempos de inferencia de toda la imagen.
Los recortes muestran FP en magenta y FN en cian, solo en regiones anotadas. Revisar antes de integrar.
Las tres escenas se usaron para desarrollar el sistema: esto NO es una prueba independiente.
Wildfires no se usa como entrada del RF. Las etiquetas humanas pueden haberse apoyado en esa visualizacion.

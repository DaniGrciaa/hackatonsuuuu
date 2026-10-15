# Detección de área quemada en Sentinel-2 (III Malackathon, reto Indra Space)

Parte **C** del equipo: delimita la cicatriz del incendio y su perímetro a partir de imágenes Sentinel-2
de un solo instante (sin foto de antes), y genera la salida visual y las estadísticas.
Los focos activos (A), el banco de pruebas (B) y la app (D) van en sus propios módulos.

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate      # o: uv venv .venv
pip install -r requirements.txt                          # numpy, scipy, opencv-python-headless, scikit-image, pillow
```

Sin redes neuronales, sin GPU y sin descargas.

## Uso

```bash
python run.py <carpeta> --out salida/ [--m-per-px 10] [--variant v3]
```

- Si `<carpeta>` contiene imágenes, se trata como una escena. Si contiene subcarpetas, **cada subcarpeta es una escena**.
- Basta con la imagen **SWIR**. La True Color y la Wildfire son opcionales.
- El tipo de cada imagen se deduce del nombre (`swir`, `true`/`tc`, `wildfire`/`fire`) y, si el nombre no sirve, del contenido.
- `--m-per-px`: metros por píxel de la imagen **original**. Si se indica, el JSON incluye hectáreas y km de perímetro. Si no se conoce, solo se dan píxeles y porcentaje: no inventamos escala.

Salida por escena en `salida/<escena>/`:

| Archivo | Contenido |
|---|---|
| `perimetro_swir.png`, `perimetro_tc.png` | Imagen original con el área quemada sombreada, el perímetro en amarillo, los huecos (embalses) en azul y los focos en rojo |
| `mascara_quemado.png` | Máscara binaria (blanco = quemado) |
| `capa_focos.png` | Capa RGBA transparente con los focos de A (vacía si `core/fire.py` no existe) |
| `mascaras_descarte/*.png` | Agua, humo, tierra clara, nube y ciudad, por separado (para la vista explicable) |
| `resultado.json` | % quemado, píxeles, nº de manchas y huecos, perímetro, polígonos, umbrales usados, manchas descartadas y por qué, tiempo y avisos |

Cada módulo también se ejecuta por separado y guarda imágenes de depuración en `debug/<escena>/`:
`python -m core.ingestion data/niebla`, `python -m core.spectral …`, `python -m core.segmentation …`
(este último dibuja todas las variantes lado a lado).

Tarda **entre 0,2 y 2,6 s por escena** en un portátil.

## Cómo funciona

### 1. Carga (`core/ingestion.py`)
- **Identificación por contenido:** la SWIR tiene mucho verde saturado (vegetación). La Wildfire parece una True Color, pero con el azul más apagado (p99 del azul 121–129 frente a 182–195 en las True Color) y píxeles naranja. Ha acertado en las 8 imágenes de ejemplo.
- **Tamaño de trabajo:** las imágenes con un lado mayor de 2600 px se reducen para ir rápido. Los polígonos se devuelven en píxeles de la imagen original.
- **Zonas no válidas:** se excluyen el logo de Copernicus (se detecta solo, buscando letras blancas en la esquina) y las zonas sin datos (negro en todas las imágenes y pegado al borde).

### 2. Bandas e índices (`core/spectral.py`)
- La SWIR de Copernicus es R = B12, G = B8A y B = B04. La True Color es R = B04, G = B03 y B = B02.
- **Cada banda se normaliza con los percentiles 1 y 99 de su propia escena**, porque el brillo cambia mucho de una escena a otra (la True Color de Niebla está muy velada). Ningún umbral está pensado para valores 0–255 de una escena concreta.
- **Índices:**
  - `NBR = (B8A − B12)/(B8A + B12)`: bajo en lo quemado.
  - `RED_DOM = B12 − max(B8A, B04)`: mide cuánto domina el rojo (el marrón rojizo de la cicatriz).
  - También brillo, saturación y NDVI.

### 3. Máscaras de descarte
| Máscara | Regla | Fiabilidad |
|---|---|---|
| Agua | Histéresis: núcleo muy negro en los tres canales (cola del 3 %) que crece por lo oscuro (cola del 10 %) | Buena con embalses; confunde algo de ceniza muy negra |
| Humo | Turquesa en la SWIR: B8A y B04 claramente por encima de B12 | Coge el humo denso |
| Tierra clara | Muy brillante en la SWIR | Razonable |
| Nube | Muy brillante y poco saturada; si hay True Color, también clara allí | Sin escena nubosa para comprobarla bien |
| Ciudad | Contraste fino denso | **Experimental:** también marca parcelas. No se usa para restar área |

### 4. Cicatriz (`core/segmentation.py`), variante por defecto **v3**
1. **Candidatos:** fuera el agua. Dentro del resto, lo oscuro (Otsu del brillo) con NBR bajo (Otsu del NBR).
2. **Semillas:** de esos candidatos, lo claramente rojizo (Otsu de `RED_DOM`, nunca por debajo de 0).
3. **Crecimiento:** desde las semillas se crece, como mucho un 2 % de la diagonal, por lo *muy* oscuro (segundo Otsu del brillo). Así entra la ceniza marrón oscura pegada a la cicatriz, pero no el bosque oscuro que no toca ninguna semilla.
4. **Evidencia de quemado:** una mancha solo cuenta si su NBR mediano es menor que −0,15. Es la física del fenómeno: en lo quemado, el SWIR (B12) supera al infrarrojo cercano (B8A).
5. **Prior espacial:** se quedan las manchas grandes y las pegadas al humo o al fuego. El matorral seco suelto se descarta.
6. **Huecos:** se rellenan los huecos pequeños (claros de ceniza) y se vuelve a restar el agua, así que los embalses de dentro no cuentan como quemado.
7. **Perímetro:** contornos con huecos (`cv2.findContours`, `RETR_CCOMP`), simplificados.

Lo que se descarta (manchas pequeñas, sueltas o sin evidencia) va en `resultado.json → descartados`, con área, centroide y motivo, para que el detector de focos pueda revisarlo.

## Por qué así: decisiones y medidas

| Variante | Ávila-Madrid | Niebla | Escena sin incendio | Escena extra |
|---|---|---|---|---|
| v0: NBR + brillo con Otsu | 37,8 % | 35,4 % | 45,4 % | — |
| v1: + tono rojizo | 4,3 % | 6,7 % | 16,2 % | — |
| v2: + prior espacial | 4,1 % | 6,3 % | 8,5 % | 9,1 % (pierde una cicatriz) |
| **v3: histéresis + evidencia** | **5,7 %** | **7,8 %** | **0,0 %** | **12,0 %** |

- **El NBR a secas no basta.** Sin foto de antes, todo terreno seco tiene NBR bajo, y v0 marca más de un tercio de la escena.
- **El tono rojizo como corte duro pierde cicatrices oscuras.** En la escena extra, v2 se dejaba una segunda cicatriz marrón oscura (RED_DOM −0,08). La histéresis la recupera sin dejar entrar el bosque oscuro: medido en Niebla y Ávila-Madrid, lo que se colaba era más claro que la ceniza (brillo mediano 0,33–0,41 frente a 0,24–0,25).
- **Otsu siempre parte la escena en dos, haya incendio o no.** En una escena de junio sin incendio (`Browser_images-9`), v2 marcaba un 8,5 % de matorral pardo. Las cicatrices reales tienen NBR mediano −0,30 / −0,52, y ese matorral 0,00, así que se corta en −0,15. Ahora esa escena da 0,0 %.
- **Solo SWIR:** el resultado es idéntico con o sin True Color (Niebla 7,79 % en los dos casos). Hoy la True Color solo se usa para la máscara de nubes y para dibujar.
- **Calibración True Color ↔ SWIR:** B04 aparece en las dos imágenes (rojo de la True Color, azul de la SWIR). Una recta robusta los relaciona con correlación 0,93–0,95: azul SWIR ≈ 1,26·x − 101 (Ávila-Madrid), 1,48·x − 140 (Niebla) y 1,11·x − 81 (extra). Confirma que las imágenes están alineadas. La hipótesis de que esa recta también vale para el verde y el rojo de la SWIR **no se pudo comprobar**, porque son bandas que la True Color no tiene.

## Limitaciones conocidas (honestas)

- **No hay verdad humana.** Los porcentajes son plausibles a ojo, no medidos con IoU.
- **La escena extra ya no es ciega.** Se vio su fallo con v2 y se corrigió con una regla general, ajustada solo con Niebla, Ávila-Madrid y la escena sin incendio.
- **Humo denso:** la cicatriz bajo humo espeso puede quedar fuera (por ejemplo, la zona gris oscura al norte de la escena extra).
- **Bordes:** en Niebla, v3 se desborda un poco al sur del anillo de fuego.
- **Ciudad:** la máscara no es fiable. Lo que mantiene Madrid fuera es el prior espacial y la evidencia de NBR.
- **Cambio de producto:** los umbrales `COMP_NBR_MAX = −0,15` y `SEED_RED_MIN = 0` son fijos sobre índices normalizados. Están justificados físicamente y medidos en 4 escenas, pero conviene revisarlos si el producto SWIR usa otra combinación de bandas.

## Créditos y referencias

- Copernicus Browser / Sentinel Hub, para las imágenes (ESA, Copernicus).
- NBR: Key & Benson, *Landscape Assessment* (FIREMON), USDA Forest Service, 2006.
- TangChao729, *Burned_Area_Segmentation* (GitHub). Según su README, un índice espectral solo da un IoU de 0,40 frente a 0,73–0,82 de las redes. Es un argumento de por qué no basta con un índice (dato tomado de su README, no reproducido por nosotros).

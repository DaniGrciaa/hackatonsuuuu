# Detección de área quemada en Sentinel-2 (III Malackathon, reto Indra Space)

## Integracion lista para persona 3

Ver **SEGUIR_PERSONA3.md**. Ejecutar `.\.venv_persona3\Scripts\python.exe preparar_persona3.py` desde la raiz.

Se exportan `affected.png`, `active.png` y `valid.png` a resolucion original, monocanal 0/255, separados de los overlays de trabajo. `resultado.json` declara las coordenadas, archivos de entrada y disponibilidad de cada detector. `evaluar_proyecto.py` puntua solo escenas anotadas y registra las demas como `sin_referencia`.

La entrada y deteccion son exclusivamente JPEG. Los nombres internos espectrales no implican recuperacion fisica de bandas. Los experimentos TIFF/dNBR de `tools_c` quedan separados de la rutina JPG. Niebla, Avila y la escena extra ya se usaron durante desarrollo: no constituyen un test independiente.


Delimita la cicatriz del incendio y su perímetro, y detecta los focos de fuego activo, a partir de imágenes
Sentinel-2 de un solo instante (sin foto de antes). Genera la salida visual y las estadísticas.
El banco de pruebas (B) y la app (D) van en sus propios módulos.

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate      # o: uv venv .venv
pip install -r requirements.txt                          # numpy, scipy, opencv-python-headless, scikit-image, pillow
```

Sin redes neuronales, sin GPU y sin descargas.

## Uso

```bash
python run.py <carpeta> --out salida/ [--m-per-px 10] [--variant v4]
```

- Si `<carpeta>` contiene imágenes, se trata como una escena. Si contiene subcarpetas, **cada subcarpeta es una escena**.
- Basta con la imagen **SWIR**. La True Color y la Wildfire son opcionales.
- El tipo de cada imagen se deduce del nombre (`swir`, `true`/`tc`, `wildfire`/`fire`) y, si el nombre no sirve, del contenido.
- `--m-per-px`: metros por píxel de la imagen **original**. Si se indica, el JSON incluye hectáreas y km de perímetro. Si no se conoce, solo se dan píxeles y porcentaje: no inventamos escala.

Salida por escena en `salida/<escena>/`:

| Archivo | Contenido |
|---|---|
| `perimetro_swir.png`, `perimetro_tc.png` | Imagen a resolución de trabajo con el perímetro en amarillo, la superficie afectada **sin actividad** en amarillo, la superficie **con posible actividad** en naranja, los posibles focos en rojo (píxeles y círculos), los huecos en azul, leyenda y una línea de resumen (% afectado, % con y sin actividad, nº de focos) |
| `mascara_quemado.png` | Máscara binaria (blanco = quemado), resolución de trabajo |
| `mascara_actividad.png` | Superficie afectada **con** posible actividad: lo quemado a menos de ~1 % de la diagonal (≈30 px) de un foco, más los propios focos. Es una definición nuestra, no una medida |
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

### 2. Rasgos visuales de JPEG (`core/spectral.py`)
- Los canales RGB son representaciones visuales. Las claves internas B12/B8A/B04 son nombres heredados: no recuperan reflectancias ni bandas originales.
- **Cada canal de color se normaliza con los percentiles 1 y 99 de su propia escena**, porque el brillo cambia mucho de una escena a otra (la True Color de Niebla está muy velada). Ningún umbral está pensado para valores 0–255 de una escena concreta.
- **Índices:**
  - `NBR` es el nombre heredado de una diferencia normalizada entre canales de color, con escalado independiente. No es NBR radiometrico; se usa como rasgo empirico.
  - `RED_DOM = B12 − max(B8A, B04)`: mide cuánto domina el rojo (el marrón rojizo de la cicatriz).
  - Tambien brillo, saturacion y otra diferencia de canales llamada internamente NDVI; no es NDVI radiometrico.

### 3. Máscaras de descarte
| Máscara | Regla | Fiabilidad |
|---|---|---|
| Agua | Hueco del histograma de max(B12, B8A) con los valores originales: el agua forma un grupo muy oscuro separado de la tierra. Vale igual con un embalse que con media imagen de mar. Después, **todos los percentiles se calculan solo sobre tierra** | Buena con embalses y con mar (comprobado en dos escenas costeras) |
| Humo | **Denso:** turquesa en la SWIR (B8A y B04 claramente por encima de B12). **Fino:** gris en la True Color (saturación < Otsu, como mucho 0,15; medido 0,03–0,13 en humo frente a 0,20–0,51 en suelo, ciudad y monte), creciendo solo desde el humo denso o desde focos reales, sin nubes muy blancas | Buena en las 5 escenas con incendio; deja huecos en el humo más claro (sobre el mar en Tenerife). La cicatriz usa solo el denso (`smoke_dense`), así que este cambio no la altera |
| Tierra clara | Muy brillante en la SWIR | Razonable |
| Nube | Muy brillante y poco saturada; si hay True Color, también clara allí | Sin escena nubosa para comprobarla bien |
| Ciudad | Contraste fino denso | **Experimental:** también marca parcelas. No se usa para restar área |

### 4. Cicatriz (`core/segmentation.py`), variante por defecto **v4**
1. **Candidatos:** fuera el agua. Dentro del resto, lo oscuro (Otsu del brillo) con NBR bajo (Otsu del NBR, **nunca por debajo de 0**: es un criterio visual empirico que debe validarse con anotaciones).
2. **Semillas:** de esos candidatos, lo claramente rojizo (Otsu de `RED_DOM`, nunca por debajo de 0).
3. **Crecimiento:** desde las semillas se crece, como mucho un 2 % de la diagonal, por lo *muy* oscuro (segundo Otsu del brillo). Así entra la ceniza marrón oscura pegada a la cicatriz, pero no el bosque oscuro que no toca ninguna semilla.
4. **Evidencia de quemado:** una mancha solo cuenta si (a) con los valores originales del JPEG SWIR, el canal rojo (B12) supera al verde (B8A) de mediana, y (b) su "NBR" normalizado es < −0,15. Son reglas empíricas sobre los canales de color del JPEG, no una prueba física de quemado. La (a) hizo falta porque el valor normalizado por percentiles pierde el orden entre canales y dejaba pasar una zona de terreno seco como "cicatriz" en una escena de antes del incendio (marcaba un 21 %).
5. **Prior espacial:** se quedan las manchas grandes y las que están cerca del humo o del fuego **y además pegadas a la mancha principal** (un foco aislado en un campo no arrastra una "cicatriz" a su alrededor). El matorral seco suelto se descarta.
6. **Huecos:** se rellenan los huecos pequeños (claros de ceniza) y se vuelve a restar el agua, así que los embalses de dentro no cuentan como quemado.
7. **Perímetro:** contornos con huecos (`cv2.findContours`, `RETR_CCOMP`), simplificados.

Lo que se descarta (manchas pequeñas, sueltas o sin evidencia) va en `resultado.json → descartados`, con área, centroide y motivo, para que el detector de focos pueda revisarlo.

### 5. Focos de fuego activo (`core/fire.py`, nivel avanzado)
Es la idea de los algoritmos de satélite de detección de incendios (MODIS/VIIRS), adaptada a la SWIR de Copernicus. Un píxel es foco si cumple las tres condiciones:
1. Está entre el 0,5 % más brillante de la escena en B12 (canal rojo de la SWIR **sin recortar**, porque el fuego lo satura).
2. Es claramente naranja o rojo: `(R − max(G, B)) / R ≥ 0,40`. Así se descartan el suelo claro, las minas y las nubes (pálidos) y los campos pardos (más verde: G≈160 frente a 106–139 en los focos reales).
3. Destaca sobre su entorno: supera la media del fondo en más de 3 desviaciones, en una ventana de 61 px. **El fondo se calcula sin los píxeles calientes**; si no, un frente largo se compara consigo mismo y no destaca. Con ese cambio, el recall en Niebla pasó del 3 % al 54 %.

Cada foco se clasifica por **ubicación**: "frente de la cicatriz" (confianza alta), "junto al humo" (media) o "aislado" (baja). Esa confianza dice dónde está, no cuánto se parece a un fuego real. El JSON separa el **área quemada con actividad** (a menos de ~30 px de un foco) de la **quemada sin actividad**.

**Lo que NO hacemos todavía:** no filtramos los focos con las máscaras de nube, ciudad o tierra clara (la tierra clara no sirve, porque el fuego también es brillante: el 57 % de los focos reales de la escena extra caen en ella), no clasificamos por intensidad ni por tipo, y la confianza no se calibra.

Medido **por focos** (cada mancha es un objeto, con 5 px de tolerancia):

| Referencia | Focos detectados / de referencia | Precisión | Recall |
|---|---|---|---|
| Niebla, Wildfire | 61 / 86 | 0,75 | 0,58 |
| Extra, Wildfire | 53 / 27 | 0,64 | 0,96 |
| Extra, marcas a mano de Jesús (solo zonas anotadas) | — | 1,00 | 0,82 |
| Práctica 26-07, regla física de QuickFire (B11 + B12 > 1, bandas crudas) | 264 / 769 | 0,82 | 0,29 |

**Ajuste de sensibilidad (barrido con las marcas a mano de Jesús en las 3 escenas, la Wildfire y 4 escenas sin incendio):**

| `RED_RATIO` / `Z_CTX` | Marcas de Jesús: recall por píxel | Zonas de fuego anotadas halladas | Escenas sin incendio: píxeles falsos |
|---|---|---|---|
| 0,30 / 2,5 | 0,39 | 100 % | 0 · 30 · 34 · 235 |
| 0,35 / 3,0 | 0,37 | 100 % | 0 · 5 · 5 · 33 |
| **0,40 / 3,0 (elegido)** | 0,36 | 100 % | **0 · 1 · 1 · 9** |

Más sensibilidad no encuentra ningún foco anotado más; solo engorda los ya hallados y multiplica los falsos (revisados a ojo: suelo rojizo, tejados y bordes de nube). Los 2 puntos que parecen fuego real en la escena del 9 de junio (la Wildfire también los pinta) se detectan con 0,40 / 3,0.

**Escenas sin incendio** (`python tools_c/negativas.py`): 0, 1, 1 y 9 píxeles de fuego, y 0 % de cicatriz en las cuatro. Es el valor intermedio de un barrido de `RED_RATIO` (0,30 daba 0 / 16 / 16 / 108 píxeles y 0,45 daba 0 / 0 / 0 / 3, pero bajaba el recall frente a las marcas de Jesús a 0,68). **Se eligió mirando estas escenas y las de referencia: no hay ninguna reservada**, así que las cifras son optimistas.

- **La Wildfire se usa solo para validar: `core/` no la lee** y no entra en ninguna decisión del programa. Es una referencia ruidosa: el script de Copernicus marca como fuego todo lo muy brillante en la SWIR, y también pinta de rosa suelo pálido, minas y parcelas. Por eso el recall contra ella subestima al detector. `python tools_c/comparar_wildfire.py --montaje` lo hace visible: de los 36 objetos de Niebla que no detectamos, **35 son brillantes pero no naranjas en la SWIR** (suelo pálido probable) y 1 no llega al umbral de brillo. Si se suponen no-fuego, el recall sube a 0,98 (Niebla) y 1,00 (extra); es una **estimación**, comprobada a ojo con 6 de los 36 casos en el montaje, no una medida.
- Las cifras de la tabla salen de ese script (ignora objetos de referencia de menos de 6 px, 5 px de tolerancia). Ajustar umbrales mirando esta comparación gasta Niebla y la extra como prueba independiente: para ajustar se usan antes las marcas a mano de Jesús, que sí distinguen fuego de suelo.
- En Niebla se detecta el **foco aislado en los campos** (abajo a la derecha) y se marca con confianza baja.
- **Se pierde mucho en la escena de práctica:** la mitad de sus focos físicos no son "rojos" en la SWIR, porque el humo sube el canal B04. Se probó a comparar solo B12 con B8A, que el humo no toca: ganaba de 3 a 5 puntos de recall, pero bajaba la precisión y daba píxeles falsos en la escena sin incendio. Se descartó.
- Hay algún foco en mitad de la cicatriz (posibles rescoldos) que no podemos confirmar.

## Por qué así: decisiones y medidas

| Variante | Ávila-Madrid | Niebla | Escena sin incendio | Escena extra |
|---|---|---|---|---|
| v0: NBR + brillo con Otsu | 37,8 % | 35,4 % | 45,4 % | — |
| v1: + tono rojizo | 4,3 % | 6,7 % | 16,2 % | — |
| v2: + prior espacial | 4,1 % | 6,3 % | 8,5 % | 9,1 % (pierde una cicatriz) |
| v3: histéresis + evidencia | 5,7 % | 8,0 % | 0,0 % | 12,0 % |
| **v4: v3 + NBR nunca por debajo de 0** | **5,7 %** | **8,0 %** | **0,0 %** | **12,0 %** |

- **El NBR a secas no siempre basta.** Sin foto de antes, todo terreno seco tiene NBR bajo, y en Niebla y Ávila-Madrid v0 marca más de un tercio de la escena. Pero depende de la escena: en la de práctica del 26-07 (ver más abajo) v0 marca un 11 % y, de hecho, sale mejor que v3 y v4 en IoU.
- **El tono rojizo como corte duro pierde cicatrices oscuras.** En la escena extra, v2 se dejaba una segunda cicatriz marrón oscura (RED_DOM −0,08). La histéresis la recupera sin dejar entrar el bosque oscuro: medido en Niebla y Ávila-Madrid, lo que se colaba era más claro que la ceniza (brillo mediano 0,33–0,41 frente a 0,24–0,25).
- **Otsu siempre parte la escena en dos, haya incendio o no.** En una escena de junio sin incendio (`Browser_images-9`), v2 marcaba un 8,5 % de matorral pardo; con solo el NBR normalizado, otra escena del 24-06 (antes del incendio) seguía dando un 21 %. Separa el NBR *original*: en las cicatrices reales la mediana es −0,10 a −0,27 (B12 sobre B8A, rojizas) y en ese terreno seco es +0,13 (verdoso). Con esa regla las 4 escenas sin incendio dan 0,00 % y las reales no cambian.
- **Solo SWIR:** el resultado es idéntico con o sin True Color (Niebla 7,79 % en los dos casos). Hoy la True Color solo se usa para la máscara de nubes y para dibujar.
- **Calibración True Color ↔ SWIR:** B04 aparece en las dos imágenes (rojo de la True Color, azul de la SWIR). Una recta robusta los relaciona con correlación 0,93–0,95: azul SWIR ≈ 1,26·x − 101 (Ávila-Madrid), 1,48·x − 140 (Niebla) y 1,11·x − 81 (extra). Es un diagnostico de semejanza entre canales; no certifica por si solo la alineacion geometrica. La hipótesis de que esa recta también vale para el verde y el rojo de la SWIR **no se pudo comprobar**, porque son bandas que la True Color no tiene.

## Prueba con dos escenas nuevas (costa con mar y Tenerife)

Con el código marcado `pre-examen` se pasaron una vez y sin tocar nada:
- **Focos:** bien en las dos (frentes activos detectados). En Tenerife, unos 8 falsos sobre conos volcánicos rojizos.
- **Cicatriz: 0 % en las dos.** Fallo de diseño: con mucho mar, el mar arrastraba los percentiles y el agua se detectaba mal (la regla suponía poca agua).

Se corrigió (agua por hueco del histograma, percentiles solo sobre tierra, evidencia `NBR original < −0,05` y `KEEP_FRAC` 0,10). La costa pasó a 9,5 %, con la cicatriz bien dibujada. Tenerife queda en 0,2 %: su cicatriz está casi toda bajo el humo. Las demás escenas no empeoran (práctica IoU 0,56; escenas sin incendio 0 %; marcas de Jesús 0,99). **Estas dos escenas ya no son examen ciego.**

## Medida contra una referencia (IoU)

Con el lote de práctica `datos_16bit` (mismo incendio del 26-07 que Ávila-Madrid, 2500×1128, bandas crudas de dos fechas)
se calcula una cicatriz de referencia con **dNBR** (NBR del 24-06 menos NBR del 26-07; umbral 0,27 = severidad moderada y
0,10 = generosa; áreas 336 y 605 km², frente a ~348 y ~620 que da el lote). La solución solo usa la SWIR y la True Color del
26-07, como en el reto real. Se ejecuta con `python tools_c/medir_dnbr.py`.

| Variante | % imagen | vs dNBR>0,27: IoU / precisión / recall | vs dNBR>0,10: IoU / precisión / recall |
|---|---|---|---|
| v0 | 11,3 | 0,59 / 0,71 / 0,77 | 0,54 / 0,93 / 0,56 |
| v1 | 5,9 | 0,46 / 0,87 / 0,49 | 0,30 / 0,97 / 0,30 |
| v2 | 4,9 | 0,38 / 0,86 / 0,41 | 0,25 / 0,96 / 0,25 |
| v3 | 8,6 | 0,54 / 0,78 / 0,64 | 0,41 / 0,93 / 0,43 |
| **v4 (por defecto)** | 9,5 | 0,55 / 0,74 / 0,68 | 0,45 / 0,92 / 0,47 |

(Tras la corrección de Jesús en `fire_like` (canal rojo original) v4 llegó a 0,56 / 0,75 / 0,70, pero eso reactivaba la regla "conservar manchas pequeñas cerca de un foco" y dejaba dos manchas falsas en Niebla, una alrededor del foco aislado de los campos. Al exigir además que estén pegadas a la cicatriz principal, bajó a 0,55 / 0,74 / 0,68 y las manchas desaparecieron. Es el coste de no dibujar cicatriz donde no la hay.)

Lectura honesta:
- **Lo que marcamos casi siempre es quemado:** la precisión contra la referencia generosa es 0,93.
- **Nos dejamos mucha cicatriz:** el recall con v4 es 0,68 (0,47 contra la referencia generosa). Las omisiones se ven en
  el sur y entre los embalses (`debug/dnbr_0726/errores_v4.png`), pero **no son sobre todo por el humo** (solo un 3,6 %
  cae en humo denso). La causa es que, con una cicatriz tan grande, los Otsu parten la propia cicatriz en dos (severa y
  moderada) y dejan fuera la moderada. La v4 corrige eso en el NBR (+0,02 de IoU, +0,04 de recall, sin cambiar nada en las
  otras escenas). En el brillo no hay un límite físico equivalente: un intento (crecer por lo tan oscuro como las semillas)
  subía el IoU a 0,57, pero borraba entera la cicatriz de Ávila-Madrid, así que se descartó. Ampliar el radio de
  crecimiento no cambia nada.
- **Los filtros que añadimos para quitar falsos positivos en otras escenas cuestan recall aquí:** v0 sale mejor que v4 en IoU
  (0,59 frente a 0,55). Es un compromiso real, todavía sin resolver.
- **Esta no es una prueba independiente:** es el mismo incendio que la escena Ávila-Madrid con la que desarrollamos.
- dNBR es una referencia, no la verdad: puede incluir campos cosechados y cambios de estación entre las dos fechas.

## Limitaciones conocidas (honestas)

- **Hay referencias humanas parciales del 21 de julio.** Sus metricas solo cubren regiones anotadas y no certifican todo el perimetro. Las otras escenas estan pendientes de anotacion.
- **La escena extra ya no es ciega.** Se vio su fallo con v2 y se corrigió con una regla general, ajustada solo con Niebla, Ávila-Madrid y la escena sin incendio.
- **Humo denso:** la cicatriz bajo humo espeso puede quedar fuera (por ejemplo, la zona gris oscura al norte de la escena extra).
- **Bordes:** en Niebla, la cicatriz se desborda un poco al sur del anillo de fuego.
- **Fragilidad conocida:** la prueba de evidencia mira el NBR mediano de toda la mancha. Si una cicatriz real se uniera con
  mucha zona no quemada, la mancha entera podría descartarse. Se probó a medirla solo sobre las semillas, pero eso dejaba
  pasar matorral en la escena sin incendio (0,6 %), así que se mantuvo como está.
- **Ciudad:** la máscara no es fiable. Lo que mantiene Madrid fuera es el prior espacial y la evidencia de NBR.
- **Cambio de producto:** los umbrales `COMP_NBR_MAX = −0,15` y `SEED_RED_MIN = 0` son fijos sobre índices normalizados. Son heuristicas visuales examinadas en 4 escenas, pero conviene revisarlos si el producto SWIR usa otra combinación de bandas.

## Créditos y referencias

- Copernicus Browser / Sentinel Hub, para las imágenes (ESA, Copernicus).
- NBR: Key & Benson, *Landscape Assessment* (FIREMON), USDA Forest Service, 2006.
- TangChao729, *Burned_Area_Segmentation* (GitHub). Según su README, un índice espectral solo da un IoU de 0,40 frente a 0,73–0,82 de las redes. Es un argumento de por qué no basta con un índice (dato tomado de su README, no reproducido por nosotros).

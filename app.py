"""Mesa de luz: app de Streamlit para explorar incendios en imágenes Sentinel-2.

    streamlit run app.py

Tres pasos: 1 Imagen (subir o elegir un ejemplo), 2 Análisis (progreso por capa), 3 Explorar (visor).
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import streamlit as st

from components.viewer import build_payload, render_html, to_html
from core.ingestion import kind_from_content, rgb_from_bytes
from layers import LAYERS, LAYERS_BY_ID
from pipeline import run_all_layers
from results_io import date_from_names, load_analysis, zip_bytes

ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples"
IMG_TYPES = ["jpg", "jpeg", "png", "tif", "tiff"]
STEPS = ("Imagen", "Análisis", "Explorar")
KIND_NAMES = {"tc": "True Color", "swir": "SWIR", "wildfire": "Wildfires"}

st.set_page_config(page_title="Mesa de luz · Incendios", page_icon="🔥", layout="wide",
                   initial_sidebar_state="collapsed")

CSS = """
<style>
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], .stDeployButton { display: none !important; }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px; }
:root { --amber: #ffb000; --line: #2b343e; --ink-2: #a9a497; --mono: ui-monospace, "JetBrains Mono", "Cascadia Mono", Consolas, monospace; }
.brand { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin-bottom: .2rem; }
.brand .tag { font: 600 12px/1 var(--mono); letter-spacing: .14em; text-transform: uppercase; color: var(--amber);
  border: 1px solid #8a6410; padding: 5px 7px; border-radius: 2px; }
.brand h1 { font-size: 1.55rem; margin: 0; padding: 0; font-weight: 650; }
.lede { color: var(--ink-2); margin: .2rem 0 1rem; }
.steps { display: flex; gap: 0; margin: .4rem 0 1.4rem; border: 1px solid var(--line); border-radius: 4px; overflow: hidden; }
.steps .s { flex: 1; display: flex; align-items: center; gap: 10px; padding: 10px 14px; color: var(--ink-2); border-right: 1px solid var(--line);
  background: #13181d; min-width: 0; }
.steps .s:last-child { border-right: 0; }
.steps .n { font: 600 13px/1 var(--mono); width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center;
  border: 1px solid #3a4551; flex: none; }
.steps .s.done .n { border-color: #8a6410; color: var(--amber); }
.steps .s.now { color: #1a1206; background: var(--amber); font-weight: 600; }
.steps .s.now .n { border-color: #1a1206; }
.steps .t { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kicker { font: 600 11px/1 var(--mono); letter-spacing: .14em; text-transform: uppercase; color: #737066; margin: 1.2rem 0 .6rem; }
.card-title { font-weight: 650; font-size: 1.05rem; margin: .5rem 0 0; }
.card-meta { font: 12px/1.5 var(--mono); color: var(--ink-2); }
.card-fig { margin: .35rem 0 .5rem; font-size: .92rem; }
.card-fig b { font-family: var(--mono); color: var(--amber); font-weight: 600; }
.layerline { font-family: var(--mono); font-size: .9rem; }
[data-testid="stFileUploaderDropzone"] { background: #13181d; border: 1px dashed #3a4551; }
/* Textos del cuadro de subida en español (Streamlit no los traduce) */
[data-testid="stFileUploaderDropzoneInstructions"] span, [data-testid="stFileUploaderDropzoneInstructions"] small { font-size: 0 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] > div::after { content: "Arrastra aquí la imagen · JPG, PNG o TIF"; font-size: 13px; color: var(--ink-2); }
[data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] p { font-size: 0 !important; }
[data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] p::after { content: "Elegir archivo"; font-size: 14px; }
button[kind="primary"]:not(:disabled) p { color: #1a1206 !important; font-weight: 600; }
[data-testid="stImage"] img { border-radius: 3px; border: 1px solid var(--line); }
@media (max-width: 640px) { .steps .t { display: none; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("step", 1)
ss.setdefault("source", None)   # {"kind": "ejemplo", "name": ...} | {"kind": "subida", "files": {...}, "title": ...}


# ---------------------------------------------------------------- caché
@st.cache_data(show_spinner=False, max_entries=3)
def analyze(tc: bytes | None, swir: bytes, wildfire: bytes | None, _progress: list | None = None) -> dict:
    """Análisis completo de unas imágenes subidas. Función pura (sin elementos de Streamlit) para que la caché
    sea segura; el progreso se comunica por la lista _progress (no forma parte de la clave de caché)."""
    imgs = {k: rgb_from_bytes(b, KIND_NAMES[k]) if b else None for k, b in (("tc", tc), ("swir", swir), ("wildfire", wildfire))}
    cb = (lambda layer, r: _progress.append((layer.id, r["status"], r["time_s"], r["error"]))) if _progress is not None else None
    return run_all_layers(imgs["tc"], imgs["swir"], imgs["wildfire"], name="tus imágenes", on_layer=cb)


@st.cache_data(show_spinner=False)
def example_list() -> list[dict]:
    out = []
    for d in sorted(EXAMPLES.glob("*/metrics.json")):
        m = json.loads(d.read_text(encoding="utf-8"))
        out.append({"name": d.parent.name, "meta": m})
    return sorted(out, key=lambda e: (e["meta"].get("orden", 99), e["name"]))


@st.cache_data(show_spinner=False, max_entries=4)
def example_analysis(name: str) -> dict:
    return load_analysis(EXAMPLES / name)


@st.cache_data(show_spinner=False, max_entries=4)
def viewer_html(key: str, _analysis: dict, title: str) -> str:
    return to_html(build_payload(_analysis, title))


@st.cache_data(show_spinner=False, max_entries=4)
def download_zip(key: str, _analysis: dict, meta_json: str) -> bytes:
    return zip_bytes(_analysis, json.loads(meta_json))


def source_key(src: dict) -> str:
    if src["kind"] == "ejemplo":
        return "ejemplo:" + src["name"]
    h = hashlib.sha1()
    for k in ("tc", "swir", "wildfire"):
        h.update((src["files"].get(k) or b"-")[:1 << 20] + str(len(src["files"].get(k) or b"")).encode())
    return "subida:" + h.hexdigest()


# ---------------------------------------------------------------- piezas comunes
def go(step: int):
    ss.step = step


def header():
    st.markdown('<div class="brand"><span class="tag">Mesa de luz</span><h1>Incendios vistos desde el satélite</h1></div>',
                unsafe_allow_html=True)
    cells = []
    for i, name in enumerate(STEPS, 1):
        cls = "now" if i == ss.step else ("done" if i < ss.step else "")
        cells.append(f'<div class="s {cls}"><span class="n">{i}</span><span class="t">{name}</span></div>')
    st.markdown(f'<div class="steps" role="list" aria-label="Pasos">{"".join(cells)}</div>', unsafe_allow_html=True)


def fmt(x: float, d: int = 1) -> str:
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def layer_line(lid: str, status: str, t: float, err: str | None) -> str:
    name = LAYERS_BY_ID[lid].nombre if lid in LAYERS_BY_ID else lid
    if status == "ok":
        return f"✅ **{name}** · <span class='layerline'>{fmt(t, 2)} s</span>"
    if status == "no_disponible":
        return f"➖ **{name}**: no disponible en este equipo. {err or ''}"
    return f"⚠️ **{name}**: no se ha podido calcular. El resto del análisis sigue. <span class='layerline'>({err})</span>"


# ---------------------------------------------------------------- paso 1
def step_image():
    st.markdown('<p class="lede">Sube las imágenes Sentinel-2 de un incendio y verás qué terreno se ha quemado, '
                'dónde sigue habiendo fuego y por dónde pasa el perímetro.</p>', unsafe_allow_html=True)
    st.markdown('<div class="kicker">Tus imágenes</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        tc = st.file_uploader("True Color", type=IMG_TYPES, key="up_tc",
                              help="La imagen en color real, como una fotografía. Opcional, pero ayuda a descartar nubes.")
        st.caption("Color real, como una foto. Opcional.")
    with c2:
        sw = st.file_uploader("SWIR", type=IMG_TYPES, key="up_swir",
                              help="Infrarrojo de onda corta (bandas B12, B8A y B04). Es donde se ve la quemadura.")
        st.caption("Infrarrojo. **Imprescindible**: aquí se ve lo quemado.")
    with c3:
        wf = st.file_uploader("Wildfires (opcional)", type=IMG_TYPES, key="up_wf",
                              help="La visualización de incendios de Copernicus. Solo se usa como fondo en el visor.")
        st.caption("Vista de incendios de Copernicus. Solo como fondo.")

    files = {"tc": tc.getvalue() if tc else None, "swir": sw.getvalue() if sw else None,
             "wildfire": wf.getvalue() if wf else None}
    # Aviso amable si una imagen parece estar en el hueco equivocado
    for k, f in files.items():
        if f is None:
            continue
        try:
            guess = kind_from_content(rgb_from_bytes(f, KIND_NAMES[k])[::4, ::4])
        except ValueError:
            st.error(f"No se puede abrir la imagen de {KIND_NAMES[k]}. Prueba con un JPG o PNG.")
            files[k] = None
            continue
        if guess != k:
            st.warning(f"La imagen subida como **{KIND_NAMES[k]}** parece más bien **{KIND_NAMES[guess]}**. "
                       "Compruébalo; si es correcta, puedes seguir.")

    if st.button("Analizar mis imágenes", type="primary", disabled=files["swir"] is None,
                 help=None if files["swir"] else "Sube al menos la imagen SWIR"):
        fecha = date_from_names(u.name for u in (sw, tc, wf) if u)
        ss.source = {"kind": "subida", "files": files, "title": "Tus imágenes", "fecha": fecha}
        go(2)
        st.rerun()

    st.markdown('<div class="kicker">O prueba con un incendio real</div>', unsafe_allow_html=True)
    examples = example_list()
    if not examples:
        st.info("No hay escenas de ejemplo. Genera la carpeta examples/ con `python precompute.py`.")
        return
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        m = ex["meta"]
        c = m["capas"]
        with col:
            thumb = EXAMPLES / ex["name"] / "thumb.jpg"
            if thumb.exists():
                st.image(str(thumb), width="stretch")
            st.markdown(f'<div class="card-title">{m.get("titulo", ex["name"])}</div>'
                        f'<div class="card-meta">{m.get("fecha") or ""}</div>', unsafe_allow_html=True)
            figs = []
            if c.get("burn", {}).get("estado") == "ok":
                figs.append(f'<b>{fmt(c["burn"]["metricas"]["pct_imagen"])} %</b> de la imagen afectada')
            if c.get("fire", {}).get("estado") == "ok":
                n = c["fire"]["metricas"]["regiones"]
                figs.append(f'<b>{n}</b> {"foco" if n == 1 else "focos"} de fuego activo')
            st.markdown(f'<div class="card-fig">{" · ".join(figs)}</div>', unsafe_allow_html=True)
            if st.button("Abrir este incendio", key="ex_" + ex["name"], width="stretch"):
                ss.source = {"kind": "ejemplo", "name": ex["name"], "title": m.get("titulo", ex["name"])}
                go(3)
                st.rerun()


# ---------------------------------------------------------------- paso 2
def run_with_progress(files: dict) -> dict | None:
    """Ejecuta analyze() en un hilo y va pintando cada capa al terminar. Si ya estaba en caché, es instantáneo."""
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

    progress: list = []
    box: dict = {}

    def work():
        try:
            box["res"] = analyze(files["tc"], files["swir"], files["wildfire"], _progress=progress)
        except Exception as e:  # noqa: BLE001
            box["err"] = e

    th = threading.Thread(target=work, daemon=True)
    add_script_run_ctx(th, get_script_run_ctx())
    t0 = time.perf_counter()
    with st.status("Analizando la imagen…", expanded=True) as status:
        th.start()
        shown = 0
        slot = st.empty()
        while th.is_alive() or shown < len(progress):
            while shown < len(progress):
                st.markdown(layer_line(*progress[shown]), unsafe_allow_html=True)
                shown += 1
            if th.is_alive():
                nxt = LAYERS[shown].nombre if shown < len(LAYERS) else "…"
                slot.caption(f"Calculando: {nxt} · {fmt(time.perf_counter() - t0)} s")
                time.sleep(0.15)
        slot.empty()
        if "err" in box:
            status.update(label="No se ha podido analizar la imagen", state="error")
            st.error(f"No se ha podido preparar la imagen: {box['err']}")
            return None
        res = box["res"]
        if shown == 0:   # venía de la caché: se enseña el registro guardado
            for lid, r in res["layers"].items():
                st.markdown(layer_line(lid, r["status"], r["time_s"], r["error"]), unsafe_allow_html=True)
        bad = [r for r in res["layers"].values() if r["status"] != "ok"]
        label = f"Análisis terminado en {fmt(res['time_s'])} s"
        if bad:
            label += f" · {len(bad)} {'capa' if len(bad) == 1 else 'capas'} sin calcular"
        status.update(label=label, state="error" if len(bad) == len(res["layers"]) else "complete", expanded=True)
    return res


def step_analysis():
    src = ss.source
    if not src:
        go(1)
        st.rerun()
    if src["kind"] == "ejemplo":
        a = example_analysis(src["name"])
        with st.status(f"Análisis precalculado de «{src['title']}»", state="complete", expanded=True):
            for lid, r in a["layers"].items():
                st.markdown(layer_line(lid, r["status"], r["time_s"] or 0, r["error"]), unsafe_allow_html=True)
        res = a
    else:
        # Mientras dura el análisis Streamlit deja a la vista, en gris, lo que había en el paso 1.
        # Se reservan los huecos de la página ahora y los sobrantes se vacían para que eso desaparezca.
        status_box, actions = st.container(), st.container()
        for _ in range(40):
            st.empty()
        with status_box:
            res = run_with_progress(src["files"])
        with actions:
            step_analysis_footer(res)
        return
    step_analysis_footer(res)


def step_analysis_footer(res: dict | None):
    for w in (res or {}).get("scene", {}).get("warnings", []):
        st.caption(f"Aviso: {w}")
    c1, c2, _ = st.columns([1, 1, 3])
    c1.button("← Cambiar imagen", on_click=go, args=(1,), width="stretch")
    if res is not None:
        c2.button("Explorar el resultado →", type="primary", on_click=go, args=(3,), width="stretch")


# ---------------------------------------------------------------- paso 3
def step_explore():
    src = ss.source
    if not src:
        go(1)
        st.rerun()
    key = source_key(src)
    if src["kind"] == "ejemplo":
        a = example_analysis(src["name"])
        meta = {k: a["meta"][k] for k in ("titulo", "fecha") if k in a["meta"]}
    else:
        f = src["files"]
        a = analyze(f["tc"], f["swir"], f["wildfire"])   # en caché tras el paso 2: no reanaliza
        meta = {"titulo": src["title"], **({"fecha": src["fecha"]} if src.get("fecha") else {})}

    top = st.columns([4, 1, 1])
    top[0].markdown(f"**{src['title']}**" + (f" · {meta['fecha']}" if meta.get("fecha") else ""))
    top[1].button("← Imagen", on_click=go, args=(1,), width="stretch")
    top[2].button("Análisis", on_click=go, args=(2,), width="stretch")

    title = src["title"] + (f" · {meta['fecha']}" if meta.get("fecha") else "")
    render_html(viewer_html(key, a, title), height=820)

    st.markdown('<div class="kicker">Descargas</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in src["title"])
    c1.download_button("Descargar máscaras y cifras (.zip)", data=download_zip(key, a, json.dumps(meta)),
                       file_name=f"mascaras_{safe}.zip", mime="application/zip", type="primary",
                       on_click="ignore", width="stretch")
    c2.caption("Una máscara PNG en blanco y negro por capa, a la resolución analizada, y un metrics.json con "
               "las cifras de cada capa. Todo en píxeles: la imagen no está georreferenciada.")


header()
{1: step_image, 2: step_analysis, 3: step_explore}[ss.step]()

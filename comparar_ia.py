"""Comparacion JPG por grupos: ninguna escena evaluada entra en su entrenamiento."""
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import joblib
import numpy as np
from PIL import Image
import sklearn

from ai_refinement import (FEATURE_NAMES, RF_CONFIG, SEED, THRESHOLD,
                           feature_tile, fit_heads, positive_probability,
                           sample_scene, tiles)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(name):
    name = Path(name).name.lower()
    while Path(name).suffix in {".jpg", ".jpeg", ".png", ".json"}:
        name = Path(name).stem
    return name


def mask(path):
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim != 2 or not np.isin(array, [0, 1, 255]).all():
        raise ValueError(f"Mascara no binaria: {path}")
    return array > 0


def load_scene(scene):
    images = []
    for key in ("tc", "swir"):
        with Image.open(scene[key]) as image:
            if image.mode != "RGB":
                raise ValueError(f"Se necesita JPG RGB: {scene[key]}")
            images.append(np.array(image))
    tc, swir = images
    affected, active = mask(scene["reference"] / "affected.png"), mask(scene["reference"] / "active.png")
    valid = mask(scene["reference"] / "valid.png") & mask(scene["output"] / "valid.png")
    if tc.shape != swir.shape or any(array.shape != tc.shape[:2] for array in (affected, active, valid)):
        raise ValueError(f"Imagenes/referencias sin alinear: {scene['id']}")
    if not valid.any() or (active & ~affected & valid).any():
        raise ValueError("Referencia vacia o fuego activo fuera de la referencia afectada.")
    return tc, swir, affected, active, valid


def metric(counts, total_pixels):
    tp, fp, fn, tn = map(int, counts)
    def divide(a, b):
        return a / b if b else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=divide(tp, tp + fp),
                recall=divide(tp, tp + fn), f1=divide(2 * tp, 2 * tp + fp + fn),
                iou=divide(tp, tp + fp + fn), valid_pixels=sum((tp, fp, fn, tn)),
                coverage_pct=100 * sum((tp, fp, fn, tn)) / total_pixels)


def confusion(prediction, reference):
    return np.array([np.sum(prediction & reference), np.sum(prediction & ~reference),
                     np.sum(~prediction & reference), np.sum(~prediction & ~reference)], dtype=np.int64)


def evaluate_scene(scene, models, output):
    start = time.perf_counter()
    tc, swir, affected, active, valid = load_scene(scene)
    baseline = {head: mask(scene["output"] / filename) for head, filename in
                (("afectado", "affected.png"), ("activo", "active.png"))}
    if any(array.shape != valid.shape for array in baseline.values()):
        raise ValueError("Predicciones base en otra resolucion.")
    totals = {name: {head: np.zeros(4, dtype=np.int64) for head in models}
              for name in ("base", "rf")}
    # Evidencias: hasta dos recortes por cabeza/tipo, solo dentro de referencias validas.
    evidence_counts = {(head, kind): 0 for head in models for kind in ("fp", "fn")}
    evidence = []
    for box in tiles(*valid.shape):
        x0, y0, x1, y1 = box
        region = valid[y0:y1, x0:x1]
        if not region.any():
            continue
        x = feature_tile(tc, swir, box)[region]
        for head, reference in (("afectado", affected), ("activo", active)):
            truth = reference[y0:y1, x0:x1][region]
            pred_base = baseline[head][y0:y1, x0:x1][region]
            pred_rf = positive_probability(models[head], x) >= THRESHOLD
            totals["base"][head] += confusion(pred_base, truth)
            totals["rf"][head] += confusion(pred_rf, truth)
            for kind, errors in (("fp", pred_rf & ~truth), ("fn", ~pred_rf & truth)):
                key = (head, kind)
                if errors.sum() < 8 or evidence_counts[key] >= 2:
                    continue
                overlay = np.array(tc[y0:y1, x0:x1])
                colors = np.zeros_like(overlay)
                painted = np.zeros(region.shape, dtype=bool)
                painted[region] = errors
                colors[painted] = (255, 0, 220) if kind == "fp" else (0, 210, 255)
                overlay[painted] = (0.35 * overlay[painted] + 0.65 * colors[painted]).astype(np.uint8)
                filename = f"{scene['id']}_{head}_{kind}_{evidence_counts[key] + 1}.png"
                Image.fromarray(overlay).save(output / "errores" / filename)
                evidence.append(dict(archivo=f"errores/{filename}", cabeza=head, tipo=kind,
                                     bbox_original=list(box), pixeles_error=int(errors.sum())))
                evidence_counts[key] += 1
    result = {name: {head: metric(counts, valid.size) for head, counts in heads.items()}
              for name, heads in totals.items()}
    result["segundos_evaluacion_anotaciones"] = time.perf_counter() - start
    result["evidencias_rf"] = evidence
    return result


def resolve_scenes(project):
    manifest = project / "escenas.json"
    rows = json.loads(manifest.read_text(encoding="utf-8"))["escenas"]
    scenes, tracked = [], [manifest]
    references = project / "referencias" / "desarrollo"
    ids = set()
    for row in rows:
        if row["id"] in ids or row["split"] != "desarrollo":
            raise ValueError("Este experimento requiere IDs unicos y split desarrollo.")
        ids.add(row["id"])
        folder = project / row["imagenes"]
        files = [path for path in folder.iterdir() if path.suffix.lower() in (".jpg", ".jpeg")]
        def unique(fragment):
            candidates = [path for path in files if fragment in path.stem.lower()]
            if len(candidates) != 1:
                raise ValueError(f"No hay un unico {fragment} en {folder}")
            return candidates[0]
        tc, swir = unique("true"), unique("swir")
        refs = [path for path in references.iterdir() if path.is_dir() and canonical(path.name) == canonical(tc.name)]
        if len(refs) != 1:
            raise ValueError(f"Referencia ausente/ambigua: {row['id']}")
        scene = dict(row, tc=tc, swir=swir, reference=refs[0], output=project / "salida" / row["id"])
        metadata = json.loads((scene["output"] / "resultado.json").read_text(encoding="utf-8"))
        if metadata["escena"] != scene["id"] or any(metadata.get(f"estado_{head}") != "disponible"
                                                   for head in ("afectado", "activo")):
            raise ValueError(f"Prediccion base no disponible para {row['id']}")
        for source in (scene["reference"], scene["output"]):
            tracked.extend(source / filename for filename in ("affected.png", "active.png", "valid.png"))
        tracked += [tc, swir, scene["output"] / "resultado.json"]
        scenes.append(scene)
    if len({scene["grupo"] for scene in scenes}) < 3:
        raise ValueError("Se requieren al menos tres grupos de adquisicion.")
    tracked += list((project / "anotaciones" / "desarrollo").glob("*.json"))
    tracked += list((project / "core").glob("*.py"))
    tracked += [project / "run.py", project / "evaluation.py"]
    return scenes, sorted(set(tracked))


def write_report(report, output):
    (output / "comparacion.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["escena", "cabeza", "metodo", "precision", "recall", "f1", "iou", "tp", "fp", "fn", "tn", "coverage_pct"]
    with (output / "comparacion.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["escenas"]:
            for head in ("afectado", "activo"):
                for method in ("base", "rf"):
                    writer.writerow(dict(escena=row["escena"], cabeza=head, metodo=method,
                                         **{key: row[method][head][key] for key in fields[3:]}))
    lines = ["# Comparacion Persona 3", "", report["alcance"], "",
             "Umbral fijo 0.5. Random Forest: 96 arboles, profundidad maxima 16, hoja minima 3.", "",
             "| Escena | Cabeza | F1 base | F1 RF | Precision RF | Recall RF |",
             "|---|---|---:|---:|---:|---:|"]
    for row in report["escenas"]:
        for head in ("afectado", "activo"):
            base, rf = row["base"][head], row["rf"][head]
            def fmt(value):
                return f"{100 * value:.2f}%" if value is not None else "indefinido"
            lines.append(f"| {row['escena']} | {head} | {fmt(base['f1'])} | {fmt(rf['f1'])} | {fmt(rf['precision'])} | {fmt(rf['recall'])} |")
    lines += ["", "## Decision del experimento", ""]
    for head, row in report["decision"].items():
        lines.append(f"- {head}: {row['decision']}. F1 medio por escena: {100*row['f1_base_macro']:.2f}% -> {100*row['f1_rf_macro']:.2f}%.")
    lines += ["", "La regla previa exige subir al menos 1 punto de F1 medio, no perder mas de 5 puntos en ninguna escena y mantener precision >=90% en cada escena.",
              "Los tiempos de evaluacion corresponden a pixeles anotados; no son tiempos de inferencia de toda la imagen.",
              "Los recortes muestran FP en magenta y FN en cian, solo en regiones anotadas. Revisar antes de integrar.",
              "Las tres escenas se usaron para desarrollar el sistema: esto NO es una prueba independiente.",
              "Wildfires no se usa como entrada del RF. Las etiquetas humanas pueden haberse apoyado en esa visualizacion."]
    (output / "COMPARACION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proyecto", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--salida", type=Path)
    parser.add_argument("--muestras-por-clase", type=int, default=8000)
    args = parser.parse_args()
    if args.muestras_por_clase < 100:
        parser.error("Usa al menos 100 muestras por clase.")
    project = args.proyecto.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.salida or project / "resultados_evaluacion" / f"ia_{stamp}"
    output = output.resolve()
    scenes, tracked = resolve_scenes(project)
    output.mkdir(parents=True, exist_ok=False)
    (output / "errores").mkdir()
    print(f"Guardando punto de partida en {output}", flush=True)
    hashes = {str(path.relative_to(project)): sha(path) for path in tracked}
    for path in tracked:
        relative = path.relative_to(project)
        if relative.parts[0] != "data":
            target = output / "punto_partida" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for name in ("ai_refinement.py", "comparar_ia.py"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    (output / "huellas_entrada.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    records = {}
    sampling = {}
    for index, scene in enumerate(scenes):
        print(f"Preparando muestras JPG: {scene['id']}", flush=True)
        data = load_scene(scene)
        record = sample_scene(*data, limit=args.muestras_por_clase, seed=SEED + index)
        records[scene["id"]] = record
        sampling[scene["id"]] = dict(disponibles=record["available"], usadas=record["sampled"])
        del data
        gc.collect()
    report = dict(fecha_utc=datetime.now(timezone.utc).isoformat(),
                  alcance="Validacion de desarrollo por grupos de adquisicion, sobre referencias humanas parciales a resolucion original.",
                  configuracion=dict(rf=RF_CONFIG, umbral=THRESHOLD, muestras_por_clase=args.muestras_por_clase,
                                     features=FEATURE_NAMES, seed=SEED),
                  versiones=dict(python=sys.version, sklearn=sklearn.__version__, numpy=np.__version__, opencv=cv2.__version__),
                  muestras=sampling, escenas=[], decision={})
    for group in sorted({scene["grupo"] for scene in scenes}):
        train = [scene for scene in scenes if scene["grupo"] != group]
        test = [scene for scene in scenes if scene["grupo"] == group]
        assert {scene["grupo"] for scene in train}.isdisjoint({scene["grupo"] for scene in test})
        print(f"Entrenando con {[scene['id'] for scene in train]}; evaluando {[scene['id'] for scene in test]}", flush=True)
        models, fit_seconds = fit_heads([records[scene["id"]] for scene in train])
        for scene in test:
            result = evaluate_scene(scene, models, output)
            row = dict(escena=scene["id"], grupo_evaluado=group,
                       entrenamiento=[item["id"] for item in train], segundos_entrenamiento=fit_seconds, **result)
            report["escenas"].append(row)
            for head in ("afectado", "activo"):
                print(f"  {scene['id']} {head}: F1 base={result['base'][head]['f1']:.4f}, RF={result['rf'][head]['f1']:.4f}", flush=True)
        del models
        gc.collect()
    for head in ("afectado", "activo"):
        base = [row["base"][head]["f1"] for row in report["escenas"]]
        rf = [row["rf"][head]["f1"] for row in report["escenas"]]
        precision = [row["rf"][head]["precision"] for row in report["escenas"]]
        base_mean, rf_mean = float(np.mean(base)), float(np.mean(rf))
        eligible = (rf_mean >= base_mean + 0.01 and all(b - r <= 0.05 for b, r in zip(base, rf))
                    and all(p is not None and p >= 0.9 for p in precision))
        report["decision"][head] = dict(f1_base_macro=base_mean, f1_rf_macro=rf_mean,
                                       mejora_pp=100 * (rf_mean - base_mean),
                                       candidato_a_integracion=eligible,
                                       decision="candidato; requiere revision visual" if eligible else "mantener detector base")
    changed = [str(path) for path in tracked if not path.exists() or sha(path) != hashes[str(path.relative_to(project))]]
    if changed:
        (output / "ENTRADAS_CAMBIADAS.json").write_text(json.dumps(changed, indent=2), encoding="utf-8")
        raise RuntimeError("Las entradas cambiaron durante el experimento. No usar esta comparacion; repetir con entradas estables.")
    # Exportar un candidato solo si supera la regla previa. Nunca conectar automaticamente.
    accepted = [head for head, row in report["decision"].items() if row["candidato_a_integracion"]]
    if accepted:
        models, fit_seconds = fit_heads(list(records.values()))
        bundle = dict(schema=1, feature_names=FEATURE_NAMES, threshold=THRESHOLD,
                      models={head: models[head] for head in accepted},
                      entrenamiento=[scene["id"] for scene in scenes],
                      nota="Modelo final entrenado con todas las escenas; metricas reportadas de los modelos por grupos, no de este ajuste final.")
        joblib.dump(bundle, output / "candidato_rf.joblib", compress=3)
        report["modelo_final"] = dict(archivo="candidato_rf.joblib", cabezas=accepted,
                                      segundos_entrenamiento=fit_seconds)
    write_report(report, output)
    print(f"Comparacion terminada: {output / 'COMPARACION.md'}", flush=True)
    for head, row in report["decision"].items():
        print(f"  {head}: {row['decision']} ({row['mejora_pp']:+.2f} puntos F1 medio)", flush=True)


if __name__ == "__main__":
    main()

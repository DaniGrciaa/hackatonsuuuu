"""Evalua predicciones existentes, sin ejecutar ni ajustar los detectores."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluar_png import cargar_mascara
from evaluation import evaluate_mask


def canonical_name(name):
    name = Path(name).name.lower()
    while Path(name).suffix in {".jpg", ".jpeg", ".png", ".json"}:
        name = Path(name).stem
    return name


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_project(project, output_root, references_root, manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = {}
    ids = set()
    rows = []
    for scene in manifest["escenas"]:
        scene_id = scene["id"]
        if scene_id in ids:
            raise ValueError(f"Escena duplicada: {scene_id}")
        ids.add(scene_id)
        group, split = scene["grupo"], scene["split"]
        if group in groups and groups[group] != split:
            raise ValueError(f"El grupo {group} aparece en splits diferentes")
        groups[group] = split
        row = {"escena": scene_id, "grupo": group, "split": split}
        output = output_root / scene_id
        metadata_path = output / "resultado.json"
        if not metadata_path.exists():
            row["estado"] = "sin_prediccion"
            rows.append(row)
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["escena"] != scene_id:
            raise ValueError(f"Metadatos de otra escena en {output}")
        tc = metadata.get("archivos_entrada", {}).get("tc")
        if not tc:
            candidates = list((project / scene["imagenes"]).glob("*True*"))
            if len(candidates) == 1:
                tc = candidates[0].name
        matches = [p for p in references_root.iterdir()
                   if p.is_dir() and tc and canonical_name(p.name) == canonical_name(tc)]
        if len(matches) > 1:
            raise ValueError(f"Referencias ambiguas para {scene_id}: {matches}")
        if not matches:
            row["estado"] = "sin_referencia"
            rows.append(row)
            continue
        reference = matches[0]
        valid = cargar_mascara(reference / "valid.png")
        original_shape = tuple(metadata["tamano_original"])
        if valid.shape != original_shape:
            raise ValueError(f"{scene_id}: referencia {valid.shape} != original {original_shape}")
        prediction_valid = cargar_mascara(output / "valid.png")
        if prediction_valid.shape != valid.shape:
            raise ValueError(f"{scene_id}: mascara de validez en otra resolucion")
        valid &= prediction_valid
        row.update({"estado": "evaluado", "alcance": "solo regiones humanas anotadas",
                    "variante": metadata["variante"], "tamano_original": list(original_shape),
                    "afectado": None, "activo": None, "sha256": {}})
        for head, filename in (("afectado", "affected.png"), ("activo", "active.png")):
            status = metadata.get(f"estado_{head}", "no_disponible")
            row[f"estado_{head}"] = status
            if status != "disponible":
                continue
            prediction_path = output / filename
            reference_path = reference / filename
            row[head] = evaluate_mask(cargar_mascara(prediction_path),
                                      cargar_mascara(reference_path), valid)
            row["sha256"][head] = {"prediccion": sha256(prediction_path),
                                    "referencia": sha256(reference_path)}
        row["sha256"]["valid_referencia"] = sha256(reference / "valid.png")
        row["sha256"]["valid_imagen"] = sha256(output / "valid.png")
        rows.append(row)
    return {"fecha_utc": datetime.now(timezone.utc).isoformat(),
            "alcance": "referencias parciales; coverage_pct es cobertura anotada, no confianza",
            "escenas": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proyecto", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--salidas", type=Path, default=Path("salida"))
    parser.add_argument("--referencias", type=Path, default=Path("referencias/desarrollo"))
    parser.add_argument("--manifest", type=Path, default=Path("escenas.json"))
    parser.add_argument("--salida", type=Path, default=Path("resultados_evaluacion/resumen_desarrollo.json"))
    args = parser.parse_args()
    project = args.proyecto.resolve()
    def resolve(path):
        return path if path.is_absolute() else project / path
    result = evaluate_project(project, resolve(args.salidas), resolve(args.referencias), resolve(args.manifest))
    output = resolve(args.salida)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in result["escenas"]:
        print(f"{row['escena']}: {row['estado']}")
        for head in ("afectado", "activo"):
            metrics = row.get(head)
            if metrics is not None:
                print(f"  {head}: F1={metrics['f1']} IoU={metrics['iou']} cobertura={metrics['coverage_pct']:.2f}%")
    print(f"Informe: {output}")


if __name__ == "__main__":
    main()

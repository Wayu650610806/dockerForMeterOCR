"""
One-time export script: converts PyTorch .pt models to ONNX format.

Run this locally (requires: pip install ultralytics torch):
    python export_onnx.py

Output files are placed in weights/ alongside the .pt files:
    weights/model1_mvp100_yolo26m_obb_v2.onnx
    weights/model2_elec_mvp100_yolo11m_fix.onnx
    weights/model2_water_mvp100_yolo26m.onnx
    weights/model3_id_mvp100_yolo26m.onnx
    weights/model_names.json   ← class names used by inference_onnx.py

After exporting, commit the .onnx and model_names.json files to Git LFS
and push, so GitHub Actions can build the v3/v4 Docker images.
"""
import json
from pathlib import Path


MODELS = {
    "model1_mvp100_yolo26m_obb_v2":    {"pt": "model1_mvp100_yolo26m_obb_v2.pt",    "imgsz": 640},
    "model3_id_mvp100_yolo26m":         {"pt": "model3_id_mvp100_yolo26m.pt",         "imgsz": 640},
    "model2_elec_mvp100_yolo11m_fix":   {"pt": "model2_elec_mvp100_yolo11m_fix.pt",   "imgsz": 640},
    "model2_water_mvp100_yolo26m":      {"pt": "model2_water_mvp100_yolo26m.pt",       "imgsz": 640},
}


def main():
    from ultralytics import YOLO

    weights_dir = Path("weights")
    names_cfg   = {}

    for stem, cfg in MODELS.items():
        pt_path = weights_dir / cfg["pt"]
        if not pt_path.exists():
            print(f"  [SKIP] {pt_path} not found")
            continue

        print(f"Exporting {pt_path} …")
        model = YOLO(str(pt_path))
        model.export(
            format="onnx",
            imgsz=cfg["imgsz"],
            simplify=True,
            dynamic=False,
        )

        onnx_path = pt_path.with_suffix(".onnx")
        task      = model.overrides.get("task", "detect")
        names_cfg[stem] = {
            "task":  task,
            "names": model.names,   # {int: str}
            "nc":    len(model.names),
        }
        print(f"  → {onnx_path}  task={task}  classes={model.names}")

    names_file = weights_dir / "model_names.json"
    with open(names_file, "w", encoding="utf-8") as f:
        json.dump(names_cfg, f, indent=2, ensure_ascii=False)
    print(f"\nClass names saved to {names_file}")
    print("Done – commit weights/*.onnx and weights/model_names.json then push.")


if __name__ == "__main__":
    main()

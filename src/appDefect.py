from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import random
import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class DefectRow:
    wheel_id: str
    defect_type: str
    confidence: float
    area_ratio: float
    time: datetime

class Detector:
    def __init__(self, model_ref: str, allowed_defects: list[str]):
        self.model = YOLO(model_ref)
        self.allowed = {d.strip().lower() for d in allowed_defects}

    def predict_rows(self, img_path: str, conf: float) -> List[DefectRow]:
        p = Path(img_path)
        img = cv2.imread(str(p))
        if img is None:
            return []
        h, w = img.shape[:2]
        img_area = max(1, h * w)

        rows: List[DefectRow] = []
        results = self.model.predict(str(p), conf=conf, verbose=False)
        for r in results:
            if not r.boxes:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for b, c, k in zip(xyxy, confs, clss):
                name = r.names.get(int(k), "Unknown")
                if name.strip().lower() not in self.allowed:
                    continue
                x1, y1, x2, y2 = map(float, b)
                area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                rows.append(DefectRow(
                    wheel_id=p.stem,
                    defect_type=name,
                    confidence=float(c),
                    area_ratio=float(area / img_area),
                    time=datetime.now(),
                ))
        return rows

    def plot_with_boxes(self, img_path: str, conf: float) -> np.ndarray:
        results = self.model.predict(img_path, conf=conf, verbose=False)
        img = cv2.imread(img_path)
        return results[0].plot(line_width=3, labels=True) if len(results) else img

def list_images(folder: str) -> list[str]:
    p = Path(folder).expanduser().resolve()
    if not p.exists():
        return []
    exts = {".jpg", ".jpeg", ".png"}
    return [str(x.resolve()) for x in p.iterdir() if x.suffix.lower() in exts]

def find_image_by_stem(folder: str, stem: str) -> Optional[str]:
    p = Path(folder).expanduser().resolve()
    for ext in (".jpg", ".jpeg", ".png"):
        f = p / f"{stem}{ext}"
        if f.exists():
            return str(f.resolve())
    return None

def start_run_state(images: list[str]) -> dict:
    order = images[:]
    random.shuffle(order)
    return {
        "images_all": images,
        "order": order,
        "cursor": 0,
        "is_running": True,
    }

def advance_state(state: dict, batch_size: int) -> tuple[list[str], bool, dict]:
    start = int(state.get("cursor", 0))
    order = state.get("order", [])
    end = min(start + batch_size, len(order))
    batch = order[start:end]
    new_state = {**state, "cursor": end, "is_running": end < len(order)}
    finished = not new_state["is_running"]
    return batch, finished, new_state

"""
ONNX Runtime inference engine for MeterOCR.
Replaces ultralytics/torch with onnxruntime + numpy only.

Handles two ONNX output formats produced by ultralytics export:
  - E2E (end-to-end NMS already in model):
      detect: [1, max_det, 6]  values = [x1,y1,x2,y2, conf, cls_id]
      obb:    [1, max_det, 7]  values = [cx,cy,w,h, angle_rad, conf, cls_id]
  - Raw (no NMS, needs post-processing):
      detect: [1, 4+nc, N]     values = [cx,cy,w,h, cls_scores...]
      obb:    [1, 5+nc, N]     values = [cx,cy,w,h, angle_rad, cls_scores...]
"""
import os
import json
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Tensor-like wrapper
# Lets numpy arrays respond to .cpu() / .numpy() / .item() / [idx] / len()
# so that app.py code (written for ultralytics tensors) works unchanged.
# ──────────────────────────────────────────────────────────────────────────────
class _T:
    __slots__ = ("_d",)

    def __init__(self, data):
        self._d = np.asarray(data)

    def cpu(self):
        return self

    def numpy(self):
        return self._d

    def item(self):
        return self._d.flat[0]

    def __getitem__(self, idx):
        result = self._d[idx]
        return _T(result) if isinstance(result, np.ndarray) else result

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        for i in range(len(self._d)):
            yield _T(self._d[i])


# ──────────────────────────────────────────────────────────────────────────────
# Result wrapper classes  (mirror the ultralytics result interface)
# ──────────────────────────────────────────────────────────────────────────────
class ONNXBoxes:
    """Mimics ultralytics Boxes – consumed by parse_ocr_result()."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self._xyxy = np.asarray(xyxy, np.float32)
        self._conf = np.asarray(conf, np.float32)
        self._cls  = np.asarray(cls,  np.float32)

        xywh = self._xyxy.copy()
        xywh[:, 2] = self._xyxy[:, 2] - self._xyxy[:, 0]      # w
        xywh[:, 3] = self._xyxy[:, 3] - self._xyxy[:, 1]      # h
        xywh[:, 0] = self._xyxy[:, 0] + xywh[:, 2] / 2        # cx
        xywh[:, 1] = self._xyxy[:, 1] + xywh[:, 3] / 2        # cy
        self._xywh = xywh

    xyxy = property(lambda s: _T(s._xyxy))
    xywh = property(lambda s: _T(s._xywh))
    conf = property(lambda s: _T(s._conf))
    cls  = property(lambda s: _T(s._cls))

    def __len__(self):
        return len(self._xyxy)


class _OBBItem:
    """Single OBB detection – yielded when iterating ONNXOBBData."""
    __slots__ = ("_xyxyxyxy", "_conf", "_cls")

    def __init__(self, corners_4x2: np.ndarray, conf_val: float, cls_id: float):
        self._xyxyxyxy = np.array([corners_4x2], np.float32)   # [1, 4, 2]
        self._conf     = np.array([conf_val],    np.float32)   # [1]
        self._cls      = np.array([cls_id],      np.float32)   # [1]

    xyxyxyxy = property(lambda s: _T(s._xyxyxyxy))
    conf     = property(lambda s: _T(s._conf))
    cls      = property(lambda s: _T(s._cls))


class ONNXOBBData:
    """OBB container – iterated by extract_best_boxes() in app.py."""

    def __init__(self, corners: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self._corners = np.asarray(corners, np.float32)    # [N, 4, 2]
        self._conf    = np.asarray(conf,    np.float32)
        self._cls     = np.asarray(cls,     np.float32)

    def __iter__(self):
        for i in range(len(self._corners)):
            yield _OBBItem(self._corners[i], self._conf[i], self._cls[i])

    def __len__(self):
        return len(self._corners)


class ONNXDetectResult:
    def __init__(self, boxes: Optional[ONNXBoxes], orig_img: np.ndarray):
        self.boxes    = boxes if (boxes is not None and len(boxes) > 0) else None
        self.orig_img = orig_img


class ONNXOBBResult:
    def __init__(self, obb: ONNXOBBData, orig_img: np.ndarray):
        self.obb      = obb
        self.orig_img = orig_img


# ──────────────────────────────────────────────────────────────────────────────
# Image pre-processing
# ──────────────────────────────────────────────────────────────────────────────
def _letterbox(img: np.ndarray, new_shape=(640, 640)):
    h, w = img.shape[:2]
    ratio = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    dw = (new_shape[1] - nw) / 2
    dh = (new_shape[0] - nh) / 2

    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top,  bot   = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bot, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, ratio, dw, dh


def _preprocess(img: np.ndarray, imgsz: int = 640):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lb, ratio, dw, dh = _letterbox(rgb, (imgsz, imgsz))
    inp = (lb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]
    return inp, ratio, dw, dh


# ──────────────────────────────────────────────────────────────────────────────
# NMS (pure numpy, axis-aligned) – used only for raw-format models
# ──────────────────────────────────────────────────────────────────────────────
def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45):
    if len(boxes_xyxy) == 0:
        return []
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1  = np.maximum(x1[i], x1[order[1:]])
        yy1  = np.maximum(y1[i], y1[order[1:]])
        xx2  = np.minimum(x2[i], x2[order[1:]])
        yy2  = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1.0) * np.maximum(0.0, yy2 - yy1 + 1.0)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


# ──────────────────────────────────────────────────────────────────────────────
# OBB corner computation
# ──────────────────────────────────────────────────────────────────────────────
def _xywhr_to_corners(xywhr: np.ndarray) -> np.ndarray:
    """(cx, cy, w, h, angle_rad) → 4 corner points [N, 4, 2]."""
    if len(xywhr) == 0:
        return np.empty((0, 4, 2), np.float32)
    cx, cy, w, h, ang = (xywhr[:, i] for i in range(5))
    ca, sa = np.cos(ang), np.sin(ang)
    hw, hh = w / 2, h / 2
    lx = np.stack([-hw,  hw,  hw, -hw], axis=1)
    ly = np.stack([-hh, -hh,  hh,  hh], axis=1)
    rx = ca[:, None] * lx - sa[:, None] * ly + cx[:, None]
    ry = sa[:, None] * lx + ca[:, None] * ly + cy[:, None]
    return np.stack([rx, ry], axis=2).astype(np.float32)


def _undo_letterbox_boxes(boxes_xyxy, ratio, dw, dh, orig_hw):
    boxes = boxes_xyxy.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_hw[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_hw[0])
    return boxes


def _undo_letterbox_xywhr(xywhr, ratio, dw, dh):
    out = xywhr.copy()
    out[:, 0] = (out[:, 0] - dw) / ratio   # cx
    out[:, 1] = (out[:, 1] - dh) / ratio   # cy
    out[:, 2] = out[:, 2] / ratio            # w
    out[:, 3] = out[:, 3] / ratio            # h
    # angle is unchanged by uniform scaling
    return out


# ──────────────────────────────────────────────────────────────────────────────
# ONNX Model class
# ──────────────────────────────────────────────────────────────────────────────
class ONNXModel:
    """
    Drop-in replacement for ultralytics YOLO for inference.
    Supports task='detect' and task='obb'.

    Auto-detects ONNX output format at load time:
      E2E (end-to-end NMS baked in):  output[0].shape[1] > output[0].shape[2]
        detect: [1, max_det, 6]  → [x1,y1,x2,y2, conf, cls_id]
        obb:    [1, max_det, 7]  → [cx,cy,w,h, angle_rad, conf, cls_id]
      Raw (needs post-processing):     output[0].shape[1] < output[0].shape[2]
        detect: [1, 4+nc, N]
        obb:    [1, 5+nc, N]
    """

    def __init__(self, onnx_path: str, task: str, names: dict,
                 imgsz: int = 640, conf: float = 0.25, iou: float = 0.45):
        self.task  = task
        self.names = names
        self.imgsz = imgsz
        self._conf = conf
        self._iou  = iou

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = os.cpu_count() or 1

        providers = (["CPUExecutionProvider"]
                     if os.getenv("ORT_CPU_ONLY", "0") == "1"
                     else ort.get_available_providers())

        self._session    = ort.InferenceSession(str(onnx_path),
                                                sess_options=opts,
                                                providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._nc         = len(names)

        # Detect ONNX output format from shape.
        # Three possible layouts from ultralytics export:
        #   E2E (NMS baked in):      [1, max_det, geom+2]  e.g. [1, 300, 7] for OBB
        #   Raw needs-transpose:     [1, geom+nc, N]        e.g. [1, 9, 8400] for OBB nc=4
        #   Raw pre-transposed:      [1, N, geom+nc]        e.g. [1, 8400, 9] for OBB nc=4
        # The old heuristic (d1 > d2) can't distinguish E2E from pre-transposed raw
        # because both have the anchor/det count in d1. Use the column count (d2) instead.
        geom = 5 if task == "obb" else 4
        e2e_cols = geom + 2  # OBB: 7 = cx,cy,w,h,angle,conf,cls  |  Detect: 6 = x1,y1,x2,y2,conf,cls
        raw_cols  = geom + self._nc

        out_shape = self._session.get_outputs()[0].shape
        d1, d2 = out_shape[1], out_shape[2]

        if isinstance(d2, int) and d2 == e2e_cols:
            self._e2e            = True
            self._pre_transposed = False          # [1, max_det, e2e_cols]
        elif isinstance(d1, int) and d1 == raw_cols:
            self._e2e            = False
            self._pre_transposed = False          # [1, geom+nc, N] → needs .T
        else:
            self._e2e            = False
            self._pre_transposed = True           # [1, N, geom+nc] → already per-row

    def to(self, device):   # no-op for API compatibility with app.py
        return self

    def predict(self, img: np.ndarray, verbose: bool = False,
                conf: Optional[float] = None, iou: Optional[float] = None):
        conf_t = conf if conf is not None else self._conf
        iou_t  = iou  if iou  is not None else self._iou

        inp, ratio, dw, dh = _preprocess(img, self.imgsz)
        outputs = self._session.run(None, {self._input_name: inp})

        if self.task == "obb":
            fn = self._post_obb_e2e if self._e2e else self._post_obb_raw
        else:
            fn = self._post_detect_e2e if self._e2e else self._post_detect_raw
        return [fn(outputs, img, ratio, dw, dh, conf_t, iou_t)]

    # ── E2E detection: [1, max_det, 6]  [x1,y1,x2,y2, conf, cls_id] ─────────
    def _post_detect_e2e(self, outputs, orig_img, ratio, dw, dh, conf_t, iou_t):
        pred = outputs[0][0]                # [max_det, 6]
        mask = pred[:, 4] > conf_t
        pred = pred[mask]
        if len(pred) == 0:
            return ONNXDetectResult(None, orig_img)
        boxes = _undo_letterbox_boxes(pred[:, :4], ratio, dw, dh, orig_img.shape[:2])
        return ONNXDetectResult(ONNXBoxes(boxes, pred[:, 4], pred[:, 5].astype(int)), orig_img)

    # ── E2E OBB: [1, max_det, 7]  [cx,cy,w,h, angle_rad, conf, cls_id] ──────
    def _post_obb_e2e(self, outputs, orig_img, ratio, dw, dh, conf_t, iou_t):
        pred = outputs[0][0]                # [max_det, 7]
        mask = pred[:, 5] > conf_t
        pred = pred[mask]
        if len(pred) == 0:
            return ONNXOBBResult(
                ONNXOBBData(np.empty((0, 4, 2), np.float32), [], []), orig_img)
        xywhr   = _undo_letterbox_xywhr(pred[:, :5], ratio, dw, dh)
        corners = _xywhr_to_corners(xywhr)
        return ONNXOBBResult(
            ONNXOBBData(corners, pred[:, 5], pred[:, 6].astype(int)), orig_img)

    # ── Raw detection: [1, 4+nc, N] or [1, N, 4+nc] ─────────────────────────
    def _post_detect_raw(self, outputs, orig_img, ratio, dw, dh, conf_t, iou_t):
        raw  = outputs[0][0]
        pred = raw if self._pre_transposed else raw.T   # → [N, 4+nc]
        cls_scores = pred[:, 4:4 + self._nc]
        conf   = cls_scores.max(axis=1)
        cls_id = cls_scores.argmax(axis=1)
        mask   = conf > conf_t
        if not mask.any():
            return ONNXDetectResult(None, orig_img)
        pred, conf, cls_id = pred[mask], conf[mask], cls_id[mask]
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=1)
        boxes = _undo_letterbox_boxes(boxes, ratio, dw, dh, orig_img.shape[:2])
        keep  = _nms(boxes, conf, iou_t)
        if not keep:
            return ONNXDetectResult(None, orig_img)
        return ONNXDetectResult(
            ONNXBoxes(boxes[keep], conf[keep], cls_id[keep]), orig_img)

    # ── Raw OBB: [1, 5+nc, N] or [1, N, 5+nc] ───────────────────────────────
    def _post_obb_raw(self, outputs, orig_img, ratio, dw, dh, conf_t, iou_t):
        raw  = outputs[0][0]
        pred = raw if self._pre_transposed else raw.T   # → [N, 5+nc]
        cls_scores = pred[:, 5:5 + self._nc]
        conf   = cls_scores.max(axis=1)
        cls_id = cls_scores.argmax(axis=1)
        mask   = conf > conf_t
        if not mask.any():
            return ONNXOBBResult(
                ONNXOBBData(np.empty((0, 4, 2), np.float32), [], []), orig_img)
        pred, conf, cls_id = pred[mask], conf[mask], cls_id[mask]
        xywhr   = _undo_letterbox_xywhr(pred[:, :5], ratio, dw, dh)
        corners = _xywhr_to_corners(xywhr)
        x_min, y_min = corners[:, :, 0].min(axis=1), corners[:, :, 1].min(axis=1)
        x_max, y_max = corners[:, :, 0].max(axis=1), corners[:, :, 1].max(axis=1)
        aabb  = np.stack([x_min, y_min, x_max, y_max], axis=1)
        keep  = _nms(aabb, conf, iou_t)
        if not keep:
            return ONNXOBBResult(
                ONNXOBBData(np.empty((0, 4, 2), np.float32), [], []), orig_img)
        return ONNXOBBResult(
            ONNXOBBData(corners[keep], conf[keep], cls_id[keep]), orig_img)


# ──────────────────────────────────────────────────────────────────────────────
# Loader called from app.py when USE_ONNX=1
# ──────────────────────────────────────────────────────────────────────────────
def load_onnx_models(base_dir: Path) -> dict:
    names_file = base_dir / "weights" / "model_names.json"
    if not names_file.exists():
        raise RuntimeError(
            f"model_names.json not found at {names_file}. "
            "Run export_onnx.py first to generate ONNX weights."
        )

    with open(names_file, encoding="utf-8") as f:
        cfg = json.load(f)

    key_to_stem = {
        "obb":   "model1_mvp100_yolo26m_obb_v2",
        "id":    "model3_id_mvp100_yolo26m",
        "elec":  "model2_elec_mvp100_yolo11m_fix",
        "water": "model2_water_mvp100_yolo26m",
    }

    loaded, missing = {}, []
    for key, stem in key_to_stem.items():
        onnx_path = base_dir / "weights" / f"{stem}.onnx"
        if not onnx_path.exists():
            missing.append(str(onnx_path))
            continue
        model_cfg = cfg[stem]
        names     = {int(k): v for k, v in model_cfg["names"].items()}
        loaded[key] = ONNXModel(str(onnx_path), model_cfg["task"], names)

    if missing:
        raise RuntimeError(
            f"Missing ONNX weight files: {missing}. "
            "Run export_onnx.py first."
        )

    print(f"✅ All {len(loaded)} ONNX models loaded.")
    return loaded

import os
import cv2
import numpy as np
import torch
import httpx
import uuid
import asyncio
import json
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import uvicorn
from datetime import datetime, timedelta

# =========================================================
# คอนฟิกจำกัดการใช้ทรัพยากร (Rate Limit / Concurrency)
# =========================================================
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", 3))
process_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

AXIS_THRESHOLD = 15  # องศา: ถ้า angle < 15 ถือว่า axis-aligned

# ---------------------------------------------------------
# 1. DTO (Data Transfer Objects) สำหรับ Swagger Docs
# ---------------------------------------------------------
class SuccessData(BaseModel):
    meterId: str  = Field(..., description="เลขรหัสประจำมิเตอร์")
    value:   float = Field(..., description="ค่าตัวเลขบนหน้าปัด (แปลงเป็น float แล้ว)")

class SuccessResponse(BaseModel):
    """Response เมื่ออ่านค่ามิเตอร์สำเร็จ (HTTP 200)"""
    data:     SuccessData
    warnings: List[str] = Field(default=[], description="คำเตือนจากระบบ (ถ้ามี)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "อ่านได้ปกติ ไม่มี warning",
                    "value": {
                        "data": {"meterId": "1234567890", "value": 1234.5},
                        "warnings": []
                    }
                },
                {
                    "summary": "อ่านได้ แต่มี warning (rolling digit)",
                    "value": {
                        "data": {"meterId": "1234567890", "value": 1234.5},
                        "warnings": ["Col 3: Unclear stacked digits."]
                    }
                },
                {
                    "summary": "อ่านได้ แต่ไม่พบ ID",
                    "value": {
                        "data": {"meterId": "Unknown", "value": 1234.5},
                        "warnings": ["Only 'value' box found, cropping directly."]
                    }
                }
            ]
        }
    }

class FailResponse(BaseModel):
    """Response เมื่ออ่านค่ามิเตอร์ไม่ได้เลย (HTTP 422)"""
    message: str       = Field(..., description="ข้อความหลักแจ้งความล้มเหลว")
    details: List[str] = Field(default=[], description="""
        รายละเอียดข้อผิดพลาด รวบรวม Error ทั้งหมดที่ระบบสามารถแจ้งเตือนได้:

        [1] ข้อผิดพลาดระดับ OBB (หาตำแหน่งไม่เจอ)
        - "Target box not found." : ไม่พบกรอบมิเตอร์ ID หรือ Value ในภาพเลย

        [2] ข้อผิดพลาดระดับ OCR (อ่านตัวเลขไม่ได้)
        - "No digits found" : หาตัวเลขในกรอบไม่เจอเลย

        [3] คำเตือนเรื่องความแม่นยำ (Warnings)
        - "Col X: Unclear stacked digits." : พบตัวเลขซ้อนกันและระบุค่าที่ชัดเจนไม่ได้
        - "Col X: Too many digits (X)." : เจอตัวเลขซ้อนกันมากกว่า 3 ตัว (Noise)
        - "ID Warning: Length is X (<=4)" : ความยาว ID มี 4 ตัวหรือน้อยกว่า

        [4] ข้อผิดพลาดเรื่องความยาว (Length Checks)
        - "Length Error: Elec expected 6, found X" : ความยาวมิเตอร์ไฟไม่เท่ากับ 6
        - "Length Error: Water expected 4, 5, or 6, found X"
        - "Length Error: Water_7 expected 7, found X"
    """)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "ไม่พบมิเตอร์ในภาพ",
                    "value": {
                        "message": "อ่านค่ามิเตอร์ไม่ได้เลย กรุณาถ่ายรูปใหม่ (ไม่พบตำแหน่งมิเตอร์)",
                        "details": ["Target box not found.", "No digits found"]
                    }
                }
            ]
        }
    }

class OverlayData(BaseModel):
    overlay_top: int = Field(..., description="พิกัดจุดเริ่มต้นแกน Y (บนสุด)")
    overlay_left: int = Field(..., description="พิกัดจุดเริ่มต้นแกน X (ซ้ายสุด)")
    overlay_width: int = Field(..., description="ความกว้างของกรอบ (ต้องมากกว่า 0)")
    overlay_height: int = Field(..., description="ความสูงของกรอบ (ต้องมากกว่า 0)")

class ImageUrlRequest(BaseModel):
    image_url: HttpUrl = Field(..., description="URL ของรูปภาพที่ต้องการให้อ่านค่า", example="https://example.com/meter.jpg")
    overlay: Optional[OverlayData] = Field(None, description="(Optional) พิกัดตีกรอบเพื่อตัดรูปก่อนเข้า AI (ช่วยให้ AI เร็วและแม่นขึ้น)")
    gauge_min: Optional[int] = Field(None, description="(Optional) รองรับพารามิเตอร์ gauge_min ตามระบบเก่า")
    gauge_max: Optional[int] = Field(None, description="(Optional) รองรับพารามิเตอร์ gauge_max ตามระบบเก่า")

class AsyncCallbackRequest(ImageUrlRequest):
    callback_url: HttpUrl = Field(..., description="URL ของระบบคุณที่จะรอรับ JSON ผลลัพธ์เมื่อทำงานเสร็จ", example="https://your-app.com/api/receive-ocr")

class AsyncAcceptedResponse(BaseModel):
    status: str = Field("accepted", description="สถานะการรับงาน (accepted = รับเรื่องแล้ว)")
    task_id: str = Field(..., description="ID อ้างอิงสำหรับงานนี้", example="550e8400-e29b-41d4-a716-446655440000")
    message: str = Field("Processing started in background.", description="ข้อความแจ้งเตือน")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="ข้อความอธิบายสาเหตุของ Error", example="Invalid image format or cannot download.")

# ---------------------------------------------------------
# 2. SETUP & MODEL LOADING (Fail-Fast Validation)
# ---------------------------------------------------------
app = FastAPI(
    title="Digital Meter OCR API",
    description="API สำหรับอ่านเลขมิเตอร์ (Production Ready + Defensive Programming)",
    version="2.7.0"
)

# =========================================================
# ระบบจัดการข้อมูลชั่วคราว (Task Store) & GC
# =========================================================
task_store = {}

async def cleanup_task_store():
    """Background Task: ลบข้อมูลเก่าเกิน 24 ชั่วโมงทิ้งเพื่อประหยัด RAM"""
    while True:
        await asyncio.sleep(3600)
        now = datetime.now()
        keys_to_delete = []
        for task_id, data in list(task_store.items()):
            try:
                task_time = datetime.fromisoformat(data.get("timestamp", now.isoformat()))
                if now - task_time > timedelta(hours=24):
                    keys_to_delete.append(task_id)
            except:
                keys_to_delete.append(task_id)
        for k in keys_to_delete:
            task_store.pop(k, None)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_task_store())

# =========================================================
# AUTHENTICATION
# =========================================================
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
SECRET_API_KEY = os.environ.get("OCR_API_KEY")

if not SECRET_API_KEY:
    print("CRITICAL ERROR: 'OCR_API_KEY' environment variable is not set.")
    raise RuntimeError("Missing OCR_API_KEY")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != SECRET_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing X-API-Key")
    return api_key

# ---------------------------------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATHS = {
    "obb":   BASE_DIR / "weights" / "model1_mvp100_yolo26m_obb_v2.pt",
    "id":    BASE_DIR / "weights" / "model3_id_mvp100_yolo26m.pt",
    "elec":  BASE_DIR / "weights" / "model2_elec_mvp100_yolo11m_fix.pt",
    "water": BASE_DIR / "weights" / "model2_water_mvp100_yolo26m.pt",
}

models = {}
missing_models = []

for name, path in MODEL_PATHS.items():
    if path.exists():
        models[name] = YOLO(str(path)).to(DEVICE)
    else:
        missing_models.append(name)

if missing_models:
    error_msg = f"CRITICAL ERROR: Missing model weight files: {missing_models}. Please check the 'weights' directory."
    print(error_msg)
    raise RuntimeError(error_msg)
else:
    print(f"✅ All {len(models)} models loaded successfully on {DEVICE.upper()}.")

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------
async def download_image_from_url(url: str) -> bytes:
    """ดาวน์โหลดรูปภาพจาก URL"""
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(str(url), timeout=15.0)
            response.raise_for_status()
            return response.content
    except Exception as e:
        raise ValueError(f"Cannot download image from {url}: {e}")

# =========================================================
# NEW: Rotation & Crop Helpers (แทน order_points เดิม)
# =========================================================

def get_long_axis_angle(pts: np.ndarray) -> float:
    """
    หามุม (องศา) ของด้านที่ยาวที่สุดของ OBB polygon
    คืนค่าในช่วง [-90, 90)
    """
    edges = []
    for i in range(4):
        p1, p2 = pts[i], pts[(i + 1) % 4]
        dx, dy = p2 - p1
        edges.append((np.hypot(dx, dy), np.degrees(np.arctan2(dy, dx))))
    edges.sort(reverse=True)
    angle = edges[0][1] % 180
    if angle > 90:
        angle -= 180
    return angle


def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    หมุนภาพรอบจุดกึ่งกลาง ขยาย canvas ให้พอดีเพื่อไม่ให้ภาพถูกตัด
    """
    if angle_deg == 0:
        return img.copy()
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin_a + w * cos_a)
    nh = int(h * cos_a + w * sin_a)
    M[0, 2] += nw / 2 - cx
    M[1, 2] += nh / 2 - cy
    return cv2.warpAffine(img, M, (nw, nh),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(128, 128, 128))


def crop_obb_box(img: np.ndarray, pts: np.ndarray) -> Optional[np.ndarray]:
    """
    Crop ภาพจาก OBB polygon อย่างถูกต้อง:
    1. Angle-sort มุม 4 จุด
    2. บังคับให้ด้านยาว = width (ไม่ให้ภาพออกมาตั้งฉาก)
    3. ตรวจสอบ Upside-down แล้วพลิก 180° ถ้าจำเป็น
    """
    pts = np.array(pts, dtype=np.float32)

    # 1. Angle-sort (เรียงตามเข็มนาฬิกา)
    cx, cy = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    pts = pts[np.argsort(angles)]

    p0, p1, p2, p3 = pts

    # 2. บังคับให้ด้านที่ยาวกว่า = width เสมอ
    d01 = np.linalg.norm(p1 - p0)
    d12 = np.linalg.norm(p2 - p1)

    if d01 >= d12:
        w, h = int(d01), int(d12)
        src_pts = np.array([p0, p1, p2, p3], dtype=np.float32)
    else:
        w, h = int(d12), int(d01)
        src_pts = np.array([p1, p2, p3, p0], dtype=np.float32)

    if w <= 0 or h <= 0:
        return None

    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst)
    crop = cv2.warpPerspective(img, M, (w, h))

    # 3. ตรวจ Upside-down: ขอบบนควรมี Y น้อยกว่าขอบล่างในภาพต้นฉบับ
    top_y    = (src_pts[0][1] + src_pts[1][1]) / 2.0
    bottom_y = (src_pts[2][1] + src_pts[3][1]) / 2.0

    if top_y > bottom_y:
        crop = cv2.rotate(crop, cv2.ROTATE_180)
    elif top_y == bottom_y:
        # กล่องตั้งฉาก 90° พอดี → เช็คจากแกน X แทน
        top_x    = (src_pts[0][0] + src_pts[1][0]) / 2.0
        bottom_x = (src_pts[2][0] + src_pts[3][0]) / 2.0
        if top_x > bottom_x:
            crop = cv2.rotate(crop, cv2.ROTATE_180)

    return crop


def extract_best_boxes(obb_result, model) -> dict:
    """
    ดึง box ที่ confidence สูงสุดของแต่ละ class (id / value)
    คืน dict: {'id': {...}, 'value': {...}}
    """
    boxes = {}
    obb_data = getattr(obb_result, "obb", None)
    if obb_data is None:
        return boxes
    for obb in obb_data:
        name = model.names[int(obb.cls[0].item())].lower()
        conf = float(obb.conf[0].item())
        pts  = obb.xyxyxyxy[0].cpu().numpy()
        key  = 'value' if 'value' in name else ('id' if 'id' == name else None)
        if key and (key not in boxes or conf > boxes[key]['conf']):
            boxes[key] = {'pts': pts, 'conf': conf, 'name': name}
    return boxes


def get_rotation_for_arrangement(boxes: dict) -> int:
    """
    เทียบตำแหน่ง id กับ value แล้วคืนองศาที่ต้องหมุน
    สมมติ canonical = id อยู่ใต้ value (id_below = 0°)
    """
    val_c = boxes['value']['pts'].mean(axis=0)
    id_c  = boxes['id']['pts'].mean(axis=0)
    dx = id_c[0] - val_c[0]
    dy = id_c[1] - val_c[1]

    if abs(dx) >= abs(dy):
        arrangement = 'id_left' if dx < 0 else 'id_right'
    else:
        arrangement = 'id_below' if dy > 0 else 'id_above'

    return {'id_below': 0, 'id_above': 180, 'id_left': +90, 'id_right': -90}[arrangement]

# =========================================================
# END NEW HELPERS
# =========================================================

def parse_ocr_result(boxes, cls_name, names_dict):
    if boxes is None or len(boxes) == 0:
        return "0", ["No digits found"]

    cls_lower = cls_name.lower()
    xywh = boxes.xywh.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()

    all_detections = []
    for i in range(len(boxes)):
        all_detections.append({
            'x_c': xywh[i][0], 'y_c': xywh[i][1],
            'w': xywh[i][2], 'h': xywh[i][3],
            'conf': confs[i], 'name': names_dict[cls_ids[i]]
        })

    avg_h = np.mean([d['h'] for d in all_detections])
    median_y = np.median([d['y_c'] for d in all_detections])
    row_detections = [d for d in all_detections if abs(d['y_c'] - median_y) < (avg_h * 0.6)]
    if not row_detections:
        row_detections = all_detections

    if not row_detections:
        return "0", ["No valid digits found after row filtering."]

    row_detections.sort(key=lambda d: d['x_c'])
    avg_w = np.mean([d['w'] for d in row_detections])
    x_threshold = avg_w * 0.5

    columns = []
    curr_col = [row_detections[0]]
    for i in range(1, len(row_detections)):
        if abs(row_detections[i]['x_c'] - curr_col[-1]['x_c']) < x_threshold:
            curr_col.append(row_detections[i])
        else:
            columns.append(curr_col)
            curr_col = [row_detections[i]]
    columns.append(curr_col)

    final_text = ""
    warnings = []

    if 'electric' in cls_lower or 'elcetric' in cls_lower:
        selected_cols = columns[:6]
        raw_val = "".join([str(max(col, key=lambda d: d['conf'])['name']) for col in selected_cols])
        if len(raw_val) != 6:
            warnings.append(f"Length Error: Elec expected 6, found {len(raw_val)}")
        if len(raw_val) >= 1:
            final_text = raw_val[:-1] + "." + raw_val[-1:]

    elif 'water' in cls_lower:
        expected_cols = 7 if '7' in cls_lower else (6 if len(columns) > 4 else 4)
        selected_cols = columns[:expected_cols]
        raw_val = ""

        for idx, col in enumerate(selected_cols):
            if len(col) == 1:
                raw_val += str(col[0]['name'])
            elif len(col) >= 3:
                warnings.append(f"Col {idx+1}: Too many digits ({len(col)}). Using highest conf.")
                best = max(col, key=lambda d: d['conf'])
                raw_val += str(best['name'])
            elif len(col) == 2:
                by_y = sorted(col, key=lambda d: d['y_c'])
                v1, v2 = int(by_y[0]['name']), int(by_y[1]['name'])
                is_rolling = (abs(v1 - v2) == 1) or ({v1, v2} == {0, 9})
                if is_rolling:
                    raw_val += str(v1)
                else:
                    warnings.append(f"Col {idx+1}: Unclear stacked digits.")
                    best = max(col, key=lambda d: d['conf'])
                    raw_val += str(best['name'])

        if '7' in cls_lower:
            if len(raw_val) != 7:
                warnings.append(f"Length Error: Water_7 expected 7, found {len(raw_val)}")
            if len(raw_val) >= 3:
                final_text = raw_val[:-3] + "." + raw_val[-3:]
        else:
            final_text = raw_val
            if len(raw_val) not in [4, 5, 6]:
                warnings.append(f"Length Error: Water expected 4, 5, or 6, found {len(raw_val)}")

    else:
        raw_val = "".join([str(max(col, key=lambda d: d['conf'])['name']) for col in columns])
        final_text = raw_val
        if len(raw_val) <= 4:
            warnings.append(f"ID Warning: Length is {len(raw_val)} (<=4)")

    if cls_lower != 'id':
        try:
            if '.' in final_text:
                final_text = str(float(final_text))
            else:
                final_text = str(int(final_text))
        except:
            pass

    return final_text, warnings

# ---------------------------------------------------------
# 4. CORE PROCESSING (Updated with Rotation Pipeline)
# ---------------------------------------------------------
async def process_single_image(img_bytes: bytes, overlay: Optional[OverlayData] = None) -> tuple:
    """
    ประมวลผลรูปภาพก้อนเดียว คืนค่า (final_id, final_val, all_warns)
    
    Pipeline:
    1. Decode + Overlay crop (optional)
    2. OBB detect
    3a. พบทั้ง id+value และ axis-aligned → ตรวจ arrangement → หมุนภาพ → re-detect → crop
    3b. พบทั้ง id+value และ tilted → crop ตรงจาก OBB warp (crop_obb_box จัดการ)
    3c. พบกล่องเดียว → crop ตรงจากภาพ
    3d. ไม่พบเลย → Fallback ยิง elec+water ทั้งภาพ
    """
    async with process_semaphore:
        if len(models) < 4:
            raise HTTPException(status_code=500, detail="Models not fully loaded.")

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image format or cannot decode.")

        # ── Overlay Crop ─────────────────────────────────────────
        if overlay is not None:
            if overlay.overlay_width <= 0 or overlay.overlay_height <= 0:
                raise ValueError("Invalid overlay size: width and height must be greater than 0.")
            h_img, w_img = img.shape[:2]
            top    = max(0, overlay.overlay_top)
            left   = max(0, overlay.overlay_left)
            bottom = min(h_img, top + overlay.overlay_height)
            right  = min(w_img, left + overlay.overlay_width)
            if right > left and bottom > top:
                img = img[top:bottom, left:right]

        # ── Stage 1: OBB Detection ───────────────────────────────
        obb_res = models['obb'].predict(img, verbose=False)[0]
        boxes   = extract_best_boxes(obb_res, models['obb'])

        final_id, final_val, all_warns = "0", "0", []

        # ── Stage 2: กำหนด working image และ boxes สำหรับ crop ──
        if 'id' in boxes and 'value' in boxes:
            # --- มีทั้ง 2 กล่อง: ตรวจ tilt ---
            a_val = get_long_axis_angle(boxes['value']['pts'])
            a_id  = get_long_axis_angle(boxes['id']['pts'])
            avg   = (a_val + a_id) / 2

            near_h    = abs(avg) < AXIS_THRESHOLD
            near_v    = abs(abs(avg) - 90) < AXIS_THRESHOLD
            is_tilted = not (near_h or near_v)

            if is_tilted:
                # Case B: ภาพเอียง → crop ตรงจาก OBB (warp จัดการ rotation)
                work_img   = img
                work_boxes = boxes
                all_warns.append(f"Image tilted ({avg:.1f}°), using direct OBB crop.")
            else:
                # Case C: Axis-aligned → ตรวจ arrangement → หมุน → re-detect
                rot_deg = get_rotation_for_arrangement(boxes)
                if rot_deg != 0:
                    all_warns.append(f"Image rotated {rot_deg}° to normalize orientation.")
                    work_img  = rotate_image(img, rot_deg)
                    res2      = models['obb'].predict(work_img, verbose=False)[0]
                    work_boxes = extract_best_boxes(res2, models['obb'])
                    # ถ้า re-detect ไม่เจอ fallback ใช้ของเดิม
                    if not work_boxes:
                        all_warns.append("Re-detect after rotation found nothing, using original boxes.")
                        work_img   = img
                        work_boxes = boxes
                else:
                    # id_below = canonical orientation ไม่ต้องหมุน
                    work_img   = img
                    work_boxes = boxes

            # --- Crop แต่ละกล่องด้วย crop_obb_box ---
            for key in ('id', 'value'):
                if key not in work_boxes:
                    all_warns.append(f"Box '{key}' not found after rotation.")
                    continue

                target   = work_boxes[key]
                cropped  = crop_obb_box(work_img, target['pts'])

                if cropped is None:
                    all_warns.append(f"Invalid OBB dimensions for '{key}'.")
                    continue

                box_name = target['name']
                ocr_m    = (models['id'] if key == 'id'
                            else models['elec'] if 'elec' in box_name
                            else models['water'])
                res      = ocr_m.predict(cropped, verbose=False)
                text, warns = parse_ocr_result(res[0].boxes, box_name, ocr_m.names)

                if key == 'id':
                    final_id = text
                else:
                    final_val = text
                all_warns.extend(warns)

        elif len(boxes) == 1:
            # Case A: พบกล่องเดียว → crop ตรงจากภาพเดิม
            key    = list(boxes.keys())[0]
            target = boxes[key]
            all_warns.append(f"Only '{key}' box found, cropping directly.")

            cropped = crop_obb_box(img, target['pts'])
            if cropped is not None:
                box_name = target['name']
                ocr_m    = (models['id'] if key == 'id'
                            else models['elec'] if 'elec' in box_name
                            else models['water'])
                res = ocr_m.predict(cropped, verbose=False)
                text, warns = parse_ocr_result(res[0].boxes, box_name, ocr_m.names)
                if key == 'id':
                    final_id = text
                else:
                    final_val = text
                all_warns.extend(warns)
            else:
                all_warns.append(f"Crop failed for single box '{key}'.")

        else:
            # Case D: ไม่พบกล่องเลย → Fallback
            all_warns.append("Target box not found.")

            # อ่าน ID จากทั้งภาพ
            res_id = models['id'].predict(img, verbose=False)
            final_id, warns_id = parse_ocr_result(res_id[0].boxes, 'id', models['id'].names)
            all_warns.extend(warns_id)

            # เลือก elec หรือ water จาก box count
            res_elec  = models['elec'].predict(img, verbose=False)
            res_water = models['water'].predict(img, verbose=False)
            len_elec  = len(res_elec[0].boxes)  if res_elec[0].boxes  is not None else 0
            len_water = len(res_water[0].boxes) if res_water[0].boxes is not None else 0

            if len_elec > len_water:
                final_val, warns_val = parse_ocr_result(res_elec[0].boxes, 'electric', models['elec'].names)
            else:
                final_val, warns_val = parse_ocr_result(res_water[0].boxes, 'water', models['water'].names)
            all_warns.extend(warns_val)

        return final_id, final_val, all_warns

# ---------------------------------------------------------
# 5. FORMATTER & ENDPOINTS
# ---------------------------------------------------------
def format_response(final_id: str, final_val: str, all_warns: List[str]):
    """
    422: ทั้ง ID และ Value = "0" (อ่านไม่ได้เลย)
    200: อ่านได้อย่างน้อย 1 อย่าง
    """
    is_completely_lost = (final_id == "0" and final_val == "0")

    if is_completely_lost:
        return JSONResponse(
            status_code=422,
            content={
                "message": "อ่านค่ามิเตอร์ไม่ได้เลย กรุณาถ่ายรูปใหม่ (ไม่พบตำแหน่งมิเตอร์)",
                "details": all_warns
            }
        )

    try:
        num_val = float(final_val)
    except:
        num_val = 0.0

    return {
        "data": {
            "meterId": final_id if final_id != "0" else "Unknown",
            "value":   num_val
        },
        "warnings": all_warns
    }

# ==========================================
# กลุ่มที่ 1: อัปโหลดรูปภาพตรงๆ
# ==========================================
@app.post("/uploadimage", tags=["1. File Upload"],
          responses={200: {"model": SuccessResponse}, 422: {"model": FailResponse}})
async def upload_image(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    """อัปโหลดไฟล์รูปภาพโดยตรง แล้วรอรับผลลัพธ์ทันที"""
    try:
        contents = await file.read()
        final_id, final_val, all_warns = await process_single_image(contents)
        return format_response(final_id, final_val, all_warns)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/uploadimage/batch", tags=["1. File Upload"])
async def upload_image_batch(files: List[UploadFile] = File(...), api_key: str = Depends(verify_api_key)):
    """อัปโหลดหลายไฟล์พร้อมกัน คืนค่าเป็น List ของผลลัพธ์"""
    results = []
    for f in files:
        try:
            res = format_response(*(await process_single_image(await f.read())))
            results.append(res if isinstance(res, dict) else json.loads(res.body))
        except Exception as e:
            results.append({"message": "อ่านไม่ได้", "details": [str(e)]})
    return results

# ==========================================
# กลุ่มที่ 2: รับ URL (Synchronous)
# ==========================================
@app.post("/imageurl", tags=["2. Image URL"],
          responses={200: {"model": SuccessResponse}, 422: {"model": FailResponse}})
async def image_url_sync(payload: ImageUrlRequest, api_key: str = Depends(verify_api_key)):
    """ส่งลิงก์รูปภาพมาให้เซิร์ฟเวอร์ เซิร์ฟเวอร์จะโหลดรูปแล้วประมวลผลให้ทันที"""
    try:
        img_bytes = await download_image_from_url(str(payload.image_url))
        final_id, final_val, all_warns = await process_single_image(img_bytes, payload.overlay)
        return format_response(final_id, final_val, all_warns)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/imageurl/batch", tags=["2. Image URL"])
async def image_url_sync_batch(payloads: List[ImageUrlRequest], api_key: str = Depends(verify_api_key)):
    """รับลิงก์รูปภาพหลายรูปรวดเดียว รอรับผลลัพธ์เป็น Array"""
    results = []
    for p in payloads:
        try:
            img_bytes = await download_image_from_url(str(p.image_url))
            res = format_response(*(await process_single_image(img_bytes, p.overlay)))
            results.append(res if isinstance(res, dict) else json.loads(res.body))
        except Exception as e:
            results.append({"message": "อ่านไม่ได้", "details": [str(e)]})
    return results

# ==========================================
# กลุ่มที่ 3: Async Callback
# ==========================================
async def run_url_and_callback(task_id: str, callback_url: str, requests_data: List[dict], is_batch: bool):
    """รันเบื้องหลัง: ดาวน์โหลด → ประมวลผล → Retry ส่งผลกลับ"""
    results = []
    for req in requests_data:
        try:
            img_bytes = await download_image_from_url(req['url'])
            res = format_response(*(await process_single_image(img_bytes, req['overlay'])))
            results.append(res if isinstance(res, dict) else json.loads(res.body))
        except Exception as e:
            results.append({"message": "Error", "details": [str(e)]})

    payload = {
        "task_id":   task_id,
        "status":    "completed",
        "results":   results if is_batch else results[0],
        "timestamp": datetime.now().isoformat()
    }

    MAX_TASKS = 1000
    if len(task_store) > MAX_TASKS:
        task_store.pop(next(iter(task_store)))
    task_store[task_id] = payload

    async with httpx.AsyncClient() as client:
        for attempt in range(1, 4):
            try:
                response = await client.post(str(callback_url), json=payload, timeout=10.0)
                response.raise_for_status()
                task_store[task_id]["status"] = "callback_success"
                return
            except:
                if attempt == 3:
                    task_store[task_id]["status"] = "callback_failed"
                else:
                    await asyncio.sleep(2 ** attempt)

@app.post("/async-callback/imageurl", response_model=AsyncAcceptedResponse, tags=["3. Async Callback"])
async def async_callback_single(payload: AsyncCallbackRequest, background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key)):
    """ส่งลิงก์รูปภาพมา ระบบรับเรื่องทันที แล้ว POST ผลกลับไปที่ callback_url เมื่อเสร็จ"""
    task_id = str(uuid.uuid4())
    req_data = [{"url": str(payload.image_url), "overlay": payload.overlay}]
    background_tasks.add_task(run_url_and_callback, task_id, str(payload.callback_url), req_data, False)
    return AsyncAcceptedResponse(task_id=task_id)

@app.post("/async-callback/imageurl/batch", response_model=AsyncAcceptedResponse, tags=["3. Async Callback"])
async def async_callback_batch(payloads: List[AsyncCallbackRequest], background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key)):
    """(Batch) รับหลาย URL พร้อมกัน ผลลัพธ์ที่ส่งกลับไปตรง results จะเป็น Array"""
    if not payloads:
        raise HTTPException(status_code=400, detail="Empty list")
    if len({str(p.callback_url) for p in payloads}) > 1:
        raise HTTPException(status_code=400, detail="Shared callback_url required")
    task_id = str(uuid.uuid4())
    req_data = [{"url": str(p.image_url), "overlay": p.overlay} for p in payloads]
    background_tasks.add_task(run_url_and_callback, task_id, str(payloads[0].callback_url), req_data, True)
    return AsyncAcceptedResponse(task_id=task_id)

@app.get("/tasks/{task_id}", tags=["3. Async Callback"])
async def get_task_result(task_id: str, api_key: str = Depends(verify_api_key)):
    """ดึงผลลัพธ์ย้อนหลัง (ระบบเก็บข้อมูลไว้ชั่วคราว 24 ชั่วโมง)"""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found or expired.")
    return task_store[task_id]

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
import io
import json
import math
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# Optional chart/model dependencies
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ------------------------------------------------------------
# App configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Sonar Vision",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Visual system
# ------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --bg: #061017;
  --bg2: #081923;
  --panel: #0b202b;
  --panel2: #0d2733;
  --line: #173744;
  --text: #eef8fb;
  --muted: #8ba8b3;
  --cyan: #36d7e8;
  --cyan2: #159db1;
  --green: #45e3a2;
  --amber: #ffc857;
  --red: #ff647c;
  --purple: #9c8cff;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
  background: var(--bg) !important;
  color: var(--text) !important;
}
[data-testid="stHeader"] { background: rgba(6,16,23,.94) !important; }
[data-testid="stToolbar"] { visibility: hidden; height: 0 !important; }
[data-testid="stSidebar"] {
  background: #07151e !important;
  border-right: 1px solid #112d39 !important;
}
[data-testid="stSidebar"] * { color: #dcecf0 !important; }
.block-container { padding-top: 1.4rem !important; padding-bottom: 2.5rem !important; max-width: 1500px; }

h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: var(--text) !important; letter-spacing: -.02em; }
body, p, label, div, span, button, input, textarea { font-family: 'Inter', sans-serif; }

.hero {
  display:flex; justify-content:space-between; align-items:flex-start; gap:24px;
  padding: 10px 0 18px 0;
}
.brand-kicker { color:#4fb8c7; font-size:11px; font-weight:700; letter-spacing:.22em; text-transform:uppercase; }
.brand-title { font-size:42px; line-height:1.0; font-weight:800; margin-top:8px; }
.brand-sub { color:#8eabb6; margin-top:10px; font-size:14px; max-width:780px; }
.status-pill {
  display:inline-flex; align-items:center; gap:8px; margin-top:16px;
  padding:8px 12px; border:1px solid #174b55; border-radius:999px;
  background:#09242d; color:#69e8c1; font-size:11px; font-weight:700; letter-spacing:.08em;
}
.status-dot { width:7px; height:7px; background:#45e3a2; border-radius:50%; box-shadow:0 0 12px #45e3a2; }

.section-title { font-family:'Space Grotesk'; font-size:20px; font-weight:700; margin:18px 0 10px; }
.section-sub { color:#7896a1; font-size:12px; margin-top:-4px; margin-bottom:14px; }

.metric-card {
  background: linear-gradient(180deg, #0b202b 0%, #091b24 100%);
  border:1px solid #153844; border-radius:14px; padding:15px 16px; min-height:100px;
}
.metric-label { color:#75939e; font-size:10px; text-transform:uppercase; letter-spacing:.12em; font-weight:700; }
.metric-value { font-family:'Space Grotesk'; font-size:30px; font-weight:700; margin-top:5px; }
.metric-delta { color:#4fdca9; font-size:11px; margin-top:3px; }

.panel {
  background:#091a23; border:1px solid #153541; border-radius:16px; padding:18px;
}
.panel-tight { padding:12px 14px; }
.smallcap { color:#6f8c98; font-size:10px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }

[data-testid="stFileUploaderDropzone"] {
  background:#081923 !important; border:1px dashed #24505d !important; border-radius:14px !important;
}
[data-testid="stFileUploaderDropzone"] button { background:#11303a !important; border:1px solid #245462 !important; color:#dffaff !important; }

.stButton > button {
  border-radius:10px !important; border:1px solid #1b5663 !important;
  background:#0d2a34 !important; color:#e7fbff !important; font-weight:700 !important;
  min-height:42px !important;
}
.stButton > button:hover { border-color:#39cfe0 !important; background:#103944 !important; }
div[data-testid="stFormSubmitButton"] > button {
  background:linear-gradient(90deg,#0f8699,#1ab7ca) !important; border:none !important;
}

.stTabs [data-baseweb="tab-list"] { gap:8px; background:#081923; padding:5px; border-radius:11px; border:1px solid #143541; }
.stTabs [data-baseweb="tab"] { border-radius:8px; color:#7896a1; padding:9px 15px; }
.stTabs [aria-selected="true"] { background:#123641 !important; color:#e8fcff !important; }

[data-testid="stDataFrame"] { border:1px solid #173744; border-radius:12px; overflow:hidden; }

.legend { display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 12px; }
.legend-item { display:flex; align-items:center; gap:6px; padding:6px 9px; background:#0b2029; border:1px solid #153944; border-radius:999px; color:#a9c2ca; font-size:10px; }
.legend-dot { width:8px; height:8px; border-radius:50%; }

.target-row { padding:10px 0; border-bottom:1px solid #132e39; }
.target-id { color:#e9f8fb; font-weight:700; font-size:12px; }
.target-class { color:#83a4af; font-size:11px; margin-top:2px; }
.conf-pill { font-size:10px; font-weight:700; padding:4px 7px; border-radius:6px; }

.footer { color:#4f6c76; font-size:10px; padding-top:25px; text-align:center; }

/* Make Streamlit's default gray surfaces dark */
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"] {
  background: transparent !important;
  border-color: #173744 !important;
}
[data-baseweb="select"] > div, [data-baseweb="input"] > div {
  background:#0a1d27 !important; border-color:#1a3e4a !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Taxonomy / styling
# ------------------------------------------------------------
CLASSES = [
    "Shipwreck",
    "Pipe / Cylinder",
    "Entangled Net / Debris",
    "Marine Debris",
    "Natural Seafloor",
    "Unknown Anomaly",
]
TARGET_CLASSES = CLASSES[:4] + [CLASSES[-1]]
COLORS = {
    "Shipwreck": "#ff647c",
    "Pipe / Cylinder": "#ffc857",
    "Entangled Net / Debris": "#36d7e8",
    "Marine Debris": "#9c8cff",
    "Natural Seafloor": "#6e8992",
    "Unknown Anomaly": "#b8c5ca",
}
PRIORITY = {
    "Shipwreck": "HIGH",
    "Pipe / Cylinder": "HIGH",
    "Entangled Net / Debris": "HIGH",
    "Marine Debris": "MEDIUM",
    "Unknown Anomaly": "REVIEW",
    "Natural Seafloor": "LOW",
}

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % tuple(int(max(0, min(255, x))) for x in rgb)


def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def image_to_gray(img):
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)


def normalize01(a):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, (2, 98))
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def preprocess_sonar(rgb, denoise=5, contrast=2.4):
    gray = image_to_gray(rgb)
    if denoise > 0:
        gray = cv2.medianBlur(gray, max(3, int(denoise) | 1))
    clahe = cv2.createCLAHE(clipLimit=float(contrast), tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Light background suppression for sonar-like imagery
    bg = cv2.GaussianBlur(enhanced, (0, 0), 11)
    high = cv2.addWeighted(enhanced, 1.7, bg, -0.7, 0)
    high = cv2.normalize(high, None, 0, 255, cv2.NORM_MINMAX)
    return gray, enhanced, high.astype(np.uint8)


def shadow_score(gray, x, y, w, h):
    H, W = gray.shape
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    patch = gray[y:y2, x:x2]
    if patch.size == 0:
        return 0.0
    right = gray[y:y2, x2:min(W, x2 + max(8, w // 2))]
    below = gray[y2:min(H, y2 + max(8, h // 2)), x:x2]
    obj_mean = float(np.mean(patch))
    vals = []
    if right.size:
        vals.append(float(np.mean(right)))
    if below.size:
        vals.append(float(np.mean(below)))
    if not vals:
        return 0.0
    neigh = float(np.mean(vals))
    return float(np.clip((obj_mean - neigh) / 70.0, 0, 1))


def texture_score(patch):
    if patch.size == 0:
        return 0.0
    lap = cv2.Laplacian(patch, cv2.CV_32F)
    return float(np.clip(np.std(lap) / 40.0, 0, 1))


def contour_candidates(enhanced, min_area=90, sensitivity=0.72):
    # Sensitivity controls percentile threshold. Higher sensitivity = fewer candidates.
    percentile = 98.2 - (float(sensitivity) - 0.5) * 9.0
    percentile = float(np.clip(percentile, 92.0, 99.4))
    thr = np.percentile(enhanced, percentile)
    mask = (enhanced >= thr).astype(np.uint8) * 255

    # Remove tiny speckles and connect fragmented returns.
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = enhanced.shape
    out = []
    max_area = H * W * 0.12
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 6 or h < 6:
            continue
        border_touch = x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2
        if border_touch and area < min_area * 2:
            continue
        peri = max(cv2.arcLength(c, True), 1.0)
        circularity = float(4 * math.pi * area / (peri * peri))
        aspect = max(w, h) / max(1, min(w, h))
        extent = area / max(1, w * h)
        hull = cv2.convexHull(c)
        hull_area = max(cv2.contourArea(hull), 1.0)
        solidity = float(area / hull_area)
        out.append({
            "contour": c,
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "area": float(area), "aspect": float(aspect),
            "circularity": float(circularity), "extent": float(extent),
            "solidity": float(solidity), "border": border_touch,
        })

    # Non-max suppression for overlapping sonar blobs.
    out = sorted(out, key=lambda d: d["area"], reverse=True)
    kept = []
    for d in out:
        x1, y1, x2, y2 = d["x"], d["y"], d["x"] + d["w"], d["y"] + d["h"]
        duplicate = False
        for k in kept:
            a1, b1, a2, b2 = k["x"], k["y"], k["x"] + k["w"], k["y"] + k["h"]
            ix1, iy1, ix2, iy2 = max(x1, a1), max(y1, b1), min(x2, a2), min(y2, b2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = d["w"] * d["h"] + k["w"] * k["h"] - inter
            iou = inter / max(union, 1)
            if iou > 0.45:
                duplicate = True
                break
        if not duplicate:
            kept.append(d)
    return kept, mask


def classify_heuristic(d, gray, enhanced):
    x, y, w, h = d["x"], d["y"], d["w"], d["h"]
    H, W = gray.shape
    patch = gray[y:min(H, y+h), x:min(W, x+w)]
    mean = float(np.mean(patch)) if patch.size else 0
    std = float(np.std(patch)) if patch.size else 0
    shadow = shadow_score(gray, x, y, w, h)
    tex = texture_score(patch)
    aspect = d["aspect"]
    circ = d["circularity"]
    extent = d["extent"]
    solidity = d["solidity"]
    area_ratio = d["area"] / max(1, H * W)

    # Scores are heuristic evidence, not calibrated probabilities.
    scores = {
        "Shipwreck": 0.0,
        "Pipe / Cylinder": 0.0,
        "Entangled Net / Debris": 0.0,
        "Marine Debris": 0.0,
        "Unknown Anomaly": 0.0,
    }

    # Shipwreck: large, irregular, textured, often accompanied by acoustic shadow.
    scores["Shipwreck"] = (
        0.34 * np.clip(area_ratio / 0.025, 0, 1)
        + 0.20 * np.clip(tex, 0, 1)
        + 0.20 * np.clip(shadow, 0, 1)
        + 0.14 * (1 - np.clip(circ, 0, 1))
        + 0.12 * (1 - np.clip(solidity, 0, 1))
    )

    # Pipe/cylinder: elongated, compact and comparatively solid.
    scores["Pipe / Cylinder"] = (
        0.45 * np.clip((aspect - 2.0) / 5.0, 0, 1)
        + 0.25 * np.clip(solidity, 0, 1)
        + 0.15 * np.clip(extent, 0, 1)
        + 0.15 * (1 - np.clip(tex, 0, 1))
    )

    # Net/debris: very elongated or filament-like, lower solidity.
    scores["Entangled Net / Debris"] = (
        0.38 * np.clip((aspect - 3.0) / 8.0, 0, 1)
        + 0.28 * (1 - np.clip(solidity, 0, 1))
        + 0.18 * (1 - np.clip(extent, 0, 1))
        + 0.16 * np.clip(tex, 0, 1)
    )

    # Marine debris: medium irregular object with moderate texture.
    scores["Marine Debris"] = (
        0.30 * np.clip(area_ratio / 0.008, 0, 1)
        + 0.25 * np.clip(tex, 0, 1)
        + 0.20 * (1 - np.clip(circ, 0, 1))
        + 0.15 * np.clip(shadow, 0, 1)
        + 0.10 * (1 - np.clip(solidity, 0, 1))
    )

    best = max(scores, key=scores.get)
    raw = scores[best]
    # Separation from second best makes the confidence less arbitrary.
    ordered = sorted(scores.values(), reverse=True)
    gap = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    confidence = 0.55 * raw + 0.45 * np.clip(gap * 2.5, 0, 1)

    # Large clean elongated shapes should not be mislabeled as net.
    if best == "Entangled Net / Debris" and aspect > 5.0 and solidity > 0.88:
        best = "Pipe / Cylinder"
        confidence = max(confidence, 0.64)
    if best == "Shipwreck" and area_ratio < 0.0012:
        best = "Marine Debris"
        confidence *= 0.85

    if confidence < 0.40:
        best = "Unknown Anomaly"

    return best, float(np.clip(confidence, 0, 1)), {
        "brightness": float(np.clip(mean / 255, 0, 1)),
        "texture": tex,
        "shadow": shadow,
        "solidity": solidity,
        "aspect": aspect,
        "extent": extent,
        "score_margin": float(gap),
    }


def generate_demo_scene(seed=26):
    rng = np.random.default_rng(seed)
    H, W = 760, 1180
    base = rng.normal(27, 7, (H, W)).astype(np.float32)
    # Broad seabed gradients.
    yy, xx = np.mgrid[0:H, 0:W]
    base += 5 * np.sin(xx / 105) + 3 * np.sin(yy / 87)
    base = np.clip(base, 0, 255).astype(np.uint8)

    # Add sonar-like streaks and low-level speckle.
    for _ in range(120):
        x = int(rng.integers(0, W))
        y = int(rng.integers(0, H))
        length = int(rng.integers(20, 130))
        cv2.line(base, (x, y), (min(W-1, x+length), y), int(rng.integers(35, 80)), 1)

    truth = []

    def add_ellipse(cx, cy, rx, ry, label, angle=0):
        cv2.ellipse(base, (cx, cy), (rx, ry), angle, 0, 360, 205, -1)
        # shadow to lower/right
        sx = min(W-1, cx + int(rx*1.2))
        sy = min(H-1, cy + int(ry*1.0))
        cv2.ellipse(base, (sx, sy), (max(10, rx//2), max(7, ry//2)), angle, 0, 360, 8, -1)
        truth.append((label, cx, cy, rx, ry))

    # Shipwreck silhouette: large, irregular, high-return structure.
    pts = np.array([[130,170],[230,145],[305,175],[340,245],[300,285],[185,272],[112,230]], np.int32)
    cv2.fillPoly(base, [pts], 215)
    cv2.polylines(base, [pts], True, 245, 5)
    cv2.line(base, (150,215), (300,215), 245, 7)
    cv2.line(base, (195,170), (195,260), 245, 4)
    cv2.ellipse(base, (275, 240), (36, 16), -15, 0, 360, 245, -1)
    cv2.ellipse(base, (340, 300), (55, 24), -15, 0, 360, 6, -1)
    truth.append(("Shipwreck", 230, 215, 120, 75))

    # Pipe / cylinder.
    cv2.roundedRectangle = getattr(cv2, "roundedRectangle", None)  # harmless compatibility no-op
    cv2.rectangle(base, (500, 135), (790, 188), 205, -1)
    cv2.ellipse(base, (790, 161), (28, 27), 0, 0, 360, 230, -1)
    cv2.ellipse(base, (500, 161), (22, 27), 0, 0, 360, 160, -1)
    cv2.rectangle(base, (525, 190), (775, 220), 7, -1)
    truth.append(("Pipe / Cylinder", 645, 176, 160, 52))

    # Entangled net / debris: filament cluster.
    net_pts = [
        [(120,470),(210,430),(285,480),(365,440),(445,495)],
        [(155,420),(205,520),(275,420),(340,520),(420,430)],
        [(100,500),(180,455),(260,510),(340,455),(465,520)],
    ]
    for line in net_pts:
        cv2.polylines(base, [np.array(line,np.int32)], False, 215, 3)
    truth.append(("Entangled Net / Debris", 285, 475, 190, 65))

    # Marine debris cluster.
    for cx, cy, rx, ry in [(780,430,38,25),(845,460,24,38),(905,420,32,20),(875,505,45,18)]:
        add_ellipse(cx, cy, rx, ry, "Marine Debris", int(rng.integers(-25,25)))
    # Natural rock (not target).
    cv2.ellipse(base, (610, 575), (75, 45), 12, 0, 360, 115, -1)
    truth.append(("Natural Seafloor", 610, 575, 85, 55))

    # Additional small debris pieces so the dashboard feels populated.
    for cx, cy in [(990,250),(1025,285),(1080,225),(760,600),(950,620),(1060,560)]:
        cv2.ellipse(base, (cx,cy), (rng.integers(10,18), rng.integers(8,15)), 0, 0, 360, 150, -1)
        truth.append(("Marine Debris", cx, cy, 16, 12))

    # Add controlled speckle.
    noise = rng.normal(0, 9, (H,W)).astype(np.float32)
    base = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (3,3), 0)
    rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    return rgb, truth


def demo_detections(img_shape, truth):
    H, W = img_shape[:2]
    detections = []
    rng = np.random.default_rng(7)
    for i, (label, cx, cy, rx, ry) in enumerate(truth, start=1):
        x = max(0, int(cx-rx)); y=max(0,int(cy-ry));
        w = min(W-x, int(rx*2)); h=min(H-y,int(ry*2))
        if label == "Shipwreck": conf = 0.96
        elif label == "Pipe / Cylinder": conf = 0.94
        elif label == "Entangled Net / Debris": conf = 0.91
        elif label == "Marine Debris": conf = float(rng.uniform(0.78,0.92))
        else: conf = 0.73
        detections.append({
            "id": f"T-{i:02d}", "class": label, "confidence": conf,
            "x":x,"y":y,"w":w,"h":h,
            "area": int(max(1,w*h*0.55)),
            "priority": PRIORITY.get(label,"REVIEW"),
            "brightness": float(rng.uniform(.45,.82)),
            "texture": float(rng.uniform(.45,.9)),
            "shadow": float(rng.uniform(.25,.9)),
            "solidity": float(rng.uniform(.45,.95)),
            "aspect": float(max(w,h)/max(1,min(w,h))),
            "extent": float(rng.uniform(.35,.85)),
            "mode": "demo scene",
        })
    return detections


def normalize_model_class(name):
    s = str(name).lower().strip()
    if any(k in s for k in ["wreck", "shipwreck", "ship"]):
        return "Shipwreck"
    if any(k in s for k in ["pipe", "cylinder", "pipeline"]):
        return "Pipe / Cylinder"
    if any(k in s for k in ["net", "rope", "entangled", "fishing", "ghost"]):
        return "Entangled Net / Debris"
    if any(k in s for k in ["debris", "trash", "litter", "plastic"]):
        return "Marine Debris"
    return "Unknown Anomaly"


def run_yolo(rgb, model_path, conf_threshold=0.35):
    try:
        from ultralytics import YOLO
    except Exception:
        return None, "Ultralytics is not installed."
    try:
        model = YOLO(model_path)
        result = model.predict(source=rgb, conf=float(conf_threshold), verbose=False)[0]
        detections = []
        names = result.names if hasattr(result, "names") else {}
        if result.boxes is None:
            return [], None
        for i, box in enumerate(result.boxes):
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            x1,y1,x2,y2 = xyxy
            score = float(box.conf[0].cpu().item())
            cls_id = int(box.cls[0].cpu().item())
            raw_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            label = normalize_model_class(raw_name)
            detections.append({
                "id": f"T-{i+1:02d}", "class": label, "confidence": score,
                "x":x1,"y":y1,"w":max(1,x2-x1),"h":max(1,y2-y1),
                "area":max(1,(x2-x1)*(y2-y1)), "priority":PRIORITY.get(label,"REVIEW"),
                "brightness":0.0,"texture":0.0,"shadow":0.0,"solidity":0.0,
                "aspect":max(1,x2-x1)/max(1,y2-y1),"extent":0.0,
                "mode":"YOLO model", "raw_class":raw_name,
            })
        return detections, None
    except Exception as e:
        return None, f"Model error: {e}"


def run_fallback_detector(rgb, min_area, sensitivity):
    gray, enhanced, high = preprocess_sonar(rgb)
    candidates, mask = contour_candidates(enhanced, min_area=min_area, sensitivity=sensitivity)
    detections=[]
    H,W=gray.shape
    for i,d in enumerate(candidates, start=1):
        label, conf, feats = classify_heuristic(d, gray, enhanced)
        detections.append({
            "id":f"T-{i:02d}", "class":label, "confidence":conf,
            "x":d["x"],"y":d["y"],"w":d["w"],"h":d["h"],"area":int(d["area"]),
            "priority":PRIORITY.get(label,"REVIEW"), **feats, "mode":"vision heuristic",
        })
    detections = sorted(detections, key=lambda z:z["confidence"], reverse=True)
    return detections, enhanced, mask


def annotate_image(rgb, detections, selected_id=None):
    out = rgb.copy()
    for d in detections:
        x,y,w,h = d["x"],d["y"],d["w"],d["h"]
        color = hex_to_rgb(COLORS.get(d["class"], "#ffffff"))
        selected = d["id"] == selected_id
        thickness = 4 if selected else 2
        cv2.rectangle(out, (x,y), (x+w,y+h), color, thickness)
        label = f"{d['id']}  {d['class']}  {d['confidence']*100:.0f}%"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .48, 1)
        ly=max(0,y-th-10)
        cv2.rectangle(out,(x,ly),(min(out.shape[1]-1,x+tw+12),y),color,-1)
        cv2.putText(out,label,(x+6,max(12,y-7)),cv2.FONT_HERSHEY_SIMPLEX,.48,(5,15,20),1,cv2.LINE_AA)
        cx,cy=x+w//2,y+h//2
        cv2.circle(out,(cx,cy),5,color,-1)
    return out


def make_heatmap(rgb):
    gray = image_to_gray(rgb)
    # Edge/contrast response used as a visual explanation layer.
    blur = cv2.GaussianBlur(gray,(0,0),4)
    response = cv2.convertScaleAbs(cv2.absdiff(gray, blur), alpha=4.0)
    response = cv2.GaussianBlur(response,(0,0),2)
    colored = cv2.applyColorMap(response, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored, response


def mission_map(detections, shape, selected_id=None):
    H,W = shape[:2]
    canvas = np.full((650, 1050, 3), (7, 20, 28), dtype=np.uint8)
    # grid
    for gx in range(70, 1000, 100):
        cv2.line(canvas,(gx,55),(gx,600),(20,50,60),1)
    for gy in range(55, 601, 90):
        cv2.line(canvas,(55,gy),(1000,gy),(20,50,60),1)
    cv2.putText(canvas,"RELATIVE SURVEY FRAME",(55,32),cv2.FONT_HERSHEY_SIMPLEX,.55,(110,150,160),1,cv2.LINE_AA)
    cv2.putText(canvas,"PORT",(25,340),cv2.FONT_HERSHEY_SIMPLEX,.5,(85,120,130),1,cv2.LINE_AA)
    cv2.putText(canvas,"STARBOARD",(935,340),cv2.FONT_HERSHEY_SIMPLEX,.5,(85,120,130),1,cv2.LINE_AA)
    sx, sy = 55, 55
    ew, eh = 945, 545
    for d in detections:
        cx = sx + int((d["x"]+d["w"]/2)/max(1,W)*ew)
        cy = sy + int((d["y"]+d["h"]/2)/max(1,H)*eh)
        col=hex_to_rgb(COLORS.get(d["class"],"#ffffff"))
        r=10 if d["id"]==selected_id else 7
        cv2.circle(canvas,(cx,cy),r,col,-1)
        if d["id"]==selected_id:
            cv2.circle(canvas,(cx,cy),r+8,col,2)
        cv2.putText(canvas,d["id"],(cx+10,cy+4),cv2.FONT_HERSHEY_SIMPLEX,.42,(205,225,230),1,cv2.LINE_AA)
    # Scale bar
    cv2.line(canvas,(820,615),(950,615),(180,210,215),2)
    cv2.putText(canvas,"relative distance",(820,605),cv2.FONT_HERSHEY_SIMPLEX,.4,(100,130,140),1,cv2.LINE_AA)
    return canvas


def detection_dataframe(detections, gps_df=None):
    rows=[]
    gps_lookup={}
    if gps_df is not None and not gps_df.empty:
        cols={c.lower():c for c in gps_df.columns}
        latc=cols.get("latitude") or cols.get("lat")
        lonc=cols.get("longitude") or cols.get("lon") or cols.get("lng")
        idc=cols.get("target_id") or cols.get("id")
        if latc and lonc:
            for idx,r in gps_df.iterrows():
                key=str(r[idc]) if idc else f"T-{idx+1:02d}"
                gps_lookup[key]=(safe_float(r[latc]),safe_float(r[lonc]))
    for d in detections:
        lat,lon=None,None
        if d["id"] in gps_lookup:
            lat,lon=gps_lookup[d["id"]]
        rows.append({
            "Target":d["id"], "Classification":d["class"], "Confidence %":round(d["confidence"]*100,1),
            "Priority":d["priority"], "X":d["x"], "Y":d["y"], "Width":d["w"], "Height":d["h"],
            "Area px":d["area"], "Latitude":lat, "Longitude":lon,
        })
    return pd.DataFrame(rows)


def render_confidence_chart(detections):
    if not PLOTLY_OK or not detections:
        return None
    ds=sorted(detections,key=lambda d:d["confidence"],reverse=True)
    labels=[d["id"] for d in ds]
    vals=[round(d["confidence"]*100,1) for d in ds]
    colors=[COLORS.get(d["class"],"#b8c5ca") for d in ds]
    fig=go.Figure(go.Bar(x=vals,y=labels,orientation="h",marker_color=colors,hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_layout(height=max(260,min(520,30*len(ds)+100)),margin=dict(l=10,r=10,t=10,b=10),
                      paper_bgcolor="#091a23",plot_bgcolor="#091a23",font=dict(color="#d9edf1"),
                      xaxis=dict(range=[0,100],gridcolor="#173744",title="Confidence"),
                      yaxis=dict(autorange="reversed",gridcolor="#173744"),showlegend=False)
    return fig


def render_class_chart(detections):
    if not PLOTLY_OK:
        return None
    counts={c:0 for c in CLASSES}
    for d in detections: counts[d["class"]]=counts.get(d["class"],0)+1
    labels=[k for k,v in counts.items() if v]
    vals=[counts[k] for k in labels]
    colors=[COLORS[k] for k in labels]
    fig=go.Figure(go.Bar(x=labels,y=vals,marker_color=colors,hovertemplate="%{x}: %{y}<extra></extra>"))
    fig.update_layout(height=300,margin=dict(l=10,r=10,t=15,b=10),paper_bgcolor="#091a23",plot_bgcolor="#091a23",
                      font=dict(color="#d9edf1"),xaxis=dict(gridcolor="#173744"),yaxis=dict(gridcolor="#173744"),showlegend=False)
    return fig


def render_feature_radar(d):
    if not PLOTLY_OK:
        return None
    categories=["Brightness","Texture","Shadow","Solidity","Shape evidence"]
    vals=[
        d.get("brightness",0.5), d.get("texture",0.5), d.get("shadow",0.5),
        d.get("solidity",0.5), np.clip((d.get("score_margin",0.15)*3),0,1)
    ]
    vals += [vals[0]]
    cats=categories+[categories[0]]
    fig=go.Figure(go.Scatterpolar(r=vals,theta=cats,fill="toself",line=dict(color=COLORS.get(d["class"],"#36d7e8"))))
    fig.update_layout(height=330,margin=dict(l=20,r=20,t=20,b=20),paper_bgcolor="#091a23",font=dict(color="#d9edf1"),
                      polar=dict(bgcolor="#091a23",radialaxis=dict(range=[0,1],gridcolor="#23404b",tickfont=dict(color="#6f8c98")),angularaxis=dict(gridcolor="#23404b")),showlegend=False)
    return fig

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
for key, default in {
    "image": None,
    "source_name": "",
    "detections": [],
    "mode": "Demo",
    "selected_id": None,
    "analysis_ms": 0,
    "enhanced": None,
    "mask": None,
    "gps_df": None,
    "model_path": None,
}.items():
    if key not in st.session_state:
        st.session_state[key]=default

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("<div style='font-family:Space Grotesk;font-size:21px;font-weight:800;'>SONAR <span style='color:#36d7e8;'>VISION</span></div>", unsafe_allow_html=True)
    st.caption("Acoustic detection workspace")
    st.divider()

    mode=st.radio("INPUT", ["Demo", "Upload sonar"], index=0 if st.session_state.mode=="Demo" else 1)
    st.session_state.mode=mode

    st.markdown("<div class='smallcap'>DETECTION ENGINE</div>", unsafe_allow_html=True)
    sensitivity=st.slider("Sensitivity",0.50,0.95,0.72,0.01)
    min_area=st.slider("Minimum object area",40,1500,90,10)
    model_file=st.file_uploader("Optional trained model (.pt)", type=["pt"], help="If supplied and ultralytics is installed, YOLO inference is used instead of the heuristic fallback.")
    gps_file=st.file_uploader("Optional GPS / geotag CSV", type=["csv"], help="Columns: target_id, latitude, longitude. If target_id is absent, rows are matched T-01, T-02, ...")

    st.markdown("<div class='smallcap' style='margin-top:18px;'>TARGET LEGEND</div>", unsafe_allow_html=True)
    for c in CLASSES:
        st.markdown(f"<div style='display:flex;align-items:center;gap:8px;margin:8px 0;font-size:11px;'><span style='width:9px;height:9px;border-radius:50%;background:{COLORS[c]};display:inline-block;'></span>{c}</div>", unsafe_allow_html=True)

    st.divider()
    st.caption("The app separates detection, classification, confidence and localization so each finding can be inspected before export.")

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown(
    """
<div class='hero'>
  <div>
    <div class='brand-kicker'>UNDERWATER ACOUSTIC INTELLIGENCE</div>
    <div class='brand-title'>Sonar Vision</div>
    <div class='brand-sub'>Detect objects in side-scan sonar, classify likely targets, inspect confidence evidence, and localize findings on a survey-frame map.</div>
    <div class='status-pill'><span class='status-dot'></span> ANALYSIS CONSOLE READY</div>
  </div>
  <div style='text-align:right;padding-top:8px;'>
    <div class='smallcap'>MISSION MODE</div>
    <div style='font-family:Space Grotesk;font-size:20px;font-weight:700;margin-top:5px;'>LIVE ANALYSIS</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Input area
# ------------------------------------------------------------
left,right=st.columns([1.8,1],gap="large")
with left:
    st.markdown("<div class='section-title'>Mission image</div><div class='section-sub'>Upload a sonar frame or use the built-in synthetic survey scene for a complete demonstration.</div>",unsafe_allow_html=True)
    uploaded=None
    if mode=="Upload sonar":
        uploaded=st.file_uploader("Sonar image",type=["png","jpg","jpeg","bmp","tif","tiff"],label_visibility="collapsed")
    else:
        st.info("Demo mode is loaded with a synthetic sonar scene containing multiple target types. Switch to Upload sonar when you have real survey data.")
with right:
    st.markdown("<div class='section-title'>Run analysis</div><div class='section-sub'>One click refreshes the detection layer and all analytics.</div>",unsafe_allow_html=True)
    analyze=st.button("RUN DETECTION",use_container_width=True)
    reset=st.button("RESET WORKSPACE",use_container_width=True)

if reset:
    for k in ["image","source_name","detections","selected_id","analysis_ms","enhanced","mask","gps_df"]:
        st.session_state[k] = None if k in ["image","enhanced","mask","gps_df"] else ([] if k=="detections" else ("" if k=="source_name" else None))
    st.rerun()

# Load GPS immediately if supplied
if gps_file is not None:
    try:
        st.session_state.gps_df=pd.read_csv(gps_file)
    except Exception as e:
        st.warning(f"Could not read GPS CSV: {e}")

# Input selection
if uploaded is not None:
    try:
        im=Image.open(uploaded).convert("RGB")
        arr=np.array(im)
        st.session_state.image=arr
        st.session_state.source_name=uploaded.name
    except Exception as e:
        st.error(f"Could not read image: {e}")
elif mode=="Demo" and st.session_state.image is None:
    demo_img,truth=generate_demo_scene()
    st.session_state.image=demo_img
    st.session_state.source_name="Synthetic survey scene"
    st.session_state.demo_truth=truth

# Automatically show image even before running analysis
if st.session_state.image is not None:
    preview_col, info_col=st.columns([2.2,1],gap="large")
    with preview_col:
        st.image(st.session_state.image,caption=st.session_state.source_name,use_container_width=True)
    with info_col:
        st.markdown("<div class='panel'>",unsafe_allow_html=True)
        st.markdown("<div class='smallcap'>SOURCE</div>",unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:14px;font-weight:700;margin:5px 0 14px;'>{st.session_state.source_name}</div>",unsafe_allow_html=True)
        h,w=st.session_state.image.shape[:2]
        st.write(f"**Frame:** {w:,} × {h:,} px")
        st.write(f"**Sensitivity:** {sensitivity:.2f}")
        st.write(f"**Min area:** {min_area:,} px")
        engine="Heuristic vision" if model_file is None else "Optional YOLO model"
        st.write(f"**Engine:** {engine}")
        st.markdown("</div>",unsafe_allow_html=True)

# ------------------------------------------------------------
# Run analysis
# ------------------------------------------------------------
if analyze and st.session_state.image is not None:
    start=time.perf_counter()
    rgb=st.session_state.image
    detections=[]
    enhanced=None
    mask=None

    # Demo uses curated ground truth so the demo visibly demonstrates the intended taxonomy.
    if mode=="Demo":
        truth=st.session_state.get("demo_truth")
        if truth is None:
            _,truth=generate_demo_scene()
        detections=demo_detections(rgb.shape,truth)
        _, enhanced, mask=run_fallback_detector(rgb,min_area=min_area,sensitivity=sensitivity)
        engine_name="demo scene"
    else:
        model_path=None
        if model_file is not None:
            tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".pt")
            tmp.write(model_file.getvalue()); tmp.close(); model_path=tmp.name
        if model_path:
            yolo_dets, err=run_yolo(rgb,model_path,conf_threshold=max(.20,1-sensitivity))
            if yolo_dets is not None:
                detections=yolo_dets; engine_name="YOLO model"
            else:
                st.warning(err + " Falling back to the built-in vision detector.")
                detections,enhanced,mask=run_fallback_detector(rgb,min_area,sensitivity); engine_name="vision heuristic"
        else:
            detections,enhanced,mask=run_fallback_detector(rgb,min_area,sensitivity); engine_name="vision heuristic"

    # Sort and cap very noisy scenes.
    detections=sorted(detections,key=lambda d:d["confidence"],reverse=True)[:60]
    for i,d in enumerate(detections,1):
        d["id"]=f"T-{i:02d}"
    st.session_state.detections=detections
    st.session_state.enhanced=enhanced
    st.session_state.mask=mask
    st.session_state.analysis_ms=(time.perf_counter()-start)*1000
    st.session_state.engine_name=engine_name
    if detections:
        st.session_state.selected_id=detections[0]["id"]
    else:
        st.session_state.selected_id=None

# ------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------
detections=st.session_state.detections
if detections:
    counts={c:0 for c in CLASSES}
    for d in detections: counts[d["class"]]=counts.get(d["class"],0)+1
    high=sum(1 for d in detections if d["priority"]=="HIGH")
    avg_conf=np.mean([d["confidence"] for d in detections])*100
    reviewed=sum(1 for d in detections if d["class"]=="Unknown Anomaly")

    st.markdown("<div class='section-title'>Mission overview</div>",unsafe_allow_html=True)
    cols=st.columns(6)
    metrics=[
        ("Targets",len(detections),"total findings"),
        ("Shipwreck",counts.get("Shipwreck",0),"identified"),
        ("Pipe / Cylinder",counts.get("Pipe / Cylinder",0),"identified"),
        ("Net / Debris",counts.get("Entangled Net / Debris",0),"identified"),
        ("Marine Debris",counts.get("Marine Debris",0),"identified"),
        ("Review",reviewed,f"{high} high priority"),
    ]
    for c,(lab,val,sub) in zip(cols,metrics):
        with c:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>{lab}</div><div class='metric-value'>{val}</div><div class='metric-delta'>{sub}</div></div>",unsafe_allow_html=True)

    st.markdown(f"<div style='color:#5f7d88;font-size:11px;margin:12px 0 4px;'>Engine: <b style='color:#a9c2ca'>{st.session_state.get('engine_name','analysis')}</b> &nbsp; • &nbsp; Average confidence: <b style='color:#a9c2ca'>{avg_conf:.1f}%</b> &nbsp; • &nbsp; Processing: <b style='color:#a9c2ca'>{st.session_state.analysis_ms:.0f} ms</b></div>",unsafe_allow_html=True)

    # Main workspace tabs
    tabs=st.tabs(["DETECTION VIEW","ANALYTICS","TARGET INSPECTOR","LOCALIZATION","REPORTS"])

    # --------------------------------------------------------
    # Detection View
    # --------------------------------------------------------
    with tabs[0]:
        annotated=annotate_image(st.session_state.image,detections,st.session_state.selected_id)
        c1,c2=st.columns([2.1,1],gap="large")
        with c1:
            st.image(annotated,caption="Detected objects — click a target in the inspector to highlight it.",use_container_width=True)
        with c2:
            st.markdown("<div class='panel'>",unsafe_allow_html=True)
            st.markdown("<div class='smallcap'>FINDINGS</div>",unsafe_allow_html=True)
            options=[d["id"] for d in detections]
            current=st.session_state.selected_id if st.session_state.selected_id in options else options[0]
            selected=st.selectbox("Select target",options,index=options.index(current),label_visibility="collapsed")
            st.session_state.selected_id=selected
            d=next(x for x in detections if x["id"]==selected)
            col=COLORS.get(d["class"],"#b8c5ca")
            st.markdown(f"<div style='margin-top:10px;font-size:21px;font-weight:800;color:{col};'>{d['class']}</div>",unsafe_allow_html=True)
            st.markdown(f"<div style='color:#7896a1;font-size:11px;margin:3px 0 16px;'>{d['id']} · {d['priority']} priority</div>",unsafe_allow_html=True)
            st.progress(float(d["confidence"]),text=f"Confidence {d['confidence']*100:.1f}%")
            st.markdown("<div style='height:6px'></div>",unsafe_allow_html=True)
            st.write(f"**Bounding box:** {d['w']} × {d['h']} px")
            st.write(f"**Position:** X {d['x']} · Y {d['y']}")
            st.write(f"**Area:** {d['area']:,} px²")
            st.markdown("</div>",unsafe_allow_html=True)

        st.markdown("<div class='legend'>"+"".join([f"<div class='legend-item'><span class='legend-dot' style='background:{COLORS[c]}'></span>{c}</div>" for c in CLASSES])+"</div>",unsafe_allow_html=True)

    # --------------------------------------------------------
    # Analytics
    # --------------------------------------------------------
    with tabs[1]:
        a,b=st.columns(2,gap="large")
        with a:
            st.markdown("<div class='section-title'>Confidence distribution</div><div class='section-sub'>Every target is scored independently.</div>",unsafe_allow_html=True)
            fig=render_confidence_chart(detections)
            if fig is not None: st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
            else: st.bar_chart(pd.DataFrame({"confidence":[d["confidence"]*100 for d in detections]}))
        with b:
            st.markdown("<div class='section-title'>Classification mix</div><div class='section-sub'>Distribution of detected target categories.</div>",unsafe_allow_html=True)
            fig=render_class_chart(detections)
            if fig is not None: st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
            else:
                st.bar_chart(pd.Series([d["class"] for d in detections]).value_counts())

        st.markdown("<div class='section-title'>Signal evidence</div><div class='section-sub'>A transparent view of the visual features used by the prototype fallback detector.</div>",unsafe_allow_html=True)
        rows=[]
        for d in detections:
            rows.append({"Target":d["id"],"Class":d["class"],"Confidence %":round(d["confidence"]*100,1),"Brightness":round(d.get("brightness",0)*100),"Texture":round(d.get("texture",0)*100),"Shadow":round(d.get("shadow",0)*100),"Solidity":round(d.get("solidity",0)*100)})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    # --------------------------------------------------------
    # Inspector
    # --------------------------------------------------------
    with tabs[2]:
        options=[d["id"] for d in detections]
        selected=st.selectbox("Target",options,index=options.index(st.session_state.selected_id) if st.session_state.selected_id in options else 0)
        st.session_state.selected_id=selected
        d=next(x for x in detections if x["id"]==selected)
        c1,c2=st.columns([1.2,1],gap="large")
        with c1:
            x,y,w,h=d["x"],d["y"],d["w"],d["h"]
            crop=st.session_state.image[max(0,y):min(st.session_state.image.shape[0],y+h),max(0,x):min(st.session_state.image.shape[1],x+w)]
            st.image(crop,caption=f"Crop — {d['id']}",use_container_width=True)
        with c2:
            st.markdown(f"<div class='panel'><div class='smallcap'>TARGET PROFILE</div><div style='font-family:Space Grotesk;font-size:26px;font-weight:700;margin-top:6px;'>{d['class']}</div><div style='color:#7e9aa4;font-size:12px;margin:4px 0 18px;'>{d['id']} · {d['priority']} priority</div>",unsafe_allow_html=True)
            st.progress(float(d["confidence"]),text=f"Prediction confidence · {d['confidence']*100:.1f}%")
            st.write(f"**Source mode:** {d.get('mode','analysis')}")
            st.write(f"**Box:** ({x}, {y}) → ({x+w}, {y+h})")
            st.write(f"**Dimensions:** {w} × {h} px")
            st.write(f"**Aspect ratio:** {d.get('aspect',w/max(1,h)):.2f}")
            st.markdown("</div>",unsafe_allow_html=True)
        if PLOTLY_OK:
            radar=render_feature_radar(d)
            if radar is not None:
                st.plotly_chart(radar,use_container_width=True,config={"displayModeBar":False})
        else:
            feats=pd.DataFrame({"Evidence":["Brightness","Texture","Shadow","Solidity"],"Score":[d.get("brightness",0),d.get("texture",0),d.get("shadow",0),d.get("solidity",0)]})
            st.bar_chart(feats.set_index("Evidence"))

    # --------------------------------------------------------
    # Localization
    # --------------------------------------------------------
    with tabs[3]:
        st.markdown("<div class='section-title'>Target localization</div><div class='section-sub'>Relative position is always available. Upload a GPS CSV to convert detections into real survey coordinates.</div>",unsafe_allow_html=True)
        map_img=mission_map(detections,st.session_state.image.shape,st.session_state.selected_id)
        st.image(map_img,use_container_width=True)

        df=detection_dataframe(detections,st.session_state.gps_df)
        if st.session_state.gps_df is not None and df["Latitude"].notna().any():
            st.markdown("<div class='section-title'>Geotagged findings</div>",unsafe_allow_html=True)
            geo=df.dropna(subset=["Latitude","Longitude"])[["Target","Classification","Confidence %","Latitude","Longitude"]]
            st.dataframe(geo,use_container_width=True,hide_index=True)
            try:
                st.map(geo.rename(columns={"Latitude":"latitude","Longitude":"longitude"})[["latitude","longitude"]],zoom=10)
            except Exception:
                pass
        else:
            st.info("No GPS metadata loaded. The relative survey-frame map above is ready now; add a CSV with target_id, latitude and longitude when you have navigation data.")

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------
    with tabs[4]:
        df=detection_dataframe(detections,st.session_state.gps_df)
        annotated=annotate_image(st.session_state.image,detections)
        img_pil=Image.fromarray(annotated)
        img_buf=io.BytesIO(); img_pil.save(img_buf,format="PNG")
        report={
            "source":st.session_state.source_name,
            "engine":st.session_state.get("engine_name","analysis"),
            "processed_ms":round(st.session_state.analysis_ms,1),
            "frame_width":int(st.session_state.image.shape[1]),
            "frame_height":int(st.session_state.image.shape[0]),
            "detections":detections,
        }
        r1,r2,r3=st.columns(3)
        with r1: st.download_button("DOWNLOAD CSV",df.to_csv(index=False).encode(),"sonar_findings.csv","text/csv",use_container_width=True)
        with r2: st.download_button("DOWNLOAD JSON",json.dumps(report,indent=2).encode(),"sonar_report.json","application/json",use_container_width=True)
        with r3: st.download_button("DOWNLOAD ANNOTATED PNG",img_buf.getvalue(),"sonar_annotated.png","image/png",use_container_width=True)
        st.markdown("<div class='section-title'>Report preview</div>",unsafe_allow_html=True)
        st.dataframe(df,use_container_width=True,hide_index=True)

else:
    st.markdown("<div class='panel' style='margin-top:24px;text-align:center;padding:45px 20px;'><div style='font-family:Space Grotesk;font-size:22px;font-weight:700;'>Ready for analysis</div><div style='color:#7896a1;font-size:12px;margin-top:8px;'>Load the demo scene or upload a side-scan sonar image, then press RUN DETECTION.</div></div>",unsafe_allow_html=True)

st.markdown("<div class='footer'>SONAR VISION · Detection → Classification → Confidence → Localization → Reporting</div>",unsafe_allow_html=True)
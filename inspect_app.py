import base64
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent
IMG_SIZE = 224
MAX_SIDE = 1400
MAX_PARTS = 20
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEVICE = torch.device("cpu")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


def selective_scan(u, dt, A, B, C, D):
    batch, length, dim = u.shape
    hidden = u.new_zeros(batch, dim, A.shape[1])
    outputs = []
    for t in range(length):
        dt_t = dt[:, t].unsqueeze(-1)
        hidden = torch.exp(dt_t * A) * hidden + (dt_t * B[:, t].unsqueeze(1)) * u[:, t].unsqueeze(-1)
        outputs.append((hidden * C[:, t].unsqueeze(1)).sum(-1) + D * u[:, t])
    return torch.stack(outputs, dim=1)


class MambaVisionMixer(nn.Module):
    def __init__(self, dim, d_state=8):
        super().__init__()
        self.half = dim // 2
        self.d_state = d_state
        self.dt_rank = max(1, math.ceil(self.half / 16))
        self.in_proj = nn.Linear(dim, dim, bias=False)
        self.conv_x = nn.Conv1d(self.half, self.half, 3, padding=1, groups=self.half)
        self.conv_z = nn.Conv1d(self.half, self.half, 3, padding=1, groups=self.half)
        self.x_proj = nn.Linear(self.half, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.half)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.half, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.half))
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        x_branch, z = self.in_proj(x).split(self.half, dim=-1)
        x_branch = F.silu(self.conv_x(x_branch.transpose(1, 2)).transpose(1, 2))
        z = F.silu(self.conv_z(z.transpose(1, 2)).transpose(1, 2))
        dt, B, C = torch.split(self.x_proj(x_branch), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        y = selective_scan(x_branch, F.softplus(self.dt_proj(dt)), -torch.exp(self.A_log.float()), B, C, self.D.float())
        return self.out_proj(torch.cat([y, z], dim=-1))


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        return self.net(x)


class MambaBlock(nn.Module):
    def __init__(self, dim, d_state):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = MambaVisionMixer(dim, d_state)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.mixer(self.norm1(tokens))
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normed = self.norm1(tokens)
        attn, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + attn
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)), inplace=False)
        return F.relu(x + self.bn2(self.conv2(h)), inplace=False)


class MambaClassifier(nn.Module):
    def __init__(self, num_classes, in_dim, dim, res_depths, hybrid_depths, hybrid_mamba, hybrid_heads, d_state, pool_size=4):
        super().__init__()
        self.pool_size = pool_size
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_dim), nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
        )
        channels = [dim, dim * 2, dim * 4, dim * 8]
        self.stage1 = nn.Sequential(*[ResBlock(channels[0]) for _ in range(res_depths[0])])
        self.down2 = nn.Sequential(nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(channels[1]), nn.ReLU(inplace=True))
        self.stage2 = nn.Sequential(*[ResBlock(channels[1]) for _ in range(res_depths[1])])
        self.down3 = nn.Sequential(nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(channels[2]), nn.ReLU(inplace=True))
        self.stage3 = self._hybrid(channels[2], hybrid_depths[0], hybrid_mamba[0], hybrid_heads[0], d_state)
        self.down4 = nn.Sequential(nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(channels[3]), nn.ReLU(inplace=True))
        self.stage4 = self._hybrid(channels[3], hybrid_depths[1], hybrid_mamba[1], hybrid_heads[1], d_state)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(channels[3], num_classes)

    @staticmethod
    def _hybrid(dim, depth, n_mamba, heads, d_state):
        blocks = []
        for i in range(depth):
            blocks.append(MambaBlock(dim, d_state) if i < n_mamba else AttentionBlock(dim, heads))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stage1(self.stem(x))
        x = self.stage2(self.down2(x))
        x = F.adaptive_avg_pool2d(self.down3(x), (self.pool_size, self.pool_size))
        x = self.stage3(x)
        x = self.stage4(self.down4(x))
        return self.fc(self.drop(self.pool(x).flatten(1)))


def load_model(path):
    blob = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = blob.get("cfg") or dict(in_dim=24, dim=48, res_depths=(1, 2), hybrid_depths=(2, 1), hybrid_mamba=(1, 1), hybrid_heads=(4, 8), d_state=4)
    model = MambaClassifier(num_classes=len(blob["labels"]), **cfg).to(DEVICE)
    model.load_state_dict(blob["state"])
    model.eval()
    return model, list(blob["labels"])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - inter
    return inter / area if area else 0.0


def nms(boxes, scores, thresh=0.35):
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) < thresh]
    return keep


def resize_max(image, max_side=MAX_SIDE):
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 0.999:
        return image.copy(), 1.0
    out = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return out, scale


def component_proposals(bgr):
    small, scale = resize_max(bgr)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ksize = max(3, int(max(blur.shape) / 28) | 1)
    background = cv2.medianBlur(blur, ksize)
    diff = cv2.absdiff(blur, background)
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (thresh > 0).mean() > 0.45:
        thresh = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = morph.shape
    image_area = float(height * width)
    boxes, scores = [], []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 0.00002 * image_area or area > 0.35 * image_area or w < 6 or h < 6:
            continue
        aspect = w / float(h)
        if aspect < 0.18 or aspect > 7.5:
            continue
        if cv2.contourArea(contour) / float(area) < 0.32:
            continue
        boxes.append([x, y, x + w, y + h])
        scores.append(area)
    kept = [boxes[i] for i in nms(boxes, scores)] if boxes else []
    full = [[int(v / scale) for v in box] for box in kept]
    full.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
    steps = [
        ("Color", cv2.cvtColor(small, cv2.COLOR_BGR2RGB)),
        ("Grayscale", gray),
        ("Blur", blur),
        ("Background", background),
        ("Difference", diff),
        ("Threshold", thresh),
        ("Morphology", morph),
    ]
    return full, steps


def crop_box(bgr, box, pad=3):
    height, width = bgr.shape[:2]
    x1 = max(0, int(box[0]) - pad)
    y1 = max(0, int(box[1]) - pad)
    x2 = min(width, int(box[2]) + pad)
    y2 = min(height, int(box[3]) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = bgr[y1:y2, x1:x2]
    return crop if crop.size else None


def to_224(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def pretty(name):
    for prefix in ("component_", "defect_"):
        if name.startswith(prefix):
            return name[len(prefix):].replace("_", " ")
    return name.replace("_", " ")


@torch.no_grad()
def classify(model, labels, rgb):
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    tensor = ((tensor - mean) / std).unsqueeze(0)
    prob = torch.softmax(model(tensor), dim=1)[0]
    index = int(prob.argmax())
    return pretty(labels[index]), float(prob[index])


def jpg_uri(image):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def draw_labels(bgr, boxes, lines, color):
    small, scale = resize_max(bgr)
    canvas = small.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, canvas.shape[1] / 1600)
    thickness = 2
    for index, (box, line) in enumerate(zip(boxes, lines), start=1):
        x1, y1, x2, y2 = [int(round(v * scale)) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        text = f"{index}. {line}"
        (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
        top = y1 - th - base - 8
        if top < 0:
            top = min(canvas.shape[0] - th - base - 6, y2 + 4)
        left = max(0, min(x1, canvas.shape[1] - tw - 8))
        cv2.rectangle(canvas, (left, top), (left + tw + 8, top + th + base + 6), color, -1)
        cv2.putText(canvas, text, (left + 4, top + th + 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def inspect(bgr):
    boxes, steps = component_proposals(bgr)
    shown = boxes[:MAX_PARTS]
    if not shown:
        height, width = bgr.shape[:2]
        shown = [[0, 0, width, height]]
    rows = []
    crops = []
    for box in shown:
        crop = crop_box(bgr, box)
        if crop is None:
            continue
        rgb = to_224(crop)
        component, component_confidence = classify(COMP_MODEL, COMP_LABELS, rgb)
        defect, defect_confidence = classify(DEFECT_MODEL, DEFECT_LABELS, rgb)
        rows.append({
            "component": component,
            "component_confidence": f"{component_confidence:.2f}",
            "defect": defect,
            "defect_confidence": f"{defect_confidence:.2f}",
            "crop": jpg_uri(rgb),
        })
        crops.append(rgb)
    component_lines = [f"{row['component']} {row['component_confidence']}" for row in rows]
    defect_lines = [f"{row['defect']} {row['defect_confidence']}" for row in rows]
    used_boxes = shown[:len(rows)]
    return {
        "found": len(boxes),
        "shown": len(rows),
        "steps": [{"title": title, "src": jpg_uri(image)} for title, image in steps],
        "original": jpg_uri(cv2.cvtColor(resize_max(bgr)[0], cv2.COLOR_BGR2RGB)),
        "component_view": jpg_uri(draw_labels(bgr, used_boxes, component_lines, (210, 90, 20))),
        "defect_view": jpg_uri(draw_labels(bgr, used_boxes, defect_lines, (40, 40, 210))),
        "rows": rows,
    }


print("loading models", flush=True)
COMP_MODEL, COMP_LABELS = load_model(ROOT / "outputs" / "mamba_component.pt")
DEFECT_MODEL, DEFECT_LABELS = load_model(ROOT / "outputs" / "mamba_defect.pt")
print("models ready", len(COMP_LABELS), len(DEFECT_LABELS), flush=True)


@app.get("/")
def home():
    return render_template("index.html", result=None, error=None)


@app.post("/inspect")
def inspect_upload():
    if request.form.get("sample") == "1":
        path = ROOT / "sample_img" / "01_missing_hole_06.jpg"
        bgr = cv2.imread(str(path))
        if bgr is None:
            return render_template("index.html", result=None, error="Sample image is missing.")
        return render_template("index.html", result=inspect(bgr), error=None)
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return render_template("index.html", result=None, error="Choose an image first.")
    data = np.frombuffer(upload.read(), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return render_template("index.html", result=None, error="That file is not an image.")
    return render_template("index.html", result=inspect(bgr), error=None)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)

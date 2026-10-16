"""Semantic Segmentation live demo (Flask + PyTorch + OpenCV).
Run:  python app.py   ->  open http://127.0.0.1:5000
"""
import base64, io, os, threading, time, urllib.request, uuid
from collections import OrderedDict

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
import torchvision.models.segmentation as seg

BASE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(BASE, "samples")
MAX_SIDE = 640  # keeps CPU inference fast on a laptop
app = Flask(__name__, static_folder=os.path.join(BASE, "static"))
torch.set_num_threads(max(1, os.cpu_count() or 4))

MODELS = {
    "deeplab": ("DeepLabV3 + MobileNetV3 (fast, ~42 MB)",
                lambda: seg.deeplabv3_mobilenet_v3_large(weights=seg.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)),
    "fcn": ("FCN-ResNet50, Long et al. 2015 (slower, ~135 MB)",
            lambda: seg.fcn_resnet50(weights=seg.FCN_ResNet50_Weights.DEFAULT)),
}
NAMES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
         "cow", "dining table", "dog", "horse", "motorbike", "person", "potted plant", "sheep",
         "sofa", "train", "tv monitor"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_loaded, _lock = {}, threading.Lock()
CACHE = OrderedDict()  # token -> class-id map, used by the metrics tab


def voc_palette():
    p = np.zeros((256, 3), np.uint8)
    for i in range(256):
        c, r, g, b = i, 0, 0, 0
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j); g |= ((c >> 1) & 1) << (7 - j); b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        p[i] = (r, g, b)
    return p


PAL = voc_palette()


def get_model(key):
    with _lock:
        if key not in _loaded:
            m = MODELS[key][1]()
            m.eval()
            _loaded[key] = m
        return _loaded[key]


def b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def predict(img, key):
    model = get_model(key)
    x = torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1)
    x = ((x - MEAN) / STD)[None]
    with torch.no_grad():
        pr = model(x)["out"].softmax(1)[0]  # per-pixel "votes" for each class
    top = pr.topk(3, dim=0)
    probs = (top.values.numpy() * 255).round().astype(np.uint8).transpose(1, 2, 0)
    tops = top.indices.numpy().astype(np.uint8).transpose(1, 2, 0)
    return pr.argmax(0).numpy().astype(np.uint8), probs, tops


def colorize(ids, bg_alpha=0):
    rgba = np.zeros((*ids.shape, 4), np.uint8)
    rgba[..., :3] = PAL[ids]
    rgba[..., 3] = np.where(ids > 0, 255, bg_alpha)
    return rgba


def instances_from_semantic(ids):
    """Approximation: connected components (8-connectivity) of every class mask.
    Touching objects of the same class merge -> why real instance segmentation exists."""
    h, w = ids.shape
    min_area = max(80, int(0.0005 * h * w))
    inst = np.zeros((h, w, 4), np.uint8)
    pano = np.zeros((h, w, 3), np.uint8); pano[:] = (34, 40, 58)
    counts, n_total = {}, 0
    for c in np.unique(ids):
        if c == 0:
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats((ids == c).astype(np.uint8), connectivity=8)
        k = 0
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            k += 1; n_total += 1
            m = lab == i
            hue = int((n_total * 47) % 180)
            col = cv2.cvtColor(np.uint8([[[hue, 200, 235]]]), cv2.COLOR_HSV2RGB)[0, 0]
            inst[m] = (*col, 255)
            pano[m] = PAL[c]
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(pano, cnts, -1, (255, 255, 255), 2)
        counts[int(c)] = k
    return inst, pano, counts


def metrics(gt, pr):
    rows = []
    for c in np.unique(gt):
        g, p = gt == c, pr == c
        inter, union = int((g & p).sum()), int((g | p).sum())
        rows.append(dict(id=int(c), name=NAMES[c], pct=round(100 * g.mean(), 2),
                         iou=inter / union, dice=2 * inter / (int(g.sum()) + int(p.sum()))))
    return dict(acc=float((gt == pr).mean()), miou=float(np.mean([r["iou"] for r in rows])),
                mdice=float(np.mean([r["dice"] for r in rows])), rows=rows)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/info")
def info():
    samples = sorted(f for f in os.listdir(SAMPLES) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return jsonify(models={k: v[0] for k, v in MODELS.items()}, samples=samples)


@app.route("/samples/<path:name>")
def sample(name):
    return send_from_directory(SAMPLES, name)


@app.route("/api/segment", methods=["POST"])
def segment():
    try:
        if "file" in request.files:
            img = Image.open(request.files["file"].stream)
        else:
            img = Image.open(os.path.join(SAMPLES, os.path.basename(request.form["sample"])))
        img = img.convert("RGB")
        s = MAX_SIDE / max(img.size)
        if s < 1:
            img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
        key = request.form.get("model", "deeplab")
        t = time.time()
        ids, probs, tops = predict(img, key)
        ms = int((time.time() - t) * 1000)
        inst, pano, counts = instances_from_semantic(ids)
        token = uuid.uuid4().hex
        CACHE[token] = ids
        while len(CACHE) > 8:
            CACHE.popitem(last=False)
        classes = [dict(id=int(c), name=NAMES[c], color=PAL[c].tolist(), pct=round(100 * float((ids == c).mean()), 2),
                        instances=counts.get(int(c), 0)) for c in np.unique(ids)]
        classes.sort(key=lambda d: -d["pct"])
        return jsonify(token=token, width=img.width, height=img.height, ms=ms, names=NAMES, classes=classes,
                       image=b64(np.asarray(img)), overlay=b64(colorize(ids)), ids=b64(ids),
                       instances=b64(inst), panoptic=b64(pano), probs=b64(probs), tops=b64(tops))
    except Exception as e:  # shown in the UI
        return jsonify(error=f"{type(e).__name__}: {e}"), 500


@app.route("/api/metrics", methods=["POST"])
def metrics_api():
    d = request.get_json()
    gt = CACHE.get(d.get("token"))
    if gt is None:
        return jsonify(error="Run a segmentation first."), 400
    pr = gt.copy()
    if d.get("mode") == "background":
        pr[:] = 0
    else:
        pr[pr == int(d.get("drop", 0))] = 0
    return jsonify(**metrics(gt, pr), truth=b64(colorize(gt, 255)[..., :3]), pred=b64(colorize(pr, 255)[..., :3]))


def fetch_samples():
    urls = {"dog.jpg": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
            "deeplab1.png": "https://raw.githubusercontent.com/pytorch/hub/master/images/deeplab1.png",
            "deeplab2.png": "https://raw.githubusercontent.com/pytorch/hub/master/images/deeplab2.png"}
    for n, u in urls.items():
        p = os.path.join(SAMPLES, n)
        if not os.path.exists(p):
            try:
                urllib.request.urlretrieve(u, p)
            except Exception:
                pass  # offline: just upload your own photos


if __name__ == "__main__":
    threading.Thread(target=fetch_samples, daemon=True).start()
    def warm():
        try:
            get_model("deeplab")  # download/load now so the first click is fast
        except Exception as e:
            print("Model warm-up failed (will retry on first click):", e)
    threading.Thread(target=warm, daemon=True).start()
    print("Open http://127.0.0.1:5000  (first segmentation downloads the model weights once)")
    app.run(host="127.0.0.1", port=5000, threaded=True)

"""Semantic Segmentation live demo (Flask + PyTorch + OpenCV + SegFormer).
Run:  python app.py   ->  open http://127.0.0.1:5000

Why the old version only found "background": it used models trained on the 20 Pascal VOC object
classes (person, car, dog...). Roads, buildings, windows, grass and sky are not in that list, so
they all became "background". The default models now are SegFormer networks trained on ADE20K
(150 scene classes incl. road, building, windowpane, grass, sky, tree, sidewalk, floor, wall...).
"""
import base64, io, os, threading, time, urllib.request, uuid
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
import torchvision.models.segmentation as seg

BASE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(BASE, "samples")
MAX_SIDE = 640  # keeps CPU inference fast on a laptop
app = Flask(__name__, static_folder=os.path.join(BASE, "static"))
app.json.sort_keys = False  # keep the model list in the order defined below
torch.set_num_threads(max(1, os.cpu_count() or 4))
os.makedirs(SAMPLES, exist_ok=True)

VOC = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
       "dining table", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv monitor"]
# "Stuff" = big regions without separate objects. Only non-stuff classes get counted as instances.
ADE_STUFF = {"wall", "building", "sky", "floor", "tree", "ceiling", "road", "grass", "sidewalk", "earth", "mountain",
             "plant", "water", "sea", "field", "fence", "rug", "sand", "skyscraper", "path", "stairs", "stairway",
             "river", "bridge", "hill", "land", "dirt track", "runway", "waterfall", "lake", "swimming pool", "pier",
             "house", "tower"}
CITY_STUFF = {"road", "sidewalk", "building", "wall", "fence", "vegetation", "terrain", "sky"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


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


def _meta(names, bg, stuff, kind):
    pal = PAL if bg == 0 else np.roll(PAL, -1, axis=0)  # models without background: no black class 0
    return dict(names=names, bg=bg, kind=kind, pal=pal,
                stuff_ids=[i for i, n in enumerate(names) if n in stuff])


def _hf(repo, stuff):
    def make():
        from transformers import SegformerForSemanticSegmentation
        m = SegformerForSemanticSegmentation.from_pretrained(repo).eval()
        names = [m.config.id2label[i].split(",")[0].strip() for i in range(len(m.config.id2label))]
        return m, _meta(names, None, stuff, "hf")
    return make


def _tv(ctor, weights):
    def make():
        return ctor(weights=weights.DEFAULT).eval(), _meta(VOC, 0, {"background"}, "tv")
    return make


MODELS = OrderedDict(
    ade_b2=("Everyday scenes: SegFormer-B2, 150 kinds incl. road, building, window, grass (best, ~110 MB)",
            _hf("nvidia/segformer-b2-finetuned-ade-512-512", ADE_STUFF)),
    ade_b0=("Everyday scenes: SegFormer-B0, 150 kinds (fast, ~15 MB)",
            _hf("nvidia/segformer-b0-finetuned-ade-512-512", ADE_STUFF)),
    city_b0=("Street scenes: SegFormer-B0, 19 Cityscapes kinds (road, sidewalk, building, car...)",
             _hf("nvidia/segformer-b0-finetuned-cityscapes-1024-1024", CITY_STUFF)),
    deeplab=("Objects only: DeepLabV3-MobileNetV3, 20 kinds (no road/building/window)",
             _tv(seg.deeplabv3_mobilenet_v3_large, seg.DeepLabV3_MobileNet_V3_Large_Weights)),
    fcn=("Objects only: FCN-ResNet50, Long et al. 2015, 20 kinds (~135 MB)",
         _tv(seg.fcn_resnet50, seg.FCN_ResNet50_Weights)),
)
_loaded, _lock = {}, threading.Lock()
CACHE = OrderedDict()  # token -> {ids, meta}, used by the metrics tab


def get_model(key):
    with _lock:
        if key not in _loaded:
            _loaded[key] = MODELS[key][1]()
        return _loaded[key]


def b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def predict(img, key):
    model, meta = get_model(key)
    H, W = img.height, img.width
    x = torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1)
    x = ((x - MEAN) / STD)[None]
    with torch.no_grad():
        if meta["kind"] == "hf":
            h32, w32 = max(32, round(H / 32) * 32), max(32, round(W / 32) * 32)
            x = F.interpolate(x, size=(h32, w32), mode="bilinear", align_corners=False)
            logits = F.interpolate(model(pixel_values=x).logits, size=(H, W), mode="bilinear", align_corners=False)
        else:
            logits = model(x)["out"]
        pr = logits.softmax(1)[0]  # per-pixel "votes" for each class
    top = pr.topk(3, dim=0)
    probs = (top.values.numpy() * 255).round().astype(np.uint8).transpose(1, 2, 0)
    tops = top.indices.numpy().astype(np.uint8).transpose(1, 2, 0)
    return pr.argmax(0).numpy().astype(np.uint8), probs, tops


def colorize(ids, meta, bg_alpha=0):
    rgba = np.zeros((*ids.shape, 4), np.uint8)
    rgba[..., :3] = meta["pal"][ids]
    rgba[..., 3] = 255 if meta["bg"] is None else np.where(ids != meta["bg"], 255, bg_alpha)
    return rgba


def instances_from_semantic(ids, meta):
    """Approximation: connected components (8-connectivity) of every 'thing' class mask.
    Touching objects of the same class merge -> why real instance segmentation exists."""
    pal, bg, skip = meta["pal"], meta["bg"], set(meta["stuff_ids"])
    h, w = ids.shape
    min_area = max(80, int(0.0005 * h * w))
    inst = np.zeros((h, w, 4), np.uint8)
    pano = pal[ids].copy()
    if bg is not None:
        pano[ids == bg] = (34, 40, 58)
    counts, n_total = {}, 0
    for c in np.unique(ids):
        c = int(c)
        if c == bg or c in skip:
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats((ids == c).astype(np.uint8), connectivity=8)
        k = 0
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            k += 1; n_total += 1
            m = lab == i
            col = cv2.cvtColor(np.uint8([[[int((n_total * 47) % 180), 200, 235]]]), cv2.COLOR_HSV2RGB)[0, 0]
            inst[m] = (*col, 255)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(pano, cnts, -1, (255, 255, 255), 2)
        counts[c] = k
    return inst, pano, counts


def fill_label(gt, bg, exclude=None):
    """Label used for a 'lazy' prediction: background if the model has one, else the most common label."""
    if bg is not None:
        return bg
    vals, cnt = np.unique(gt[gt != exclude] if exclude is not None else gt, return_counts=True)
    return int(vals[cnt.argmax()]) if len(vals) else 0


def metrics(gt, pr, names):
    rows = []
    for c in np.unique(gt):
        g, p = gt == c, pr == c
        inter, union = int((g & p).sum()), int((g | p).sum())
        rows.append(dict(id=int(c), name=names[c], pct=round(100 * g.mean(), 2),
                         iou=inter / union, dice=2 * inter / (int(g.sum()) + int(p.sum()))))
    return dict(acc=float((gt == pr).mean()), miou=float(np.mean([r["iou"] for r in rows])),
                mdice=float(np.mean([r["dice"] for r in rows])), rows=rows)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/info")
def info():
    samples = sorted(f for f in os.listdir(SAMPLES) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return jsonify(models=[[k, v[0]] for k, v in MODELS.items()], samples=samples)


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
        key = request.form.get("model") or next(iter(MODELS))
        t = time.time()
        ids, probs, tops = predict(img, key)
        ms = int((time.time() - t) * 1000)
        meta = get_model(key)[1]
        names = meta["names"]
        inst, pano, counts = instances_from_semantic(ids, meta)
        token = uuid.uuid4().hex
        CACHE[token] = dict(ids=ids, meta=meta)
        while len(CACHE) > 8:
            CACHE.popitem(last=False)
        classes = [dict(id=int(c), name=names[c], color=meta["pal"][c].tolist(),
                        pct=round(100 * float((ids == c).mean()), 2), instances=counts.get(int(c), 0),
                        thing=int(c) != meta["bg"] and int(c) not in meta["stuff_ids"]) for c in np.unique(ids)]
        classes.sort(key=lambda d: -d["pct"])
        return jsonify(token=token, width=img.width, height=img.height, ms=ms, names=names, bg=meta["bg"],
                       classes=classes, image=b64(np.asarray(img)), overlay=b64(colorize(ids, meta)), ids=b64(ids),
                       instances=b64(inst), panoptic=b64(pano), probs=b64(probs), tops=b64(tops))
    except Exception as e:  # shown in the UI
        return jsonify(error=f"{type(e).__name__}: {e}"), 500


@app.route("/api/metrics", methods=["POST"])
def metrics_api():
    d = request.get_json()
    c = CACHE.get(d.get("token"))
    if c is None:
        return jsonify(error="Run a segmentation first."), 400
    gt, meta = c["ids"], c["meta"]
    pr = gt.copy()
    if d.get("mode") == "background":
        pr[:] = fill_label(gt, meta["bg"])
    else:
        drop = int(d.get("drop", 0))
        pr[pr == drop] = fill_label(gt, meta["bg"], exclude=drop)
    return jsonify(**metrics(gt, pr, meta["names"]), truth=b64(colorize(gt, meta, 255)[..., :3]),
                   pred=b64(colorize(pr, meta, 255)[..., :3]))


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
            get_model(next(iter(MODELS)))  # download/load now so the first click is fast
        except Exception as e:
            print("Model warm-up failed (will retry on first click):", e)
    threading.Thread(target=warm, daemon=True).start()
    print("Open http://127.0.0.1:5000  (the first run downloads the model weights once)")
    app.run(host="127.0.0.1", port=5000, threaded=True)

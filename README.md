# Semantic Segmentation Live Demo (v2, beginner-friendly)

Web demo for **Topics 3.2 Segmentation Techniques > Semantic Segmentation** (Flask + PyTorch + OpenCV).

## Run
**Windows:** double-click `run.bat`. **Manual:** `pip install -r requirements.txt`, `python app.py`, open http://127.0.0.1:5000
The first run downloads pretrained weights once (DeepLabV3-MobileNet ~42 MB, FCN-ResNet50 ~135 MB). CPU only is fine.
Sample photos download into `samples/` (or add your own .jpg/.png).

## Tabs (use them in this order)
1. **The idea** - a 96-pixel toy picture (camera view vs the computer's labels), the "votes" idea, and a plain-word glossary. Works with no photo.
2. **Try it** - real photo. Views: colors on photo, colors only, before/after slider, "how sure is the computer?" heatmap. The **pixel inspector** magnifies the real pixels under the mouse and shows the model's top-3 votes for that pixel. Click a class to isolate it. A plain-English summary appears under the picture.
3. **Compare** - semantic vs instance vs panoptic, with object counts (instance/panoptic are approximated with connected components).
4. **Score it** - click a 4x4 grid to see pixel accuracy, IoU, mean IoU and Dice change live; the "lazy computer" shows why accuracy alone can fool us. Then repeat on a real photo.
5. **Inside the network** - your photo shrunk step by step and grown back colored, plus the U-Net diagram with encoder, bottleneck, decoder and skip connections.

"Large text" (top right) enlarges everything for a projector.

## Files
`app.py` (server + models) - `static/index.html` (interface) - `requirements.txt` - `run.bat` / `run.sh`
Labels: the 20 Pascal VOC classes plus background.

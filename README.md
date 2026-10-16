# Semantic Segmentation Live Demo (fixed: detects roads, buildings, windows, grass...)

## What was wrong
The first versions used DeepLabV3/FCN models trained on **Pascal VOC**, which only know 20 object kinds
(person, car, dog, bus, ...) plus "background". Roads, buildings, windows, grass, sky and walls are not in
that list, so they were all painted as **background**.

## The fix
The default models are now **SegFormer** networks trained on **ADE20K** (150 scene kinds: wall, building, sky,
floor, tree, road, windowpane, grass, sidewalk, person, car, door, ...). The model menu offers:
- SegFormer-B2 ADE20K: best quality (~110 MB download, first run only)
- SegFormer-B0 ADE20K: fast (~15 MB)
- SegFormer-B0 Cityscapes: 19 street kinds (road, sidewalk, building, car, pole, sign, vegetation, sky...)
- DeepLabV3 and FCN (VOC, objects only): kept to show the difference
Models without a background class are handled everywhere (overlay, counts, scores). Models are downloaded from
huggingface.co on first use, so the first run needs internet.

## Run
Windows: double-click `run.bat`, or run `powershell -ExecutionPolicy Bypass -File ./run.ps1` from PowerShell. Linux/macOS: run `./run.sh`.
The launchers create a local virtual environment, install the requirements, and use its Python automatically.
Then open http://127.0.0.1:5000.

Use photos of streets, buildings, rooms or parks. Note: Cityscapes has no "window" class; use the ADE20K models for windows.

## Tabs
1 The idea - 2 Try it (pixel inspector, confidence heatmap, before/after) - 3 Compare - 4 Score it - 5 Inside the network

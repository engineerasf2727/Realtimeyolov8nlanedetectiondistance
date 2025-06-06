# Realtimeyolov8nlanedetectiondistance

This repository demonstrates lane detection and object detection using YOLOv8. It includes scripts and pretrained weights.

## Exporting YOLOv8 weights

The repository now contains a helper script `convert_to_coreml.py` which converts the provided `weights/yolov8n.pt` file to ONNX and Core ML formats. Usage:

```bash
python3 convert_to_coreml.py madebybaris/weights/yolov8n.pt --onnx yolov8n.onnx --coreml yolov8n.mlpackage
```

This requires the `ultralytics`, `coremltools`, and `onnx` Python packages.

import argparse
from ultralytics import YOLO
import coremltools as ct

parser = argparse.ArgumentParser(description="Export YOLOv8 weights to ONNX and CoreML")
parser.add_argument("weights", help="path to .pt weights file")
parser.add_argument("--img", type=int, default=640, help="inference image size")
parser.add_argument("--onnx", default="model.onnx", help="output ONNX file")
parser.add_argument("--coreml", default="model.mlpackage", help="output CoreML file")
args = parser.parse_args()

model = YOLO(args.weights)

onnx_path = args.onnx
coreml_path = args.coreml

# Export to ONNX
model.export(format="onnx", imgsz=args.img, dynamic=True, simplify=True, opset=12, half=False, half_float=False, device="cpu", optimize=False, int8=False, path=onnx_path)

# Load the ONNX model and convert to CoreML
mlmodel = ct.converters.onnx.convert(model=onnx_path)
mlmodel.save(coreml_path)
print(f"Saved ONNX model to {onnx_path} and CoreML model to {coreml_path}")

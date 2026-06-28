#!/bin/bash
# Build standalone executable for Linux and Windows (cross-compile)
set -e

cd "$(dirname "$0")"

echo "Building Liodon Dental standalone executable..."

pyinstaller \
  --onefile \
  --name "liodon-dental" \
  --add-data "model.onnx:." \
  --hidden-import "uvicorn.logging" \
  --hidden-import "uvicorn.loops" \
  --hidden-import "uvicorn.loops.auto" \
  --hidden-import "uvicorn.protocols" \
  --hidden-import "uvicorn.protocols.http" \
  --hidden-import "uvicorn.protocols.http.auto" \
  --hidden-import "uvicorn.protocols.websockets" \
  --hidden-import "uvicorn.protocols.websockets.auto" \
  --hidden-import "uvicorn.lifespan" \
  --hidden-import "uvicorn.lifespan.on" \
  --hidden-import "onnxruntime" \
  --hidden-import "onnxruntime.capi._pybind_state" \
  --hidden-import "PIL" \
  --hidden-import "multipart" \
  --collect-submodules "onnxruntime.capi" \
  --exclude-module "torch" \
  --exclude-module "torchvision" \
  --exclude-module "torchaudio" \
  --exclude-module "ultralytics" \
  --exclude-module "onnxruntime.tools" \
  --exclude-module "onnxruntime.transformers" \
  --exclude-module "onnxruntime.training" \
  --exclude-module "onnxruntime.quantization" \
  --exclude-module "scipy" \
  --exclude-module "sklearn" \
  --exclude-module "matplotlib" \
  --exclude-module "pandas" \
  --exclude-module "cv2" \
  --exclude-module "tensorflow" \
  server.py

echo ""
echo "Done: dist/liodon-dental"
echo "Run: ./dist/liodon-dental"
echo "Then open http://localhost:8080 in your browser"

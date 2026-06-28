"""
Shared ONNX inference for dental panoramic X-ray detection.
No PyTorch or Ultralytics dependency at runtime — only onnxruntime + Pillow.
"""

import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASSES = ["caries", "periapical_lesion", "impacted_tooth"]
COLORS  = ["#2196F3", "#00BCD4", "#FFFFFF"]  # blue, cyan, white
CONF    = 0.45
IOU     = 0.35
IMGSZ   = 640


def _letterbox(img: Image.Image):
    """Letterbox to IMGSZ×IMGSZ with grey padding, return canvas + metadata."""
    orig_w, orig_h = img.size
    scale = min(IMGSZ / orig_w, IMGSZ / orig_h)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)
    pad_x = (IMGSZ - new_w) // 2
    pad_y = (IMGSZ - new_h) // 2
    canvas = Image.new("RGB", (IMGSZ, IMGSZ), (114, 114, 114))
    canvas.paste(img.resize((new_w, new_h), Image.BILINEAR), (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y, orig_w, orig_h


def _preprocess(img: Image.Image):
    """Letterbox + normalize to (1, 3, 640, 640) float32."""
    canvas, scale, pad_x, pad_y, orig_w, orig_h = _letterbox(img.convert("RGB"))
    arr = np.array(canvas, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]
    return arr, scale, pad_x, pad_y, orig_w, orig_h


def _iou(a, b):
    """IoU between boxes [x1,y1,x2,y2]."""
    xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (ua + 1e-6)


def _nms(boxes, scores, iou_thresh):
    """Greedy NMS. boxes: (N,4), scores: (N,). Returns kept indices."""
    order = np.argsort(scores)[::-1]
    kept  = []
    while len(order):
        i = order[0]
        kept.append(i)
        rest = order[1:]
        order = rest[[_iou(boxes[i], boxes[j]) < iou_thresh for j in rest]]
    return kept


def _postprocess(output, scale, pad_x, pad_y, orig_w, orig_h, conf=CONF, iou=IOU):
    """
    YOLO11 ONNX output: (1, 7, 8400) — [x_center, y_center, w, h, cls0, cls1, cls2]
    Coordinates in letterboxed 640px space — undo padding + scale back to original.
    """
    pred = output[0].squeeze(0).T   # (8400, 7)
    cls_scores = pred[:, 4:]
    cls_ids    = cls_scores.argmax(axis=1)
    confs      = cls_scores.max(axis=1)

    mask    = confs >= conf
    boxes   = pred[mask, :4]
    confs   = confs[mask]
    cls_ids = cls_ids[mask]

    if len(boxes) == 0:
        return []

    # xywh → xyxy in letterbox space
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    # Remove letterbox padding and rescale to original image
    x1 = np.clip((x1 - pad_x) / scale, 0, orig_w)
    y1 = np.clip((y1 - pad_y) / scale, 0, orig_h)
    x2 = np.clip((x2 - pad_x) / scale, 0, orig_w)
    y2 = np.clip((y2 - pad_y) / scale, 0, orig_h)
    xyxy = np.stack([x1, y1, x2, y2], axis=1)

    results = []
    for cid in np.unique(cls_ids):
        m    = cls_ids == cid
        idxs = np.where(m)[0]
        kept = _nms(xyxy[m], confs[m], iou)
        for k in kept:
            idx = idxs[k]
            results.append((int(cid), float(confs[idx]), *xyxy[idx].tolist()))

    return results


def _draw(img: Image.Image, detections):
    """Draw bounding boxes and labels on image."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except Exception:
        font = small = ImageFont.load_default()

    for cls_id, conf, x1, y1, x2, y2 in detections:
        color = COLORS[cls_id]
        lw    = max(3, int(img.width / 400))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        label = f"{CLASSES[cls_id]} {conf:.2f}"
        bbox  = draw.textbbox((x1, y1 - 30), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 30), label, fill="black", font=font)

    # Legend
    lx, ly = 20, img.height - 120
    for i, (cls, col) in enumerate(zip(CLASSES, COLORS)):
        draw.rectangle([lx, ly + i*36, lx+24, ly + i*36+24], fill=col, outline="black", width=1)
        draw.text((lx + 32, ly + i*36), cls.replace("_", " ").title(), fill="white", font=small)

    return img


class DentalDetector:
    def __init__(self, model_path: str):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name  = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def predict(self, img: Image.Image, conf=CONF, iou=IOU):
        """Run inference. Returns annotated PIL image + list of detections."""
        tensor, scale, pad_x, pad_y, orig_w, orig_h = _preprocess(img)
        output = self.sess.run([self.output_name], {self.input_name: tensor})
        detections = _postprocess(output, scale, pad_x, pad_y, orig_w, orig_h, conf, iou)
        annotated  = _draw(img.copy(), detections)
        return annotated, detections

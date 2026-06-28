"""
Liodon AI — Dental Panoramic Detector
Local web server. Run this file; open http://localhost:8080 in your browser.
"""

import io
import os
import sys
import base64
import threading
import webbrowser
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from PIL import Image

# Resolve model path (works both in dev and PyInstaller bundle)
if getattr(sys, "frozen", False):
    BASE = Path(sys._MEIPASS)
else:
    BASE = Path(__file__).parent

MODEL_PATH = BASE / "model.onnx"

from inference import DentalDetector
detector = DentalDetector(str(MODEL_PATH))

app = FastAPI()

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Liodon AI — Dental Panoramic Detector</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e0e0e0; min-height: 100vh; }
  header { background: #1a1d27; border-bottom: 1px solid #2a2d3a;
           padding: 16px 32px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 20px; font-weight: 600; color: #fff; }
  header span { font-size: 13px; color: #888; }
  .badge { background: #2196F3; color: #fff; font-size: 11px; font-weight: 600;
           padding: 2px 8px; border-radius: 99px; }
  main { max-width: 1100px; margin: 40px auto; padding: 0 24px; }
  #drop-zone { border: 2px dashed #2a2d3a; border-radius: 12px;
               padding: 60px 40px; text-align: center; cursor: pointer;
               transition: all 0.2s; background: #1a1d27; }
  #drop-zone:hover, #drop-zone.drag-over { border-color: #2196F3; background: #1a2035; }
  #drop-zone svg { width: 48px; height: 48px; color: #555; margin-bottom: 16px; }
  #drop-zone h2 { font-size: 18px; margin-bottom: 8px; color: #ccc; }
  #drop-zone p  { color: #666; font-size: 14px; }
  #file-input { display: none; }
  .btn { display: inline-block; background: #2196F3; color: #fff; border: none;
         padding: 10px 24px; border-radius: 8px; cursor: pointer; font-size: 14px;
         font-weight: 600; margin-top: 16px; transition: background 0.2s; }
  .btn:hover { background: #1976D2; }
  #status { margin: 24px 0; font-size: 14px; color: #888; text-align: center; }
  #results { display: none; }
  #results h2 { font-size: 16px; font-weight: 600; margin-bottom: 16px; color: #ccc; }
  #result-img { width: 100%; border-radius: 8px; border: 1px solid #2a2d3a; }
  .findings { display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }
  .finding { padding: 6px 14px; border-radius: 99px; font-size: 13px; font-weight: 600; }
  .caries            { background: #1565C0; color: #fff; }
  .periapical_lesion { background: #00838F; color: #fff; }
  .impacted_tooth    { background: #37474F; color: #fff; }
  .download-btn { display: inline-block; margin-top: 12px; background: #1a1d27;
                  border: 1px solid #2a2d3a; color: #ccc; padding: 8px 20px;
                  border-radius: 8px; text-decoration: none; font-size: 13px; }
  .download-btn:hover { border-color: #2196F3; color: #fff; }
  .disclaimer { margin-top: 40px; padding: 16px; background: #1a1d27;
                border: 1px solid #2a2d3a; border-radius: 8px; font-size: 12px;
                color: #666; line-height: 1.6; }
  .spinner { display: inline-block; width: 20px; height: 20px; border: 2px solid #333;
             border-top-color: #2196F3; border-radius: 50%; animation: spin 0.8s linear infinite;
             vertical-align: middle; margin-right: 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<header>
  <h1>Liodon AI</h1>
  <span>Dental Panoramic Detector</span>
  <span class="badge">v1</span>
</header>
<main>
  <div id="drop-zone" onclick="document.getElementById('file-input').click()">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
      <path stroke-linecap="round" stroke-linejoin="round"
            d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/>
    </svg>
    <h2>Drop panoramic X-ray here</h2>
    <p>or click to browse — PNG, JPG, BMP supported</p>
    <button class="btn" onclick="event.stopPropagation(); document.getElementById('file-input').click()">
      Select File
    </button>
    <input type="file" id="file-input" accept="image/*">
  </div>

  <div id="status"></div>

  <div id="results">
    <h2>Findings</h2>
    <div class="findings" id="findings-list"></div>
    <img id="result-img" src="" alt="Annotated X-ray">
    <br>
    <a id="download-link" class="download-btn" href="#" download="liodon_dental_result.png">
      Download annotated image
    </a>
  </div>

  <div class="disclaimer">
    <strong>Clinical disclaimer:</strong> This tool is intended as a decision-support aid only.
    All findings must be reviewed and confirmed by a licensed dental professional.
    Not cleared by the FDA for diagnostic use.
  </div>
</main>

<script>
const dropZone  = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const status    = document.getElementById('status');
const results   = document.getElementById('results');

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) processFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) processFile(fileInput.files[0]); });

async function processFile(file) {
  status.innerHTML = '<span class="spinner"></span> Analyzing...';
  results.style.display = 'none';

  const form = new FormData();
  form.append('file', file);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: form });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();

    document.getElementById('result-img').src = 'data:image/png;base64,' + data.image;
    document.getElementById('download-link').href = 'data:image/png;base64,' + data.image;

    const list = document.getElementById('findings-list');
    list.innerHTML = '';
    if (data.detections.length === 0) {
      list.innerHTML = '<span style="color:#666">No findings detected above threshold</span>';
    } else {
      const counts = {};
      data.detections.forEach(d => counts[d.class] = (counts[d.class] || 0) + 1);
      Object.entries(counts).forEach(([cls, n]) => {
        const el = document.createElement('span');
        el.className = 'finding ' + cls;
        el.textContent = cls.replace(/_/g,' ').replace(/\\b\\w/g, c=>c.toUpperCase()) + (n>1 ? ' ×'+n : '');
        list.appendChild(el);
      });
    }

    status.innerHTML = '';
    results.style.display = 'block';
  } catch (err) {
    status.innerHTML = '<span style="color:#f44">Error: ' + err.message + '</span>';
  }
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    annotated, detections = detector.predict(img)

    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    det_list = [
        {"class": ["caries", "periapical_lesion", "impacted_tooth"][d[0]],
         "confidence": round(d[1], 3),
         "box": [round(x, 1) for x in d[2:]]}
        for d in detections
    ]

    return JSONResponse({"image": b64, "detections": det_list})


def main():
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:8080")).start()
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")


if __name__ == "__main__":
    main()

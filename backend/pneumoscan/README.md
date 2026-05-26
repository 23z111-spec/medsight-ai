# PneumoScan AI — Backend

FastAPI backend for the Chest X-Ray screening dashboard.
Built for PS5 internship project — EfficientNet-B3 + Grad-CAM.

---

## Project Structure

```
pneumoscan/
├── main.py                  ← FastAPI app + CORS
├── requirements.txt
├── models/
│   └── efficientnet_b3_chestxray.pth   ← drop your weights here
├── app/
│   ├── model_loader.py      ← model build, load, preprocess
│   ├── schemas.py           ← Pydantic request/response shapes
│   └── routes/
│       ├── health.py        ← GET  /health
│       ├── predict.py       ← POST /predict
│       └── patients.py      ← GET  /patients/{id}
└── utils/
    └── gradcam.py           ← Grad-CAM heatmap generator
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
uvicorn main:app --reload --port 8000
```

API docs auto-generated at → http://localhost:8000/docs

---

## Switching from Mock to Real Model

Open `app/model_loader.py` and make two changes:

```python
MODEL_PATH = "models/efficientnet_b3_chestxray.pth"  # ← your .pth file
MOCK_MODE  = False                                     # ← flip this
```

Restart the server. Done.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Model status, device, mock mode |
| POST | `/predict` | Upload X-ray → findings + Grad-CAM |
| GET | `/patients` | List all patients |
| GET | `/patients/{id}` | Single patient + scan history |

---

## Connecting to dashboard.html

In `dashboard.html`, replace the `handleUpload()` function with:

```javascript
async function handleUpload(event) {
  const file = event.target.files[0];
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData,
  });

  const data = await res.json();

  // Update confidence
  document.getElementById('confVal').textContent =
    (data.confidence * 100).toFixed(1) + '%';

  // Update heatmap overlay
  document.getElementById('hmOverlay').style.backgroundImage =
    `url(data:image/png;base64,${data.gradcam_base64})`;

  // Update findings panel...
  console.log(data);
}
```

---

## Training Tips (EfficientNet-B3 on NIH ChestXray14)

- Use `torchvision.models.efficientnet_b3(weights='IMAGENET1K_V1')` for transfer learning
- Class weights to handle imbalance: `torch.nn.BCEWithLogitsLoss(pos_weight=...)`
- Target AUC ≥ 0.90 is achievable in 20–30 epochs on Colab A100
- Save best checkpoint: `torch.save(model.state_dict(), 'efficientnet_b3_chestxray.pth')`
- Copy the `.pth` into the `models/` folder here when done

"""
Model Loader
============
Loads EfficientNet-B3 fine-tuned on NIH ChestXray14.

WHEN YOUR TRAINING IS DONE:
  1. Copy your .pth file into the /models/ folder
  2. Set MODEL_PATH below to point to it
  3. Set MOCK_MODE = False
  4. Restart the server — that's it.
"""

import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG  ← only things you need to change
# ─────────────────────────────────────────────
MODEL_PATH  = "models/efficientnet_b3_chestxray.pth"   # path to your weights
NUM_CLASSES = 5                                          # pneumonia, tb, cardiomegaly, pleural effusion, normal
MOCK_MODE   = True    # ← set False once weights are ready
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Labels must match your training label order
LABELS = [
    "Pneumonia",
    "Tuberculosis",
    "Cardiomegaly",
    "Pleural Effusion",
    "Normal",
]

# ─────────────────────────────────────────────
# IMAGE PREPROCESSING  (matches training transforms)
# ─────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.Grayscale(num_output_channels=3),   # X-rays are grayscale; EfficientNet needs 3ch
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats — used for transfer learning
        std=[0.229, 0.224, 0.225],
    ),
])


def build_model() -> nn.Module:
    """Build EfficientNet-B3 with a custom classifier head."""
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def load_model() -> nn.Module | None:
    """Load model weights. Returns None in mock mode."""
    if MOCK_MODE:
        logger.warning("MOCK_MODE is ON — returning dummy predictions. Set MOCK_MODE=False when weights are ready.")
        return None

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found at '{MODEL_PATH}'.\n"
            "Either set MOCK_MODE=True or copy your .pth file there."
        )

    model = build_model()
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    logger.info(f"Model loaded from {MODEL_PATH} on {DEVICE}")
    return model


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Convert raw image bytes → normalised tensor ready for inference."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)   # add batch dim
    return tensor

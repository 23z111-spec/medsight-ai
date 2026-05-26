"""
Grad-CAM  (Gradient-weighted Class Activation Mapping)
=======================================================
Selvaraju et al., 2017  —  https://arxiv.org/abs/1610.02391

Produces a heatmap showing WHICH regions of the X-ray the model
focused on when making its prediction.  Sent back to the dashboard
as a base64-encoded PNG so no file I/O is needed.
"""

import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

from app.model_loader import DEVICE


class GradCAM:
    """
    Hooks into the last convolutional block of EfficientNet-B3
    (features[-1]) and computes the class-weighted heatmap.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # EfficientNet-B3: target layer is the last conv block
        target_layer = self.model.features[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, tensor: torch.Tensor, class_idx: int) -> str:
        """
        Run forward + backward pass, produce heatmap.

        Args:
            tensor:     Preprocessed image tensor  [1, 3, H, W]
            class_idx:  Index of the target class  (usually argmax of logits)

        Returns:
            base64-encoded PNG string of the coloured heatmap
        """
        self.model.zero_grad()
        tensor.requires_grad_(True)

        logits = self.model(tensor)                     # forward pass
        score = logits[0, class_idx]
        score.backward()                                # backward pass

        # Pool gradients across spatial dims → channel weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)   # [1, C, 1, 1]

        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Resize to input resolution (300×300) and convert to uint8
        cam_np = cam.squeeze().cpu().numpy()
        cam_resized = cv2.resize(cam_np, (300, 300))
        cam_uint8 = np.uint8(255 * cam_resized)

        # Apply COLORMAP_JET for the red-yellow heatmap look
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Encode as base64 PNG
        pil_img = Image.fromarray(heatmap_rgb)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded


def mock_gradcam() -> str:
    """
    Returns a placeholder heatmap when MOCK_MODE is ON.
    Generates a simple red-yellow radial gradient as a stand-in.
    """
    size = 300
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = (int(size * 0.65), int(size * 0.33))   # roughly RUL position
    for y in range(size):
        for x in range(size):
            dist = ((x - center[0])**2 + (y - center[1])**2) ** 0.5
            val = max(0, 1 - dist / 120)
            img[y, x] = [int(255 * val), int(100 * val), 0]  # red → orange fade

    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

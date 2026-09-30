"""Grad-CAM: visualize which image regions drive a model's prediction."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    """Return the last nn.Conv2d in the model (works for all registered architectures)."""
    last = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    if last is None:
        raise ValueError("No Conv2d layer found in model.")
    return last


class GradCAM:
    """Standard Grad-CAM via forward/backward hooks on a target conv layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model.eval()
        self.target_layer = target_layer or find_last_conv_layer(model)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inputs, output):
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, image: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """image: (1, C, H, W) on the model's device. Returns CAM as (H, W) in [0, 1]."""
        self.model.zero_grad()
        with torch.enable_grad():
            image = image.detach().requires_grad_(True)
            logits = self.model(image)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
            score = logits[0, class_idx]
            score.backward(retain_graph=True)
        if self._gradients is None or self._activations is None:
            raise RuntimeError("Grad-CAM hooks did not fire for the target layer.")

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)          # (1, K, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)      # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def close(self) -> None:
        for h in self._handles:
            h.remove()

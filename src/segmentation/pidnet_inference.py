"""
PIDNet inference helpers.

The official PIDNet package (github.com/XuJiacong/PIDNet) must be on sys.path
before these functions are called.  run_full_pipeline.py handles that.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

# ImageNet normalisation used by PIDNet
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def choose_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_bgr_to_tensor(
    frame_bgr: np.ndarray,
    input_size: tuple[int, int],  # (width, height)
) -> torch.Tensor:
    """
    Resize → RGB → normalize → CHW float32 tensor with batch dim.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, input_size, interpolation=cv2.INTER_LINEAR)
    chw = ((rgb - _MEAN) / _STD).transpose(2, 0, 1)
    return torch.from_numpy(chw).unsqueeze(0)


def load_pidnet_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> int:
    """
    Load a .pt / .pth checkpoint into model (strict=False, shape-matched).

    Handles full dicts with 'state_dict' key and DDP 'module.' prefixes.

    Returns the number of tensors successfully loaded.
    """
    ckpt = torch.load(str(checkpoint_path), map_location=device)

    # Unwrap common wrapper keys
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "net"):
            if key in ckpt:
                ckpt = ckpt[key]
                break

    model_sd = model.state_dict()
    matched: dict = {}
    for k, v in ckpt.items():
        k_bare = k
        # Strip common wrappers: DDP 'module.' and framework 'model.' prefixes
        for prefix in ("module.", "model."):
            if k_bare.startswith(prefix):
                k_bare = k_bare[len(prefix):]
        if k_bare in model_sd and v.shape == model_sd[k_bare].shape:
            matched[k_bare] = v

    model.load_state_dict(matched, strict=False)
    return len(matched)


def predict_mask_logits(
    model: torch.nn.Module,
    frame_bgr: np.ndarray,
    input_size: tuple[int, int],
    device: torch.device,
    align_corners: bool,
    output_index: int,
) -> torch.Tensor:
    """
    Run one forward pass and return raw logits tensor (1, C, H', W').

    output_index=-1 picks index 1 when the model returns a list (augment=False
    returns a single tensor; augment=True returns [p, main, d] — we want index 1).
    """
    tensor = normalize_bgr_to_tensor(frame_bgr, input_size).to(device)
    out = model(tensor)
    if isinstance(out, (list, tuple)):
        idx = output_index if output_index >= 0 else (1 if len(out) > 1 else 0)
        logits = out[idx]
    else:
        logits = out
    return logits

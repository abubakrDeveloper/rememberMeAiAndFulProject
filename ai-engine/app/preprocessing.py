"""Frame preprocessing optimized for low-quality CCTV footage.

Typical classroom CCTV issues addressed:
- Low resolution / small faces at distance
- Poor or uneven lighting (fluorescent flicker, backlighting)
- Compression artifacts from DVR/NVR encoding
- Motion blur from low frame-rate capture
"""

import cv2
import numpy as np


def enhance_for_cctv(
    frame: np.ndarray,
    *,
    upscale_factor: float = 1.0,
    clahe_clip: float = 2.5,
    clahe_grid: int = 8,
    denoise_strength: int = 5,
    sharpen: bool = True,
) -> np.ndarray:
    """Apply a sequence of enhancements to a raw CCTV frame.

    Pipeline:
    1. Optional upscale (for very low-res feeds like 320x240 or 640x480).
    2. CLAHE on the luminance channel to fix uneven / low lighting.
    3. Mild denoising to reduce DVR compression artifacts.
    4. Optional sharpening to recover edge detail after denoising.
    """
    enhanced = frame

    # 1. Upscale small frames so face detector can find tiny faces
    if upscale_factor > 1.0:
        h, w = enhanced.shape[:2]
        new_w = int(w * upscale_factor)
        new_h = int(h * upscale_factor)
        enhanced = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 2. CLAHE on L channel of LAB color space
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge([l_channel, a_channel, b_channel])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # 3. Mild denoising
    if denoise_strength > 0:
        enhanced = cv2.fastNlMeansDenoisingColored(
            enhanced,
            None,
            h=denoise_strength,
            hForColorComponents=denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    # 4. Gentle unsharp mask to recover edges
    if sharpen:
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.0)
        enhanced = cv2.addWeighted(enhanced, 1.4, blurred, -0.4, 0)

    return enhanced


def compute_upscale_factor(frame_width: int, target_min_width: int = 960) -> float:
    """Return how much to upscale so faces are large enough for the detector."""
    if frame_width >= target_min_width:
        return 1.0
    return target_min_width / max(1, frame_width)


def enhance_roi(
    roi: np.ndarray,
    *,
    clahe_clip: float = 3.0,
    clahe_grid: int = 4,
) -> np.ndarray:
    """Lighter enhancement targeted at a single person crop before face detection."""
    if roi.size == 0:
        return roi

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

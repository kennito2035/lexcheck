from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from config import YOLO_WEIGHTS_PATH
from pipeline.handwriting.detector import LetterDetection

logger = logging.getLogger(__name__)


def generate_gradcam_heatmap(
    original_bgr: np.ndarray,
    resized_rgb_float: np.ndarray,
    output_dir: Path,
    weights_path: Path | None = None,
    detections: list[LetterDetection] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = output_dir / "gradcam_heatmap.png"

    weights_path = weights_path or YOLO_WEIGHTS_PATH
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found at {weights_path}")

    try:
        from ultralytics import YOLO
        import torch
        import torch.nn as nn
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError as exc:
        raise ImportError(
            "Missing Grad-CAM dependencies. Ensure ultralytics, torch, and pytorch-grad-cam are installed."
        ) from exc

    try:
        yolo = YOLO(str(weights_path))
        torch_model = getattr(yolo, "model", None)
        if torch_model is None:
            raise RuntimeError("YOLO model does not expose a torch model via .model")

        class _ForwardTensorWrapper(nn.Module):
            def __init__(self, model: nn.Module) -> None:
                super().__init__()
                self.model = model

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.model(x)
                if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                    return out[0]
                if isinstance(out, torch.Tensor):
                    return out
                raise RuntimeError(f"Unexpected YOLO forward output type: {type(out)!r}")

        torch_model.eval()
        wrapped_model = _ForwardTensorWrapper(torch_model)

        from config import YOLO_IMG_SIZE

        def _select_focus_detection(dets: list[LetterDetection] | None) -> LetterDetection | None:
            if not dets:
                return None
            reversals = [d for d in dets if d.class_name == "Reversal"]
            pool = reversals if reversals else list(dets)
            return max(pool, key=lambda d: float(d.confidence))

        focus_det = _select_focus_detection(detections)

        class _BoxMeanTarget:
            def __init__(self, det: LetterDetection | None) -> None:
                self._det = det

            def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
                if not isinstance(model_output, torch.Tensor):
                    raise RuntimeError(f"Unexpected model_output type for CAM target: {type(model_output)!r}")

                if self._det is None:
                    return model_output.mean()

                if model_output.ndim != 4:
                    return model_output.mean()

                _, _, h, w = model_output.shape
                x1, y1, x2, y2 = self._det.bbox_xyxy
                ix1 = int(max(0, min(w - 1, round(float(x1) / float(YOLO_IMG_SIZE) * w))))
                iy1 = int(max(0, min(h - 1, round(float(y1) / float(YOLO_IMG_SIZE) * h))))
                ix2 = int(max(ix1 + 1, min(w, round(float(x2) / float(YOLO_IMG_SIZE) * w))))
                iy2 = int(max(iy1 + 1, min(h, round(float(y2) / float(YOLO_IMG_SIZE) * h))))
                return model_output[:, :, iy1:iy2, ix1:ix2].mean()

        detect_prefix: str | None = None
        for name, module in torch_model.named_modules():
            if module.__class__.__name__ == "Detect":
                detect_prefix = name
                break

        last_conv: nn.Module | None = None
        last_conv_name: str | None = None
        for name, module in torch_model.named_modules():
            if detect_prefix is not None and (name == detect_prefix or name.startswith(detect_prefix + ".")):
                continue
            if isinstance(module, nn.Conv2d):
                last_conv = module
                last_conv_name = name
        if last_conv is None:
            raise RuntimeError("No Conv2d layer found for Grad-CAM target.")

        cam = GradCAM(model=wrapped_model, target_layers=[last_conv])
        input_tensor = torch.from_numpy(resized_rgb_float.transpose(2, 0, 1)).unsqueeze(0)
        input_tensor = input_tensor.to(next(wrapped_model.parameters()).device).float()
        input_tensor.requires_grad_(True)
        for p in wrapped_model.parameters():
            p.requires_grad_(True)

        try:
            with torch.enable_grad():
                grayscale_cam = cam(input_tensor=input_tensor, targets=[_BoxMeanTarget(focus_det)])
        except RuntimeError as exc:
            msg = str(exc)
            if (
                "does not require grad" not in msg
                and "grad can be implicitly created only for scalar outputs" not in msg
                and "grad can be implicitly created" not in msg
            ):
                raise
            from pytorch_grad_cam import EigenCAM

            eigen_cam = EigenCAM(model=wrapped_model, target_layers=[last_conv])
            grayscale_cam = eigen_cam(input_tensor=input_tensor, targets=None)
        cam_map = grayscale_cam[0]
        if last_conv_name is not None:
            logger.info("Grad-CAM target layer: %s", last_conv_name)

        h, w = cam_map.shape[:2]
        focus_dets = None
        if detections:
            reversals = [d for d in detections if d.class_name == "Reversal"]
            focus_dets = reversals if reversals else detections

        if focus_dets:
            import cv2

            mask = np.zeros((h, w), dtype=np.float32)
            box_map = np.zeros((h, w), dtype=np.float32)
            used = 0
            for d in focus_dets:
                x1, y1, x2, y2 = d.bbox_xyxy
                ix1 = int(max(0, min(w - 1, round(float(x1) / float(YOLO_IMG_SIZE) * w))))
                iy1 = int(max(0, min(h - 1, round(float(y1) / float(YOLO_IMG_SIZE) * h))))
                ix2 = int(max(0, min(w, round(float(x2) / float(YOLO_IMG_SIZE) * w))))
                iy2 = int(max(0, min(h, round(float(y2) / float(YOLO_IMG_SIZE) * h))))
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                mask[iy1:iy2, ix1:ix2] = 1.0
                used += 1

                class_weight = 0.35
                if d.class_name == "Reversal":
                    class_weight = 1.0
                elif d.class_name == "Corrected":
                    class_weight = 0.6
                box_map[iy1:iy2, ix1:ix2] += float(d.confidence) * class_weight
            logger.info("Grad-CAM detections used for attention: %s/%s", used, len(focus_dets))
            if mask.max() > 0:
                mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=7.0, sigmaY=7.0)
                mmax = float(mask.max())
                if mmax > 0:
                    mask = mask / mmax
                cam_map = cam_map.astype(np.float32) * mask

            box_map = cv2.GaussianBlur(box_map, (0, 0), sigmaX=3.0, sigmaY=3.0)
            bmin = float(box_map.min())
            bmax = float(box_map.max())
            if bmax > bmin:
                box_map = (box_map - bmin) / (bmax - bmin + 1e-8)

            cam_min = float(cam_map.min())
            cam_max = float(cam_map.max())
            if cam_max > cam_min:
                cam_map = (cam_map - cam_min) / (cam_max - cam_min + 1e-8)

            if float(cam_map.std()) < 1e-3 and float(box_map.max()) > 0:
                logger.info("Grad-CAM map low-variance; using detection-based attention map fallback.")
                cam_map = box_map.astype(np.float32)
            else:
                cam_map = np.clip(0.70 * cam_map + 0.30 * box_map, 0.0, 1.0)
        else:
            gray = resized_rgb_float.mean(axis=2)
            p10 = float(np.percentile(gray, 10))
            p90 = float(np.percentile(gray, 90))
            thresh = (p10 + p90) / 2.0
            if float(gray.mean()) < 0.5:
                fg = (gray >= thresh).astype(np.float32)
            else:
                fg = (gray <= thresh).astype(np.float32)
            cam_map = cam_map.astype(np.float32) * fg

        cam_map = np.clip(cam_map, 0.0, 1.0)

        cam_overlay = show_cam_on_image(resized_rgb_float, cam_map, use_rgb=True)

        import cv2

        overlay_bgr = cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR)
        if detections:
            def _per_class_panel(class_name: str, title: str, color: tuple[int, int, int]) -> np.ndarray:
                class_box_map = np.zeros((h, w), dtype=np.float32)
                class_dets = [d for d in detections if d.class_name == class_name]
                class_mask = np.zeros((h, w), dtype=np.float32)

                if class_name == "Reversal":
                    class_weight = 1.0
                elif class_name == "Corrected":
                    class_weight = 0.6
                else:
                    class_weight = 0.35

                for d in class_dets:
                    x1, y1, x2, y2 = d.bbox_xyxy
                    ix1 = int(max(0, min(w - 1, round(float(x1) / float(YOLO_IMG_SIZE) * w))))
                    iy1 = int(max(0, min(h - 1, round(float(y1) / float(YOLO_IMG_SIZE) * h))))
                    ix2 = int(max(0, min(w, round(float(x2) / float(YOLO_IMG_SIZE) * w))))
                    iy2 = int(max(0, min(h, round(float(y2) / float(YOLO_IMG_SIZE) * h))))
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                    class_box_map[iy1:iy2, ix1:ix2] += float(d.confidence) * class_weight
                    class_mask[iy1:iy2, ix1:ix2] = 1.0

                class_box_map = cv2.GaussianBlur(class_box_map, (0, 0), sigmaX=3.0, sigmaY=3.0)
                bmin = float(class_box_map.min())
                bmax = float(class_box_map.max())
                if bmax > bmin:
                    class_box_map = (class_box_map - bmin) / (bmax - bmin + 1e-8)

                hard_mask = (class_mask > 0).astype(np.uint8)

                class_mask = cv2.GaussianBlur(class_mask, (0, 0), sigmaX=7.0, sigmaY=7.0)
                mmax = float(class_mask.max())
                if mmax > 0:
                    class_mask = class_mask / mmax

                cam_in = cam_map.astype(np.float32) * class_mask
                cmin = float(cam_in.min())
                cmax = float(cam_in.max())
                if cmax > cmin:
                    cam_in = (cam_in - cmin) / (cmax - cmin + 1e-8)

                combined = np.clip(0.75 * cam_in + 0.25 * class_box_map.astype(np.float32), 0.0, 1.0)

                base_bgr = cv2.cvtColor((np.clip(resized_rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                attn_u8 = (combined * 255.0).astype(np.uint8)
                heat = cv2.applyColorMap(attn_u8, cv2.COLORMAP_JET)

                panel_bgr = base_bgr.copy()
                mask_bool = hard_mask.astype(bool)
                if mask_bool.any():
                    blended = cv2.addWeighted(base_bgr, 0.65, heat, 0.35, 0.0)
                    panel_bgr[mask_bool] = blended[mask_bool]

                for d in class_dets:
                    x1, y1, x2, y2 = d.bbox_xyxy
                    ix1 = int(max(0, min(w - 1, round(float(x1) / float(YOLO_IMG_SIZE) * w))))
                    iy1 = int(max(0, min(h - 1, round(float(y1) / float(YOLO_IMG_SIZE) * h))))
                    ix2 = int(max(0, min(w, round(float(x2) / float(YOLO_IMG_SIZE) * w))))
                    iy2 = int(max(0, min(h, round(float(y2) / float(YOLO_IMG_SIZE) * h))))
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                    cv2.rectangle(panel_bgr, (ix1, iy1), (ix2, iy2), color, 1)

                cv2.putText(
                    panel_bgr,
                    f"{title} (n={len(class_dets)})",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                return panel_bgr

            panel_normal = _per_class_panel("Normal", "NORMAL", (120, 220, 120))
            panel_corrected = _per_class_panel("Corrected", "CORRECTED", (0, 165, 255))
            panel_reversal = _per_class_panel("Reversal", "REVERSAL", (60, 60, 255))

            out_bgr = cv2.hconcat([panel_normal, panel_corrected, panel_reversal])
        else:
            out_bgr = overlay_bgr
        cv2.imwrite(str(heatmap_path), out_bgr)
        return heatmap_path
    except (RuntimeError, ValueError, StopIteration, AttributeError, TypeError) as exc:
        raise RuntimeError("Grad-CAM heatmap generation failed") from exc

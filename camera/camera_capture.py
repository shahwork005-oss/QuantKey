#!/usr/bin/env python3
"""
Stage 6: Full Camera Capture Pipeline
CorridorKey Edge Deployment - Camera camera_capture.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

from infer_pi import load_int8_model, infer


def capture_frame(cap: cv2.VideoCapture, model_input_size: int = 512) -> Image.Image:
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture frame from camera")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    pil_image = pil_image.resize((model_input_size, model_input_size), Image.Resampling.LANCZOS)
    return pil_image


def process_frame(
    session,
    pil_image: PILImage,
    original_frame: np.ndarray,
    model_input_size: int = 512,
) -> np.ndarray:
    result = infer(session, pil_image, target_size=model_input_size)

    result_h, result_w = result.shape[:2]
    original_h, original_w = original_frame.shape[:2]

    result_pil = Image.fromarray((result * 255).astype(np.uint8))
    result_pil = result_pil.resize((original_w, original_h), Image.Resampling.LANCZOS)
    return np.array(result_pil)


def main():
    parser = argparse.ArgumentParser(description="CorridorKey camera capture with real-time inference")
    parser.add_argument("--model", default="models/corridorkey_int8.onnx", help="Path to ONNX model")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--output-dir", default="output", help="Output directory for processed frames")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument(
        "--model-size",
        type=int,
        default=512,
        help="Model input size (width and height)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    session = load_int8_model(args.model)

    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    print("Starting capture loop... Press 'q' to quit")
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            original_h, original_w = frame.shape[:2]
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_image = pil_image.resize((args.model_size, args.model_size), Image.Resampling.LANCZOS)

            result = infer(session, pil_image, target_size=args.model_size)

            result_pil = Image.fromarray((result * 255).astype(np.uint8))
            result_pil = result_pil.resize((original_w, original_h), Image.Resampling.LANCZOS)
            result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

            output_path = os.path.join(args.output_dir, f"frame_{frame_count:06d}.png")
            cv2.imwrite(output_path, result_bgr)

            cv2.imshow("CorridorKey", result_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_count += 1

            elapsed = time.time() - start_time
            if frame_count % 30 == 0:
                print(f"FPS: {frame_count / elapsed:.2f}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        print(f"Processed {frame_count} frames in {elapsed:.2f}s ({frame_count / elapsed:.2f} FPS)")


if __name__ == "__main__":
    main()

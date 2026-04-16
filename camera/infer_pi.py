#!/usr/bin/env python3
"""
Stage 5: Raspberry Pi inference test
CorridorKey Edge Deployment - Camera infer_pi.py
"""

import argparse
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image
from PIL.Image import Image as PILImage


def load_int8_model(model_path: str):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, sess_options, providers=providers)
    return session


def preprocess_image(image: np.ndarray, target_size: int = 512) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = image[:3]
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    image = (image - mean) / std
    return image


def infer(session, image_pil: Image.Image, target_size: int = 512):
    input_blob = session.get_inputs()[0]
    h = int(input_blob.shape[2])
    w = int(input_blob.shape[3])

    image_resized = image_pil.resize((w, h), Image.Resampling.LANCZOS)
    image_input = preprocess_image(np.array(image_resized), target_size)
    image_input = np.expand_dims(image_input, axis=0)

    output = session.run(None, {"input": image_input})
    return output[0][0]


def main():
    parser = argparse.ArgumentParser(description="CorridorKey raspberry pi inference test")
    parser.add_argument("--model", default="models/corridorkey_int8.onnx", help="Path to ONNX model")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default="output.png", help="Path to output image")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    session = load_int8_model(args.model)

    image = Image.open(args.image).convert("RGB")

    print("Running inference...")
    result = infer(session, image)

    result_image = Image.fromarray((result * 255).astype(np.uint8))
    result_image.save(args.output)
    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()

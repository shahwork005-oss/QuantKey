# CorridorKey Edge Deployment Guide

This document covers deploying CorridorKey on edge devices (Raspberry Pi, NVIDIA Jetson, etc.) using quantized INT8 models.

## Project Structure

```
CorridorKey/
├── CorridorKeyModule/       # Upstream module (untouched)
├── models/
│   ├── CorridorKey.pth           # Original PyTorch weights
│   ├── corridorkey_fp32_simplified.onnx  # FP32 ONNX export
│   └── corridorkey_int8.onnx     # INT8 quantized model
├── quantize/
│   ├── export_onnx.py           # Stage 3: ONNX export
│   ├── calibrate_int8.py        # Stage 4: INT8 calibration
├── camera/
│   ├── infer_pi.py              # Stage 5: RPi inference test
│   ├── camera_capture.py        # Stage 6: Full camera pipeline
├── calibration_frames/        # Training data for calibration
└── EDGE_DEPLOY.md              # This document
```

## Model Files

| File | Description | Use Case |
|------|--------------|----------|
| `CorridorKey.pth` | Original PyTorch weights | Training/FP32 export |
| `corridorkey_fp32_simplified.onnx` | FP32 ONNX | Validation/comparison |
| `corridorkey_int8.onnx` | INT8 quantized | Edge deployment |

## Quantization Process

### Stage 3: Export ONNX (FP32)

```bash
python quantize/export_onnx.py --model models/CorridorKey.pth --output models/corridorkey_fp32_simplified.onnx
```

### Stage 4: INT8 Calibration

```bash
python quantize/calibrate_int8.py --model models/corridorkey_fp32_simplified.onnx --output models/corridorkey_int8.onnx --frames calibration_frames/
```

### Stage 5: Raspberry Pi Inference Test

```bash
python camera/infer_pi.py --model models/corridorkey_int8.onnx --image test.jpg --output result.png
```

### Stage 6: Full Camera Pipeline

```bash
python camera/camera_capture.py --model models/corridorkey_int8.onnx --output-dir output/
```

## Requirements

- Python 3.10+
- onnxruntime (with INT8 support)
- OpenCV (opencv-python-headless)
- NumPy
- Pillow

## Calibration Frames

100-200 green screen shots are required for INT8 calibration. These frames should:
- Use a consistent chroma key background (green screen)
- Include various keying scenarios (hair, transparency, edges)
- Cover expected lighting conditions

**Note:** Calibration frames are NOT committed to git (see `.gitignore`).

## Performance Notes

- INT8 provides ~3-4x speedup over FP32 on ARM CPUs
- Memory footprint: ~74MB for INT8 vs ~289MB for FP32
- Raspberry Pi 4: ~5-10 FPS achievable with INT8

## Troubleshooting

### Low FPS on Raspberry Pi
- Use INT8 model (not FP32)
- Limit to 4 threads: `sess_options.intra_op_num_threads = 4`
- Consider smaller input resolution (384x384)

### Out of Memory
- Close other applications
- Use smaller batch size
- Reduce camera resolution

## License

See LICENSE file in the CorridorKey root directory.
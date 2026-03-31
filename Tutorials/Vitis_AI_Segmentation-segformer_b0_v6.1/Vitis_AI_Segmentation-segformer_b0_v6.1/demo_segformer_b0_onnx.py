#Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#SPDX-License-Identifier: MIT

import os
import cv2
import numpy as np
import onnxruntime as ort
import argparse

# --------------------------------------------------
# Arguments
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", required=True)
parser.add_argument("--onnx_model", required=True)
parser.add_argument("--output_path", default="result.png")
parser.add_argument("--target", default="cpu", choices=["cpu", "npu"])
parser.add_argument("--cache_key", default="segformer_cache")
args = parser.parse_args()

# --------------------------------------------------
# Create Session
# --------------------------------------------------
if args.target == "cpu":
    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(args.onnx_model, providers=providers)

else:
    provider_options = {
        "config_file": "vitisai_config.json",
        "cache_dir": "./",
        "cache_key": args.cache_key,
        "ai_analyzer_visualization": True,
        "ai_analyzer_profiling": True,
        "log_level": "info",
        "target": "VAIML",
    }

    sess = ort.InferenceSession(
        args.onnx_model,
        providers=["VitisAIExecutionProvider"],
        provider_options=[provider_options]
    )

# --------------------------------------------------
# Model Info (AUTO)
# --------------------------------------------------
input_info = sess.get_inputs()[0]
output_info = sess.get_outputs()[0]

input_name = input_info.name
input_shape = input_info.shape
input_type = input_info.type

print("==== Model Info ====")
print("Input name :", input_name)
print("Input shape:", input_shape)
print("Input type :", input_type)
print("Output shape:", output_info.shape)
print()

# Detect layout
if input_shape[1] == 3:
    layout = "NCHW"
    H, W = input_shape[2], input_shape[3]
else:
    layout = "NHWC"
    H, W = input_shape[1], input_shape[2]

print("Layout:", layout)
print("Model size:", H, W)

# Detect dtype
if "int8" in input_type:
    model_dtype = np.int8
elif "float16" in input_type or "fp16" in input_type:
    model_dtype = np.float16
else:
    model_dtype = np.float32

print("Model dtype:", model_dtype)
print()

# --------------------------------------------------
# Preprocess
# --------------------------------------------------
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

img = cv2.imread(args.image_path)
if img is None:
    raise FileNotFoundError(args.image_path)

orig_h, orig_w = img.shape[:2]

# Resize to model size
img_resized = cv2.resize(img, (W, H))

# BGR -> RGB
img_rgb = img_resized[:, :, ::-1] / 255.0

# Normalize
img_norm = (img_rgb - mean) / std

# Layout transform
if layout == "NCHW":
    input_tensor = img_norm.transpose(2, 0, 1)[None]
else:
    input_tensor = img_norm[None]

# If INT8 model (common for Vitis)
if model_dtype == np.int8:
    input_tensor = np.clip(input_tensor * 127, -128, 127).astype(np.int8)
elif model_dtype == np.float16:
    input_tensor = input_tensor.astype(np.float16)
else:  # float32
    input_tensor = input_tensor.astype(np.float32)

# --------------------------------------------------
# Inference
# --------------------------------------------------
outputs = sess.run(None, {input_name: input_tensor})[0]

print("Output shape:", outputs.shape)

# --------------------------------------------------
# Decode Output (AUTO layout)
# --------------------------------------------------
num_classes = output_info.shape[1] if layout == "NCHW" else output_info.shape[-1]

if outputs.shape[1] == num_classes:   # NCHW
    pred = np.argmax(outputs[0], axis=0)
else:                                 # NHWC
    pred = np.argmax(outputs[0], axis=-1)

# Resize back
pred = cv2.resize(pred.astype(np.uint8),
                  (orig_w, orig_h),
                  interpolation=cv2.INTER_NEAREST)

# --------------------------------------------------
# Cityscapes Color Map
# --------------------------------------------------
colors = np.array([
    [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156], [190,153,153],
    [153,153,153], [250,170, 30], [220,220, 0], [107,142, 35], [152,251,152],
    [ 70,130,180], [220, 20, 60], [255,  0,  0], [  0,  0,142], [  0,  0, 70],
    [  0, 60,100], [  0, 80,100], [  0,  0,230], [119, 11, 32]
], dtype=np.uint8)

color_mask = colors[pred]

overlay = cv2.addWeighted(img, 0.5, color_mask, 0.5, 0)

os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
cv2.imwrite(args.output_path, overlay)

print("Saved:", args.output_path)



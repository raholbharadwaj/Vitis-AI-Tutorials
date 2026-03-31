#Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#SPDX-License-Identifier: MIT

from transformers import SegformerForSemanticSegmentation
import torch

model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

model = SegformerForSemanticSegmentation.from_pretrained(model_name)
model.eval()

dummy = torch.randn(1, 3, 256, 512)

torch.onnx.export(
    model,
    dummy,
    "segformer_b0_cityscapes_256x512.onnx",
    input_names=["input"],
    output_names=["logits"],
    opset_version=20,
    do_constant_folding=True
)

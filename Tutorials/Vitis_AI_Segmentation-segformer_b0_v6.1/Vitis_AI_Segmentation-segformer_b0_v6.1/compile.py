#Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#SPDX-License-Identifier: MIT

import onnxruntime

provider_options_dict = {
    "config_file": 'vitisai_config.json',
    "cache_dir":   './',
    "cache_key":   'segformer_b0_cityscapes_256x512',
    'ai_analyzer_visualization': True,
    'ai_analyzer_profiling': True,
    "log_level": 'info',
    "target": 'VAIML',
}
   
print(f"Creating ORT inference session")
session = onnxruntime.InferenceSession(
    'segformer_b0_cityscapes_256x512.onnx',
    providers=["VitisAIExecutionProvider"],
    provider_options=[provider_options_dict]
)   

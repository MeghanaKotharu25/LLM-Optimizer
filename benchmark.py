import time
import torch
import torch.nn as nn
import os
import onnxruntime as ort
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
# ... (Keep existing imports)

class UniversalModel:
    def __init__(self, model_or_path, is_onnx=False, use_cache=True, batch_size=1):
        self.is_onnx = is_onnx
        self.use_cache = use_cache
        self.batch_size = batch_size
        
        if is_onnx:
            self.session = ort.InferenceSession(model_or_path, providers=['CPUExecutionProvider'])
        else:
            self.model = model_or_path
            # Systems Knob: KV Cache
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = self.use_cache
            self.model.eval()

    def __call__(self, input_ids, attention_mask=None):
        # Support Batching by repeating inputs if needed, or simply pass through
        if self.is_onnx:
            # ... (Existing ONNX logic) ...
            return torch.from_numpy(outputs[0])
        else:
            with torch.no_grad():
                # Systems Knob: Batching handled by PyTorch natively
                out = self.model(input_ids, attention_mask=attention_mask)
                return out.logits
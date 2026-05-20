import time
import torch
import torch.nn as nn
import os
import onnxruntime as ort
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

class UniversalModel:
    """Wrapper to make ONNX and PyTorch models act the same way."""
    def __init__(self, model_or_path, is_onnx=False):
        self.is_onnx = is_onnx
        if is_onnx:
            # Use CPU for benchmarking to ensure consistency on Mac M1
            self.session = ort.InferenceSession(model_or_path, providers=['CPUExecutionProvider'])
        else:
            self.model = model_or_path
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False
            self.model.eval()

    def __call__(self, input_ids, attention_mask=None):
        if self.is_onnx:
            # If no mask provided, create a default mask of all 1s
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            
            inputs = {
                'input_ids': input_ids.cpu().numpy().astype(np.int64),
                'attention_mask': attention_mask.cpu().numpy().astype(np.int64)
            }
            outputs = self.session.run(None, inputs)
            return torch.from_numpy(outputs[0]) # Returns Logits as Tensor
        else:
            with torch.no_grad():
                out = self.model(input_ids, attention_mask=attention_mask)
                return out.logits if hasattr(out, "logits") else out

def measure_latency(u_model, tokenizer, prompt="AI is the future", runs=10):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Warmup
    _ = u_model(input_ids, attention_mask=attention_mask)

    start = time.perf_counter()
    for _ in range(runs):
        _ = u_model(input_ids, attention_mask=attention_mask)
    end = time.perf_counter()
    
    return (end - start) / runs

def calculate_perplexity(u_model, tokenizer, num_samples=20):
    """Calculates perplexity by manually computing CrossEntropy from logits."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"][:num_samples])
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = encodings.input_ids
    
    stride = 512
    nlls = []
    loss_fct = nn.CrossEntropyLoss(reduction="none")

    for i in tqdm(range(0, input_ids.size(1), stride), desc="PPL Evaluation"):
        begin_loc = max(i + stride - 1024, 0)
        end_loc = min(i + stride, input_ids.size(1))
        
        input_chunk = input_ids[:, begin_loc:end_loc]
        # For evaluation on sequences, we use an all-ones mask for the chunk
        attn_mask = torch.ones_like(input_chunk)
        target_ids = input_chunk.clone()
        
        logits = u_model(input_chunk, attention_mask=attn_mask)
        
        # Shift for causal LM (predict next token)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss.mean())

    return torch.exp(torch.stack(nlls).mean()).item()

def measure_size(path_or_model):
    if isinstance(path_or_model, str):
        if os.path.exists(path_or_model):
            return os.path.getsize(path_or_model) / (1024 * 1024)
        return 0
    else:
        param_size = 0
        for param in path_or_model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size / (1024 * 1024)
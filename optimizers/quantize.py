import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def apply_onnx_quantization(model, tokenizer):
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapped_model = Wrapper(model)
    onnx_path = "model.onnx"
    dummy = tokenizer("The quick brown fox", return_tensors="pt")
    
    # 1. Export
    torch.onnx.export(
        wrapped_model, (dummy["input_ids"], dummy["attention_mask"]),
        onnx_path, input_names=["input_ids", "attention_mask"],
        output_names=["logits"], 
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"}
        },
        opset_version=17
    )

    # 2. Dynamic Quantization (The "Safe" Way)
    quantized_path = "model_quant.onnx"
    print("   -> Running Robust Dynamic Quantization...")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QUInt8,
        # We avoid the 'CalibrationDataReader' because dynamic 
        # quantization handles LayerNorms on-the-fly.
    )
    return quantized_path
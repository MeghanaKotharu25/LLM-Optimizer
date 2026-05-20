import copy
import pandas as pd
import os
from benchmark import UniversalModel, measure_latency, measure_size, calculate_perplexity
from optimizers.prune import apply_pruning
from optimizers.quantize import apply_onnx_quantization

class ModelOptimizer:
    def __init__(self, base_model, tokenizer, output_file="results.csv"):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.output_file = output_file
        self.results = []

    def evaluate(self, u_model, name, model_path=None):
        stats = {
            "Model": name,
            "Latency(s)": round(measure_latency(u_model, self.tokenizer), 4),
            "Size(MB)": round(measure_size(model_path if u_model.is_onnx else u_model.model), 2),
            "Perplexity": round(calculate_perplexity(u_model, self.tokenizer), 2)
        }
        self.results.append(stats)
        # Save to CSV immediately so you don't lose data if it crashes
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_file, index=False)
        return stats

    def run_full_benchmark(self):
        print("--- Running Full Benchmark Suite ---")
        
        # 1. Base
        u_base = UniversalModel(self.base_model, is_onnx=False)
        self.evaluate(u_base, "Base")

        # 2. Pruning
        for amount in [0.1, 0.2]:
            print(f"Testing Pruning {int(amount*100)}%...")
            pruned_model = apply_pruning(copy.deepcopy(self.base_model), amount=amount)
            u_pruned = UniversalModel(pruned_model, is_onnx=False)
            self.evaluate(u_pruned, f"Pruned-{int(amount*100)}%")

        # 3. Quantization
        print("Testing Quantization...")
        quant_path = apply_onnx_quantization(self.base_model, self.tokenizer)
        u_quant = UniversalModel(quant_path, is_onnx=True)
        self.evaluate(u_quant, "Quantized-ONNX", model_path=quant_path)

        print(f"\n✅ All results saved to {self.output_file}")
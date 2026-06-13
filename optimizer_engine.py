import copy
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from benchmark import UniversalModel, measure_latency, measure_size, calculate_perplexity
from optimizers.prune import apply_pruning
from optimizers.quantize import apply_onnx_quantization
from config_space import get_optimization_grid

class ModelOptimizer:
    def __init__(self, output_file="results.csv"):
        self.output_file = output_file
        self.results = []

    def run_full_benchmark(self):
        grid = get_optimization_grid()
        for config in grid:
            name = f"{config['model']}-Prune{int(config['prune']*100)}-Quant{config['quantize']}"
            print(f"\n🚀 Benchmarking: {name}")
            
            tokenizer = AutoTokenizer.from_pretrained(config['model'])
            model = AutoModelForCausalLM.from_pretrained(config['model'])
            
            # Apply Optimizations
            if config['prune'] > 0:
                model = apply_pruning(model, amount=config['prune'])
            
            if config['quantize']:
                path = apply_onnx_quantization(model, tokenizer)
                u_model = UniversalModel(path, is_onnx=True)
                size = measure_size(path)
            else:
                u_model = UniversalModel(model, is_onnx=False)
                size = measure_size(model)

            # Measure
            self.results.append({
                "Model": name,
                "Latency(s)": round(measure_latency(u_model, tokenizer), 4),
                "Size(MB)": round(size, 2),
                "Perplexity": round(calculate_perplexity(u_model, tokenizer), 2)
            })
            pd.DataFrame(self.results).to_csv(self.output_file, index=False)
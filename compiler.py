import pandas as pd
import os
import psutil

class LLMCompiler:
    def __init__(self, results_file="results.csv"):
        if not os.path.exists(results_file):
            raise FileNotFoundError("Run benchmark_2.py first!")
        self.df = pd.read_csv(results_file)

    def get_device_profile(self):
        """Profiles the current machine to inform compilation decisions."""
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_count()
        
        # Heuristics for "Device Capability"
        profile = {
            "ram_gb": mem.available / (1024**3),
            "cpu_count": cpu,
            "is_low_end": (cpu < 4) or (mem.available < (2 * 1024**3))
        }
        print(f"🖥️  System Profile: {profile['cpu_count']} Cores, {profile['ram_gb']:.2f}GB RAM Available")
        return profile

    def compile(self, max_latency, weights={'latency': 0.6, 'ppl': 0.4}):
        profile = self.get_device_profile()
        candidates = self.df.copy()

        # 1. Hardware-Aware Filtering (The "Insane" logic)
        if profile['is_low_end']:
            print("⚠️ Low-end hardware detected. Forcing Quantized models only.")
            # Blacklist "Base" models if they contain "Base" in the name
            candidates = candidates[~candidates['Model'].str.contains("Base")]
        
        if profile['ram_gb'] < 2.0:
            # Drop models larger than 400MB if RAM is scarce
            candidates = candidates[candidates['Size(MB)'] < 400]

        # 2. Performance Constraint
        candidates = candidates[candidates['Latency(s)'] <= max_latency]
        
        if candidates.empty:
            return None, "No models meet these aggressive hardware/latency constraints."

        # 3. Intelligent Selection via Scoring
        # Normalize (0 to 1, where 1 is best)
        candidates['norm_lat'] = 1 - (candidates['Latency(s)'] / candidates['Latency(s)'].max())
        candidates['norm_ppl'] = 1 - (candidates['Perplexity'] / candidates['Perplexity'].max())
        
        candidates['score'] = (weights['latency'] * candidates['norm_lat']) + \
                             (weights['ppl'] * candidates['norm_ppl'])
        
        best_choice = candidates.sort_values('score', ascending=False).iloc[0]
        return best_choice, "Success"
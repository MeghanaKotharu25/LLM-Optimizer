import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from optimizers.prune import apply_pruning
from compiler import LLMCompiler
from benchmark import UniversalModel

class OptimizedRuntime:
    def __init__(self, target_latency=0.03):
        self.compiler = LLMCompiler()
        print("🚀 Calibrating environment...")
        
        # 1. Ask compiler for best config
        self.config, _ = self.compiler.compile(target_latency)
        print(f"✅ Ready. Selected: {self.config['Model']}")
        
        # 2. Extract model name (e.g., 'gpt2' or 'distilgpt2')
        model_name = self.config['Model'].split('-')[0]
        
        # 3. Load Base Model
        raw_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 4. Apply Optimization if chosen
        if "Prune" in self.config['Model']:
            # Extract ratio from string (e.g., 'Prune10' -> 0.1)
            # Find the number after 'Prune'
            import re
            match = re.search(r'Prune(\d+)', self.config['Model'])
            amount = int(match.group(1)) / 100 if match else 0.1
            raw_model = apply_pruning(raw_model, amount=amount)
        
        # 5. Initialize Universal Engine
        self.engine = UniversalModel(model_or_path=raw_model, is_onnx=False, use_cache=True)

    def generate(self, text, max_new_tokens=20, batch_size=1):
        # 1. Use Tokenizer padding for batching
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer([text] * batch_size, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # 2. KV Cache is currently 'True' in engine init
        # 3. Generate Loop
        for _ in range(max_new_tokens):
            with torch.no_grad():
                # Pass the mask through to engine
                logits = self.engine(input_ids, attention_mask=attention_mask)
                
                # Take last token for each item in batch
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                # Update mask
                attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1))], dim=-1)
                
        # Decode first item of batch
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
if __name__ == "__main__":
    runtime = OptimizedRuntime(target_latency=0.03)
    print(runtime.generate("The future of artificial intelligence is"))
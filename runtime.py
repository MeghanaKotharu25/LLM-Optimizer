from compiler import LLMCompiler
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from optimizers.prune import apply_pruning
import torch

class OptimizedRuntime:
    def __init__(self, target_latency=0.03):
        # 1. Ask the compiler for the best config
        compiler = LLMCompiler()
        self.config, status = compiler.compile(max_latency=target_latency)
        
        if not self.config:
            raise RuntimeError(f"Compiler failed: {status}")
            
        print(f"🚀 Initializing production engine with: {self.config['Model']}")
        
        # 2. Setup
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # 3. Apply the specific optimization chosen by the compiler
        if "Pruned" in self.config['Model']:
            ratio = 0.2 if "20%" in self.config['Model'] else 0.1
            self.model = apply_pruning(self.model, amount=ratio)
            
        self.model.eval()

    def generate(self, text, max_new_tokens=20):
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if next_token.item() == self.tokenizer.eos_token_id: break
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    runtime = OptimizedRuntime(target_latency=0.03)
    print(runtime.generate("The future of artificial intelligence is"))
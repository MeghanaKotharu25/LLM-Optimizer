import torch
import pandas as pd
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from optimizers.prune import apply_pruning

class OptimizedRuntime:
    def __init__(self, model_name="gpt2", results_file="results.csv"):
        self.device = torch.device("cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load winner
        df = pd.read_csv(results_file)
        winner = df[df['Perplexity'] < 25].sort_values('Latency(s)').iloc[0]
        
        print(f"🚀 Initializing production engine with: {winner['Model']}")
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        if "Pruned" in winner['Model']:
            amount = 0.2 if "20%" in winner['Model'] else 0.1
            self.model = apply_pruning(self.model, amount=amount)
        
        self.model.eval()

    def generate(self, text, max_new_tokens=20):
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                # Direct logit output
                outputs = self.model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Simple Greedy Decoding
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                    
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    runtime = OptimizedRuntime()
    print("\n--- Generated Output ---")
    print(runtime.generate("The future of artificial intelligence is"))
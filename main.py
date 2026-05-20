from transformers import GPT2LMHeadModel, GPT2Tokenizer
from optimizer_engine import ModelOptimizer

def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    optimizer = ModelOptimizer(model, tokenizer)
    optimizer.run_full_benchmark()

if __name__ == "__main__":
    main()
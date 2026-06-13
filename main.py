from optimizer_engine import ModelOptimizer
from compiler import LLMCompiler

def main():
    # 1. Run the Benchmarking Engine
    optimizer = ModelOptimizer() 
    optimizer.run_full_benchmark()

    # 2. Run the Intelligent Compiler
    compiler = LLMCompiler()
    
    # Compiler selects the best config based on hardware + latency
    best, status = compiler.compile(max_latency=0.015, weights={'latency': 0.9, 'ppl': 0.1})
    
    if best is not None:
        print(f"\n🚀 Compiler Recommendation: {best['Model']}")
        print(f"Metrics: Latency={best['Latency(s)']}s, PPL={best['Perplexity']}")
    else:
        print(f"Error: {status}")

if __name__ == "__main__":
    main()
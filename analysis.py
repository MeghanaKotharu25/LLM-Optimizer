import pandas as pd

def run_analysis():
    df = pd.read_csv("results.csv")
    # OES = 1 / (PPL * Latency)
    df['OES'] = 1 / (df['Perplexity'] * df['Latency(s)'])
    
    print("\n--- RESEARCH FINDINGS ---")
    
    # Baseline: No pruning, no quantization
    baseline = df[(df['Model'].str.contains("Prune0-QuantFalse")) & (df['Model'].str.contains("gpt2"))].iloc[0]
    
    # Smartest choice in the data
    smart = df.sort_values('OES', ascending=False).iloc[0]
    
    print(f"Baseline OES: {baseline['OES']:.2f}")
    print(f"Smart Compiler OES: {smart['OES']:.2f}")
    print(f"\n✅ Efficiency Improvement: {((smart['OES']/baseline['OES'])-1)*100:.1f}%")

if __name__ == "__main__":
    run_analysis()
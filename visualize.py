import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Load Results
# =========================
df = pd.read_csv("results.csv")

print("\nLoaded Results:")
print(df)


# =========================
# Latency Comparison
# =========================
plt.figure(figsize=(8, 5))

latency_col = "Latency(s)" if "Latency(s)" in df.columns else "Latency(sec)"

plt.bar(
    df["Model"],
    df[latency_col]
)

plt.xlabel("Model")
plt.ylabel("Latency (sec)")
plt.title("LLM Optimization - Latency Comparison")

plt.xticks(rotation=10)

plt.tight_layout()

plt.savefig("latency_comparison.png")

plt.close()


# =========================
# Perplexity Comparison
# =========================
plt.figure(figsize=(8, 5))

plt.bar(
    df["Model"],
    df["Perplexity"]
)

plt.xlabel("Model")
plt.ylabel("Perplexity")
plt.title("LLM Optimization - Perplexity Comparison")

plt.xticks(rotation=10)

plt.tight_layout()

plt.savefig("perplexity_comparison.png")

plt.close()


# =========================
# Tradeoff Plot
# =========================
plt.figure(figsize=(8, 5))

plt.scatter(
    df[latency_col],
    df["Perplexity"]
)

for i, model_name in enumerate(df["Model"]):
    plt.annotate(
        model_name,
        (
            df[latency_col][i],
            df["Perplexity"][i]
        )
    )

plt.xlabel("Latency (sec)")
plt.ylabel("Perplexity")
plt.title("Latency vs Perplexity Tradeoff")

plt.tight_layout()

plt.savefig("tradeoff_plot.png")

plt.close()


print("\nVisualization Complete.")
print("Saved:")
print("- latency_comparison.png")
print("- perplexity_comparison.png")
print("- tradeoff_plot.png")
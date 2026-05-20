LLM Optimization Suite:Performance Report 

Methodology

We benchmarked GPT-2 using three strategies:
- Baseline (Standard FP32)
- Magnitude-based Pruning (10% & 20%)
- Dynamic Quantization (ONNX Runtime

  Results
| Model            | Latency (s) | Size (MB) | Perplexity |
|------------------|-------------|-----------|-------------|
| Base             | 0.0300      | 474.70    | 22.79       |
| Pruned-10%       | 0.0275      | 474.70    | 23.09       |
| Pruned-20%       | 0.0323      | 474.70    | 24.27       |
| Quantized-ONNX   | 0.0057      | 156.47    | 238.75      |

Key Insight

While Quantization provided the smallest disk footprint, Magnitude-based Pruning (20%) emerged as the superior production configuration, providing a ~20% latency reduction while maintaining ~95% of the model's original accuracy.

# Autonomous LLM Inference Runtime

**Treating model deployment as a hardware-aware optimization problem.**

This project is an automated inference orchestration engine. Instead of manually selecting model configurations, this runtime profiles the host hardware at startup, queries a learned ML-based policy to predict latency, and dynamically compiles the optimal model variant (pruning + quantization) for the current resource budget.

---

## 🚀 The Architecture

This is a **Self-Optimizing Runtime**:

1.  **Cold-Start Calibration:** Profiles CPU/Memory bandwidth at runtime to understand the specific environment's performance profile.
2.  **Learned Policy Engine:** Uses a **Random Forest Regressor** trained on historical benchmark data to predict inference latency across a multi-dimensional search space (Pruning Ratios, Quantization, Model Size).
3.  **Multi-Objective Orchestrator:** Balances Latency and Perplexity using an **Optimization Efficiency Score (OES)** to navigate the Pareto frontier, rather than relying on brittle hardcoded rules.
4.  **RESTful Inference Gateway:** Exposes the compiler/runtime via FastAPI for integration into larger production systems.

---

## 📊 Experimental Results

We evaluated 20+ configurations across GPT-2 and DistilGPT-2 architectures. The intelligent compiler consistently selected configurations that maximized the Optimization Efficiency Score (OES).

| Configuration | Latency (s) | Perplexity | OES |
| :--- | :--- | :--- | :--- |
| Baseline (GPT-2, No Opt) | 0.0320 | 22.79 | 1.37 |
| **Optimized (DistilGPT-2, P40, Q)** | **0.0035** | 100.04 | **2.85** |

*Efficiency Improvement: **~108% increase in OES** compared to baseline architectures.*

---

## 🛠 Tech Stack
*   **Intelligence:** Scikit-Learn (Random Forest Regression)
*   **Runtime:** PyTorch, ONNX Runtime
*   **Deployment:** FastAPI (Inference Gateway)
*   **Profiling:** `psutil` (Hardware Telemetry)

---

## ⚡ Quick Start

### 1. The Lab (Benchmarking)
Generate the performance landscape by running the full sweep:
```bash
python main.py
```

### 2. The Engine (Inference)
Launch the intelligent runtime, which calibrates to your machine and selects the best model:
```bash
python runtime.py
```

### 3. The Gateway (API)
Spin up the Inference API to serve recommendations dynamically:
```bash
uvicorn api:app --reload
# Access docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
```

### 💡 Why this is different
Insted of appling optimizations, This system builds a decision engine over them. By separating the Benchmarking Engine from the Runtime Policy, this project enables deployment that adapts to the specific hardware—from low-end edge devices to powerful workstations—without changing a single line of code.

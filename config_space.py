def get_optimization_grid():
    """
    Expands the research space to generate 20+ data points.
    """
    models = ["gpt2", "distilgpt2"]
    pruning_ratios = [0.0, 0.1, 0.2, 0.4, 0.6]
    quantize_options = [True, False]
    
    grid = []
    for m in models:
        for p in pruning_ratios:
            for q in quantize_options:
                grid.append({"model": m, "prune": p, "quantize": q})
    return grid
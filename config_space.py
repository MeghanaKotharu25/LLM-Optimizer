# config_space.py
def get_optimization_grid():
    """
    Defines the search space. 
    Easy to add new models or strategies here.
    """
    return [
        {"model": "gpt2", "prune": 0.0, "quantize": False},
        {"model": "gpt2", "prune": 0.1, "quantize": False},
        {"model": "gpt2", "prune": 0.0, "quantize": True},
        {"model": "distilgpt2", "prune": 0.0, "quantize": False},
        {"model": "distilgpt2", "prune": 0.0, "quantize": True},
    ]
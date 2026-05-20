import torch.nn.utils.prune as prune

def apply_pruning(model, amount=0.1):
    """
    Applies magnitude-based pruning specifically to the MLP layers.
    Renamed to apply_pruning to match main.py orchestration.
    """
    for name, module in model.named_modules():
        # We target the dense layers (MLP) as they are the most redundant.
        # This is safer than pruning the Attention layers.
        if "mlp.c_fc" in name or "mlp.c_proj" in name:
            prune.l1_unstructured(module, name="weight", amount=amount)
            # This makes the pruning permanent so it's ready for benchmarking
            prune.remove(module, 'weight')
            
    return model
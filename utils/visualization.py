import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.metrics import count_parameters, measure_model_memory, measure_inference_time

def plot_weight_distribution(model, layer_name=None):
    """
    Plot the weight distribution of a specific layer or all layers
    
    Args:
        model: PyTorch model
        layer_name: Optional name of specific layer to plot
    """
    plt.figure(figsize=(12, 8))
    
    # Collect weights
    weights_dict = {}
    for name, param in model.named_parameters():
        if 'weight' in name and 'bn' not in name:
            if layer_name is None or layer_name in name:
                weights_dict[name] = param.detach().cpu().numpy().flatten()
    
    # Plot distributions
    num_plots = len(weights_dict)
    if num_plots == 0:
        print("No weights found with the specified layer name.")
        return
    
    for i, (name, weights) in enumerate(weights_dict.items()):
        plt.subplot(num_plots, 1, i+1)
        plt.hist(weights, bins=100)
        plt.title(f'Weight Distribution: {name}')
        plt.xlabel('Weight Value')
        plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()

def visualize_ternary_weights(model, layer_name):
    """
    Visualize the distribution of ternary weights (-1, 0, +1)
    
    Args:
        model: PyTorch model
        layer_name: Name of layer to visualize
    """
    # Find the specified layer's weights
    weight = None
    for name, param in model.named_parameters():
        if layer_name in name and 'weight' in name:
            weight = param.detach().cpu()
            break
    
    if weight is None:
        print(f"Layer {layer_name} not found.")
        return
    
    # Get ternary weights
    threshold = 0.05 * torch.mean(torch.abs(weight))
    ternary = torch.zeros_like(weight)
    ternary[weight > threshold] = 1
    ternary[weight < -threshold] = -1
    
    # Count values
    neg_count = torch.sum(ternary == -1).item()
    zero_count = torch.sum(ternary == 0).item()
    pos_count = torch.sum(ternary == 1).item()
    total = ternary.numel()
    
    # Create pie chart
    plt.figure(figsize=(10, 6))
    plt.pie(
        [neg_count, zero_count, pos_count],
        labels=['-1', '0', '+1'],
        autopct='%1.1f%%',
        colors=['red', 'gray', 'blue']
    )
    plt.title(f'Ternary Weight Distribution for {layer_name}')
    
    # Print stats
    print(f"Layer: {layer_name}")
    print(f"Total weights: {total}")
    print(f"-1: {neg_count} ({neg_count/total*100:.2f}%)")
    print(f" 0: {zero_count} ({zero_count/total*100:.2f}%)")
    print(f"+1: {pos_count} ({pos_count/total*100:.2f}%)")
    
    plt.show()

def compare_model_efficiency(models_dict, input_size):
    """
    Compare efficiency metrics between different models
    
    Args:
        models_dict: Dictionary of models {name: model}
        input_size: Input size for models (C, H, W)
    """
    # Metrics to compare
    metrics = {
        'Parameters (M)': lambda m: count_parameters(m) / 1e6,
        'Memory (MB)': measure_model_memory,
        'Inference Time (ms)': lambda m: measure_inference_time(
            m, torch.ones(1, *input_size).to(next(m.parameters()).device)
        )
    }
    
    # Collect results
    results = {name: {} for name in models_dict.keys()}
    
    for name, model in models_dict.items():
        for metric_name, metric_fn in metrics.items():
            results[name][metric_name] = metric_fn(model)
    
    # Plot comparisons
    plt.figure(figsize=(15, 5))
    
    for i, metric_name in enumerate(metrics.keys()):
        plt.subplot(1, 3, i+1)
        values = [results[name][metric_name] for name in models_dict.keys()]
        plt.bar(list(models_dict.keys()), values)
        plt.title(metric_name)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print table
    print("Model Efficiency Comparison:")
    header = "Model".ljust(20) + " | " + " | ".join([m.ljust(15) for m in metrics.keys()])
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for name in models_dict.keys():
        row = name.ljust(20) + " | "
        row += " | ".join([f"{results[name][m]:.4f}".ljust(15) for m in metrics.keys()])
        print(row)
    
    print("-" * len(header))
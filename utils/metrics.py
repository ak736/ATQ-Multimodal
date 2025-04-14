import time
import torch
import numpy as np

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_model_memory(model):
    """
    Measure the memory usage of a model in MB
    """
    # Get model size in bytes
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    # Convert to MB
    return param_size / (1024 * 1024)

def measure_inference_time(model, input_tensor, num_runs=50):
    """
    Measure average inference time over multiple runs
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor for inference
        num_runs: Number of runs to average over
        
    Returns:
        Average inference time in milliseconds
    """
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = model(input_tensor)
        
        # Measure time
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(input_tensor)
        end_time = time.time()
    
    # Calculate average time in ms
    avg_time = (end_time - start_time) * 1000 / num_runs
    return avg_time

def estimate_flops(model, input_size):
    """
    Rough estimate of FLOPs for the model
    This is a simplified version and doesn't account for all operations
    
    Args:
        model: PyTorch model
        input_size: Input size (C, H, W)
        
    Returns:
        Estimated FLOPs
    """
    # This is a very simplified FLOP counter
    flops_dict = {}
    
    def count_conv2d(m, x, y):
        # Count for Conv2D: kernels * input_channels * output_channels * output_height * output_width
        x = x[0]
        batch_size, input_channels, input_height, input_width = x.size()
        output_height, output_width = y.size()[2:]
        
        kernel_ops = m.kernel_size[0] * m.kernel_size[1]
        flops = batch_size * output_height * output_width * input_channels * kernel_ops * m.out_channels
        return flops
    
    def count_linear(m, x, y):
        # Count for Linear: input_features * output_features
        x = x[0]
        batch_size = x.size(0)
        flops = batch_size * m.in_features * m.out_features
        return flops
    
    def add_hooks(m):
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_hook(lambda m, x, y: flops_dict.update({m: count_conv2d(m, x, y)}))
        elif isinstance(m, torch.nn.Linear) or hasattr(m, 'in_features') and hasattr(m, 'out_features'):
            m.register_forward_hook(lambda m, x, y: flops_dict.update({m: count_linear(m, x, y)}))
    
    # Register hooks
    model.apply(add_hooks)
    
    # Run model with dummy input
    device = next(model.parameters()).device
    dummy_input = torch.ones(1, *input_size).to(device)
    _ = model(dummy_input)
    
    # Sum up the flops
    total_flops = sum(flops_dict.values())
    return total_flops

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    
    Args:
        output: Model output (logits)
        target: Ground truth labels
        topk: Tuple of k values to compute accuracy for
        
    Returns:
        Tuple of accuracies for each k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def confusion_matrix(preds, labels, num_classes=None):
    """
    Compute confusion matrix
    
    Args:
        preds: Model predictions (class indices)
        labels: Ground truth labels
        num_classes: Number of classes (if None, determined from inputs)
        
    Returns:
        Confusion matrix as numpy array
    """
    if num_classes is None:
        num_classes = max(torch.max(preds).item(), torch.max(labels).item()) + 1
    
    conf_mat = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    for p, t in zip(preds, labels):
        conf_mat[t, p] += 1
    
    return conf_mat.cpu().numpy()
import os
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from models.image_classifier import ATQImageClassifier
from data.datasets import get_mnist_data, get_fashion_mnist_data
from utils.metrics import measure_model_memory, measure_inference_time, count_parameters
from utils.visualization import plot_weight_distribution, visualize_ternary_weights
from atq.bit_packing import TernaryBitPacking

def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset (use full dataset for better results)
    if args.dataset == 'mnist':
        train_loader, val_loader, test_loader = get_mnist_data(args.batch_size, subset_fraction=1.0)
        input_channels = 1
    elif args.dataset == 'fashion_mnist':
        train_loader, val_loader, test_loader = get_fashion_mnist_data(args.batch_size, subset_fraction=1.0)
        input_channels = 1
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Create enhanced ATQ model with wider layers
    hidden_size = 256 if args.wider_layers else 128  # Default to wider if enabled
    model = ATQImageClassifier(
        num_classes=10,
        input_channels=input_channels,
        use_rpb=args.use_rpb,
        sparsity_target=args.sparsity,
        hidden_size=hidden_size
    ).to(device)
    
    # Create baseline model with matching architecture for fair comparison
    baseline_model = nn.Sequential(
        # Feature extraction - identical to ATQ
        nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        
        # Classifier - use same width as ATQ for fair comparison
        nn.Linear(64 * 7 * 7, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_size, 10)
    ).to(device)
    
    # Print model information
    print(f"ATQ Model Parameters: {count_parameters(model):,}")
    print(f"Baseline Model Parameters: {count_parameters(baseline_model):,}")
    
    # Optimizers
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=0.0001
    )
    
    baseline_optimizer = optim.Adam(
        baseline_model.parameters(), 
        lr=args.learning_rate
    )
    
    # Learning rate schedulers
    if args.use_cosine_lr:
        # Cosine annealing with warmup
        def get_lr_lambda(warmup_steps, total_steps, min_factor=0.1):
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    # Linear warmup
                    return float(current_step) / float(max(1, warmup_steps))
                
                # Cosine decay after warmup
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(min_factor, 0.5 * (1.0 + math.cos(math.pi * progress)))
            return lr_lambda
        
        # Calculate total training steps
        total_steps = len(train_loader) * args.epochs
        warmup_steps = total_steps // 10  # Use 10% of steps for warmup
        
        atq_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=get_lr_lambda(warmup_steps, total_steps)
        )
        
        baseline_scheduler = optim.lr_scheduler.LambdaLR(
            baseline_optimizer, 
            lr_lambda=get_lr_lambda(warmup_steps, total_steps)
        )
    else:
        # Step decay - simpler but still effective
        atq_scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=args.epochs // 4, 
            gamma=0.5
        )
        
        baseline_scheduler = optim.lr_scheduler.StepLR(
            baseline_optimizer, 
            step_size=args.epochs // 4, 
            gamma=0.5
        )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Progressive sparsity settings
    initial_sparsity = 0.05  # Start with low sparsity
    final_sparsity = args.sparsity
    
    # Training tracking
    best_val_acc = 0.0
    train_accuracies = []
    val_accuracies = []
    sparsity_schedule = []
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        baseline_model.train()
        
        # Progressive sparsity: start low, gradually increase
        # Use 70% of epochs to reach target sparsity, then maintain
        current_sparsity = initial_sparsity + (final_sparsity - initial_sparsity) * min(1.0, epoch / (args.epochs * 0.7))
        sparsity_schedule.append(current_sparsity)
        
        # Progressive L1 regularization: stronger as training progresses
        # This helps establish pattern early but then allows fine-tuning
        l1_weight = args.l1_factor * min(1.0, epoch / (args.epochs * 0.5))
        
        # Update model's sparsity target
        if args.use_rpb:
            for m in model.modules():
                if hasattr(m, 'sparsity_target'):
                    m.sparsity_target = current_sparsity
        
        # Training stats
        train_loss = 0.0
        l1_reg_total = 0.0
        train_correct = 0
        train_total = 0
        baseline_train_correct = 0
        
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # TRAIN BASELINE MODEL
            baseline_optimizer.zero_grad()
            baseline_outputs = baseline_model(inputs)
            baseline_loss = criterion(baseline_outputs, targets)
            baseline_loss.backward()
            baseline_optimizer.step()
            
            # TRAIN ATQ MODEL
            optimizer.zero_grad()
            atq_outputs = model(inputs)
            
            # Standard cross-entropy loss
            atq_loss = criterion(atq_outputs, targets)
            
            # Add knowledge distillation loss if enabled
            if args.distill:
                # Temperature-scaled softmax for knowledge distillation
                temperature = 4.0  # Higher temperature for smoother probabilities
                with torch.no_grad():
                    teacher_logits = baseline_outputs / temperature
                
                student_logits = atq_outputs / temperature
                distill_loss = nn.KLDivLoss(reduction='batchmean')(
                    F.log_softmax(student_logits, dim=1),
                    F.softmax(teacher_logits, dim=1)
                ) * (temperature ** 2)
                
                # Combined loss (0.7 CE + 0.3 KD)
                loss = 0.7 * atq_loss + 0.3 * distill_loss
            else:
                loss = atq_loss
            
            # Add L1 regularization on weights to encourage sparsity
            if args.use_l1:
                l1_reg = 0
                for name, param in model.named_parameters():
                    if 'weight' in name and 'bn' not in name:
                        l1_reg += torch.sum(torch.abs(param))
                
                # Add to loss with progressive weight
                loss = loss + l1_weight * l1_reg
                l1_reg_total += l1_reg.item()
            
            # Backpropagate and update weights
            loss.backward()
            
            # Optional gradient clipping for stability
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
            optimizer.step()
            
            # Update learning rates if using cosine scheduler with per-step updates
            if args.use_cosine_lr:
                atq_scheduler.step()
                baseline_scheduler.step()
            
            # Track statistics
            train_loss += loss.item()
            _, predicted = atq_outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Track baseline statistics
            _, baseline_predicted = baseline_outputs.max(1)
            baseline_train_correct += baseline_predicted.eq(targets).sum().item()
            
            # Print progress every few batches
            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                train_time = time.time() - start_time
                print(f'Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | '
                      f'ATQ: {100.*train_correct/train_total:.1f}% | '
                      f'Base: {100.*baseline_train_correct/train_total:.1f}% | '
                      f'Loss: {train_loss/(batch_idx+1):.3f} | '
                      f'Sparsity: {current_sparsity:.2f} | '
                      f'Time: {train_time:.1f}s')
        
        # Update learning rates if using epoch-based schedulers
        if not args.use_cosine_lr:
            atq_scheduler.step()
            baseline_scheduler.step()
        
        # Record epoch training accuracy
        train_acc = 100. * train_correct / train_total
        train_accuracies.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Track statistics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_accuracies.append(val_acc)
        print(f'Validation | Epoch {epoch+1} | Acc: {val_acc:.1f}%')
        
        # Check weight distribution metrics if using RPB
        if args.use_rpb and (epoch+1) % 5 == 0:
            # Count zeros in the first quantized layer's weights
            if hasattr(model, 'classifier') and hasattr(model.classifier[0], 'weight'):
                layer = model.classifier[0]
                if hasattr(layer, 'sparsity_target'):
                    # Get quantized weights
                    w_ternary, _ = layer.get_quantized_weights()
                    
                    # Count values
                    total = w_ternary.numel()
                    zeros = (w_ternary == 0).sum().item()
                    neg_ones = (w_ternary == -1).sum().item()
                    pos_ones = (w_ternary == 1).sum().item()
                    
                    # Calculate percentages
                    zero_percent = 100 * zeros / total
                    neg_percent = 100 * neg_ones / total
                    pos_percent = 100 * pos_ones / total
                    
                    print(f"Weight distribution: "
                          f"-1: {neg_percent:.1f}% | "
                          f"0: {zero_percent:.1f}% | "
                          f"+1: {pos_percent:.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            torch.save(model.state_dict(), f'checkpoints/atq_model_{args.dataset}.pth')
            print(f'Model saved with accuracy: {best_val_acc:.1f}%')
    
    # Test both models
    print("\nTesting models...")
    model.eval()
    baseline_model.eval()
    test_correct = 0
    test_total = 0
    baseline_test_correct = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # ATQ model
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            # Baseline model
            baseline_outputs = baseline_model(inputs)
            _, baseline_predicted = baseline_outputs.max(1)
            baseline_test_correct += baseline_predicted.eq(targets).sum().item()
    
    test_acc = 100. * test_correct / test_total
    baseline_test_acc = 100. * baseline_test_correct / test_total
    
    # Print test results
    print(f'ATQ Test Accuracy: {test_acc:.1f}%')
    print(f'Baseline Test Accuracy: {baseline_test_acc:.1f}%')
    
    # Bit-packing analysis (theoretical memory savings)
    if args.bit_packing and args.use_rpb:
        print("\nBit-packing analysis:")
        # Get first layer's weights
        layer = model.classifier[0]
        if hasattr(layer, 'get_quantized_weights'):
            w_ternary, _ = layer.get_quantized_weights()
            
            # Original FP32 size
            fp32_size = w_ternary.numel() * 4  # 4 bytes per float32
            
            # Bit-packed size (2 bits per value)
            bit_packed_size = (w_ternary.numel() * 2) // 8  # 2 bits per value, 8 bits per byte
            
            # Theoretical compression ratio
            compression_ratio = fp32_size / bit_packed_size
            
            print(f"Original FP32 size: {fp32_size/1024:.2f} KB")
            print(f"Bit-packed size: {bit_packed_size/1024:.2f} KB")
            print(f"Theoretical compression ratio: {compression_ratio:.1f}x")
    
    # Measure inference time
    atq_time = measure_inference_time(model, torch.ones(1, input_channels, 28, 28).to(device), num_runs=50)
    baseline_time = measure_inference_time(baseline_model, torch.ones(1, input_channels, 28, 28).to(device), num_runs=50)
    
    # Model efficiency comparison
    print("\nEfficiency Comparison:")
    print(f"ATQ Model: {count_parameters(model):,} params | {measure_model_memory(model):.2f} MB | {atq_time:.2f} ms | {test_acc:.1f}%")
    print(f"Baseline: {count_parameters(baseline_model):,} params | {measure_model_memory(baseline_model):.2f} MB | {baseline_time:.2f} ms | {baseline_test_acc:.1f}%")
    
    # Improvement ratios
    param_ratio = count_parameters(baseline_model) / count_parameters(model)
    memory_ratio = measure_model_memory(baseline_model) / measure_model_memory(model)
    speed_ratio = baseline_time / atq_time
    
    print(f"Ratios: Params {param_ratio:.2f}x | Memory {memory_ratio:.2f}x | Speed {speed_ratio:.2f}x | Acc Delta {test_acc - baseline_test_acc:.1f}%")
    
    # Visualize ternary weight distribution
    if args.use_rpb:
        layer_name = "classifier.0.weight"
    else:
        layer_name = "classifier.0.weight"
        
    visualize_ternary_weights(model, layer_name)
    
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    plt.savefig('plots/ternary_distribution.png')
    plt.close()
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs+1), train_accuracies, label='Train')
    plt.plot(range(1, args.epochs+1), val_accuracies, label='Validation')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig('plots/training_curve.png')
    plt.close()
    
    # Plot sparsity schedule
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs+1), sparsity_schedule)
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Target Sparsity')
    plt.title('Progressive Sparsity Schedule')
    plt.savefig('plots/sparsity_schedule.png')
    plt.close()
    
    return model, test_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ATQ Image Classification')
    parser.add_argument('--dataset', type=str, default='fashion_mnist', choices=['mnist', 'fashion_mnist'],
                      help='Dataset to use (default: fashion_mnist)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (default: 256)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs (default: 20)')
    parser.add_argument('--use-rpb', action='store_true', help='Use Residual Precision Boosting')
    parser.add_argument('--distill', action='store_true', help='Use knowledge distillation')
    parser.add_argument('--sparsity', type=float, default=0.3, help='Target sparsity (0-1, default: 0.3)')
    parser.add_argument('--wider-layers', action='store_true', help='Use wider layers for ATQ model')
    parser.add_argument('--use-cosine-lr', action='store_true', help='Use cosine learning rate schedule')
    parser.add_argument('--l1-factor', type=float, default=1e-5, help='L1 regularization factor')
    parser.add_argument('--use-l1', action='store_true', help='Use L1 regularization for sparsity')
    parser.add_argument('--clip-grad', action='store_true', help='Apply gradient clipping')
    parser.add_argument('--bit-packing', action='store_true', help='Analyze bit-packing compression')
    
    args = parser.parse_args()
    train(args)
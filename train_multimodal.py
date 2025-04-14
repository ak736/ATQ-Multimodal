import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
from tqdm import tqdm

from models.multimodal_classifier import MultimodalATQ
from data.multimodal_data import prepare_fashion_mnist_multimodal
from utils.metrics import accuracy, confusion_matrix
from evaluate import evaluate_model

def train_multimodal(args):
    """
    Train a multimodal model with Adaptive Ternary Quantization
    """
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare dataset
    print("Preparing dataset...")
    train_loader, val_loader, vocab, class_names = prepare_fashion_mnist_multimodal(
        batch_size=args.batch_size,
        add_noise=args.add_text_noise
    )
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create model
    print("Creating model...")
    model = MultimodalATQ(
        num_classes=len(class_names),
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        vision_threshold=args.vision_threshold,
        text_threshold=args.text_threshold,
        fusion_threshold=args.fusion_threshold,
        use_residual=args.use_residual,
        residual_scale=args.residual_scale
    )
    model = model.to(device)
    
    # Print model size information
    size_info = model.get_model_size_info()
    print("Model size information:")
    for key, value in size_info.items():
        print(f"  {key}: {value}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for images, texts, labels in progress_bar:
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_acc += torch.sum(preds == labels).item()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Calculate average metrics
        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for images, texts, labels in val_loader:
                images = images.to(device)
                texts = texts.to(device)
                labels = labels.to(device)
                
                outputs = model(images, texts)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_acc += torch.sum(preds == labels).item()
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader.dataset)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{args.num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  New best model with validation accuracy: {val_acc:.4f}")
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))
    
    # Save training history
    history = {
        "train_loss": train_losses,
        "train_acc": train_accuracies,
        "val_loss": val_losses,
        "val_acc": val_accuracies
    }
    
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train")
    plt.plot(val_accuracies, label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"))
    
    # Final evaluation
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pth")))
    
    # Evaluate on validation set
    results = evaluate_model(model, val_loader, criterion, device, class_names)
    
    # Save results
    with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
        # Convert numpy types to regular Python types for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value
        json.dump(serializable_results, f, indent=4)
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    return model, history, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multimodal model with ATQ")
    
    # General settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--output_dir", type=str, default="./outputs/multimodal", help="Output directory")
    
    # Dataset settings
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--add_text_noise", action="store_true", help="Add noise to text descriptions")
    
    # Model settings
    parser.add_argument("--embed_dim", type=int, default=128, help="Text embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--vision_threshold", type=float, default=0.05, help="Threshold for vision quantization")
    parser.add_argument("--text_threshold", type=float, default=0.05, help="Threshold for text quantization")
    parser.add_argument("--fusion_threshold", type=float, default=0.05, help="Threshold for fusion quantization")
    parser.add_argument("--use_residual", action="store_true", help="Use residual connections")
    parser.add_argument("--residual_scale", type=float, default=0.1, help="Scale for residual connections")
    
    # Training settings
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    
    args = parser.parse_args()
    train_multimodal(args)
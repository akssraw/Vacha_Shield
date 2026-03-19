import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import AudioCNN
from dataset_loader import ASVspoof2019Loader

def train(args):
    print("=== Vacha-Shield ASVspoof Training Pipeline ===")
    
    # 1. Device Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    # 2. Setup Dataset & DataLoader
    train_dataset = ASVspoof2019Loader(
        dataset_path=args.dataset_dir,
        protocol_file=args.protocol_file,
        subset='train',
        feature_type=args.feature_type
    )
    
    if len(train_dataset) == 0:
        print("Error: Dataset appears to be empty. Please check paths.")
        return
        
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 3. Initialize Model, Loss, Optimizer
    model = AudioCNN(num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    print(f"Starting training on {len(train_dataset)} samples for {args.epochs} epochs...")

    # 4. Training Loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            # Move to GPU/CPU
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print intermediate progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        avg_loss = running_loss / len(train_loader)
        print(f"--> Epoch [{epoch+1}/{args.epochs}] Completed. Average Loss: {avg_loss:.4f}\n")
        
        # Save dynamically if this is the best epoch
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"[*] New best loss achieved! Saving model to '{args.save_path}'")
            torch.save(model.state_dict(), args.save_path)

    print("Training finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vacha-Shield prototype on ASVspoof 2019 LA Dataset.")
    parser.add_argument('--dataset_dir', type=str, default="data/ASVspoof2019_LA_train", help="Path to LA train folder (containing /flac)")
    parser.add_argument('--protocol_file', type=str, default="data/ASVspoof2019_LA_train/ASVspoof2019.LA.cm.train.trn.txt", help="Path to protocol txt file")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for Adam")
    parser.add_argument('--feature_type', type=str, default="spectrogram", choices=["spectrogram", "mfcc"], help="Type of feature to extract")
    parser.add_argument('--save_path', type=str, default="model.pth", help="Where to save the trained model weights")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of dataloader workers")
    
    args = parser.parse_args()
    
    # Just check if directory exists before starting
    if not os.path.exists(args.dataset_dir):
        print(f"WARNING: Dataset directory '{args.dataset_dir}' not found.")
        print("Please download ASVspoof 2019 Logical Access and set the --dataset_dir argument.")
    else:
        train(args)

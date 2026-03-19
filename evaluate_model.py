import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from model import AudioCNN
from dataset_loader import ASVspoof2019Loader

def evaluate(args):
    print("=== Vacha-Shield Evaluation Engine ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}\n")
    
    # 1. Load Model
    model = AudioCNN(num_classes=1)
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found. Train the model first.")
        return
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval() # Important: set to eval mode to disable dropout etc.
    print(f"Loaded model weights from '{args.model_path}'.")
    
    # 2. Setup Testing Dataset
    print(f"Loading Dev/Eval Dataset from {args.dataset_dir}...")
    test_dataset = ASVspoof2019Loader(
        dataset_path=args.dataset_dir,
        protocol_file=args.protocol_file,
        subset='eval', # typically 'dev' or 'eval'
        feature_type=args.feature_type
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, # No need to shuffle for evaluation
        num_workers=args.num_workers
    )

    # 3. Inference Loop
    all_true_labels = []
    all_pred_labels = []
    all_probabilities = []
    
    print("Starting evaluation inference...")
    
    with torch.no_grad(): # No backprop needed
        for batch_idx, (features, labels) in enumerate(test_loader):
            features = features.to(device)
            
            # Forward Pass
            logits = model(features)
            
            # Convert logits to probabilities using Sigmoid
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # Predict class 1 if prob > 0.5, else class 0
            preds = (probs > 0.5).astype(float)
            
            # Store everything for scikit-learn metrics
            all_true_labels.extend(labels.numpy().flatten())
            all_pred_labels.extend(preds.flatten())
            all_probabilities.extend(probs.flatten())
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Processed batch [{batch_idx+1}/{len(test_loader)}]")

    # 4. Metrics & Reporting
    true_labels = np.array(all_true_labels)
    predicted_labels = np.array(all_pred_labels)
    
    acc = accuracy_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, target_names=['Human (0)', 'AI/Spoof (1)'])
    
    print("\n" + "="*40)
    print("Evaluation Results")
    print("="*40)
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print("                 Predicted Human | Predicted Spoof")
    print(f"Actual Human:          {cm[0][0]:<14} | {cm[0][1]}")
    print(f"Actual Spoof:          {cm[1][0]:<14} | {cm[1][1]}")
    print("\nClassification Report:")
    print(report)
    print("="*40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Vacha-Shield model on ASVspoof Dev/Eval Sets.")
    parser.add_argument('--dataset_dir', type=str, default="data/ASVspoof2019_LA_dev", help="Path to LA dev/eval folder")
    parser.add_argument('--protocol_file', type=str, default="data/ASVspoof2019_LA_dev/ASVspoof2019.LA.cm.dev.trl.txt", help="Path to protocol txt file")
    parser.add_argument('--model_path', type=str, default="model.pth", help="Path to trained weights")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation")
    parser.add_argument('--feature_type', type=str, default="spectrogram", help="Type of feature expected by model")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of dataloader workers")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_dir):
        print(f"WARNING: Dataset directory '{args.dataset_dir}' not found. Cannot evaluate.")
    else:
        evaluate(args)

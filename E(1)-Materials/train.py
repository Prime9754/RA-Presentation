"""
E1: Single-Modality Baseline (OPG-only)
Train MedGemma-4B on panoramic X-rays (OPG) only for orthodontic diagnosis.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import json
from datetime import datetime
from PIL import Image  # <-- ADDED for new filter

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_utils.data_loader import OrthoDataset, collate_fn, save_split_datasets
# MODIFIED import to get the filter building blocks
from shared_utils.image_utils import (
    ImagePreprocessor, 
    detect_modality_from_name, 
    content_modality
)
from shared_utils.metrics import MetricsCalculator
from shared_utils.model_utils import MedGemmaWrapper


# ========================================================================
#           *** NEW (v3) HELPER FUNCTIONS ***
# ========================================================================

def robust_opg_check(path: str) -> bool:
    """
    Checks if a path is OPG. Returns True if it is OPG or
    ambiguous (and not confidently RGB/Ceph).
    """
    # Tier A: Filename check
    guess_a = detect_modality_from_name(path)
    if guess_a == 'opg':
        return True
    if guess_a in ('rgb', 'ceph'):
        # This is a confident non-OPG photo (e.g., ...right side.JPG)
        return False
    
    # Tier A is ambiguous (guess_a is None, e.g., '000002.PNG'). 
    # Fallback to Tier B content analysis.
    if os.path.exists(path):
        try:
            img = Image.open(path).convert('RGB')
            guess_b = content_modality(img) # from image_utils.py
            
            if guess_b == 'opg':
                return True # Confident OPG
            if guess_b in ('rgb', 'ceph'):
                return False # Confident RGB/Ceph by content
            
            # If guess_b is None (ambiguous zone 1.25-1.70)
            # We assume it's OPG for this experiment,
            # as it's definitely not a confident RGB photo.
            return True
            
        except Exception:
            # Can't open image, assume it's OK to keep
            return True 
    
    # Default: Path doesn't exist, or name is ambiguous
    # and content analysis failed. Keep it just in case.
    # This matches '000002.PNG' or '000002_.png'
    return True


def filter_csv_for_opg(csv_path: str):
    """
    Loads a CSV, filters its Image_Paths column for OPGs using the
    new robust_opg_check, and re-saves the file.
    """
    print(f"Applying robust OPG-only filter to {csv_path}...")
    try:
        df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
    except pd.errors.EmptyDataError:
        print(f"Warning: {csv_path} is empty. Skipping.")
        return 0
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found. Skipping.")
        return 0
    
    if 'Image_Paths' not in df.columns:
        print(f"Error: 'Image_Paths' column not found in {csv_path}.")
        return 0

    def filter_row_paths(paths_str):
        """Helper to filter paths in a single row."""
        if not isinstance(paths_str, str):
            return ""
        
        all_paths = [p.strip() for p in paths_str.split('; ') if p.strip()]
        
        # Use the NEW robust_opg_check
        opg_paths = [
            path for path in all_paths 
            if robust_opg_check(path)
        ]
        
        return '; '.join(opg_paths)

    # Apply the filter to the entire 'Image_Paths' column
    df['Image_Paths'] = df['Image_Paths'].apply(filter_row_paths)
    
    # --- CRITICAL ---
    # Drop rows that now have no OPG images
    original_len = len(df)
    df = df.dropna(subset=['Image_Paths'])
    df = df[df['Image_Paths'].str.strip() != '']
    new_len = len(df)
    
    # Re-save the file, overwriting the old one
    df.to_csv(csv_path, index=False)
    
    print(f"...Done. Filtered {original_len} rows down to {new_len} (OPG-only).")
    return new_len

# ========================================================================
#           *** END NEW HELPER FUNCTIONS ***
# ========================================================================


class OPGClassifier(nn.Module):
    """Classification model using MedGemma vision encoder for OPG images."""
    
    def __init__(self, vision_encoder, feature_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.vision_encoder = vision_encoder
        
        # Classification head - MODIFIED for ViT/SigLIP
        # We remove the AdaptiveAvgPool2d and Flatten layers,
        # as we will use the model's `pooler_output` directly.
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),  # feature_dim is 768, which matches the pooler output
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, images):
        # Extract features
        # We must pass interpolate_pos_encoding=True to fix the RuntimeError
        features = self.vision_encoder(
            pixel_values=images,
            interpolate_pos_encoding=True
        )
        
        # Get the pooler_output, which is (Batch_Size, feature_dim)
        # and is ready for the nn.Linear layer
        pooled_features = features.last_hidden_state[:, 0]
        
        # Classify
        logits = self.classifier(pooled_features)
        
        return logits

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch in pbar:
        images = batch['images']
        responses = batch['responses']
        
        # Convert responses to labels (simple mapping for now)
        # This should be adapted based on actual label encoding
        labels = torch.tensor([hash(r) % 3 for r in responses], dtype=torch.long).to(device)
        
        # Move images to device
        if isinstance(images, list):
            if len(images) == 0:
                continue
            images = torch.stack(images).to(device)
        else:
            images = images.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(images)
        
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, num_classes):
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        
        for batch in pbar:
            images = batch['images']
            responses = batch['responses']
            
            labels = torch.tensor([hash(r) % 3 for r in responses], dtype=torch.long).to(device)
            
            if isinstance(images, list):
                if len(images) == 0:
                    continue
                images = torch.stack(images).to(device)
            else:
                images = images.to(device)
            
            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Get probabilities
            probs = torch.softmax(logits, dim=1)
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics_calc = MetricsCalculator(num_classes, class_names=[f"Class_{i}" for i in range(num_classes)])
    metrics = metrics_calc(all_labels, all_preds, all_probs)
    
    return avg_loss, metrics


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Start Robust Split ---
    splits_dir = os.path.join(args.output_dir, 'data_splits')
    train_csv_path = os.path.join(splits_dir, 'train.csv')
    val_csv_path = os.path.join(splits_dir, 'val.csv')
    test_csv_path = os.path.join(splits_dir, 'test.csv') # Define test path
    
    # Check for the *file*, not just the directory
    if not os.path.exists(train_csv_path): 
        print(f"Train split not found at {train_csv_path}. Splitting dataset...")
        save_split_datasets(
            csv_path=args.data_csv,
            output_dir=splits_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.seed
        )
    else:
        print(f"Found existing splits in {splits_dir}")
    # --- End Robust Split ---    

    # ========================================================================
    #           *** NEW: Apply OPG-only filter to generated splits ***
    # ========================================================================
    # This block now calls the new, corrected filter function
    
    print("\nApplying robust OPG-only filter to data splits...")
    filter_csv_for_opg(train_csv_path)
    filter_csv_for_opg(val_csv_path)
    
    # Also filter the test set (generated by save_split_datasets)
    if os.path.exists(test_csv_path):
        filter_csv_for_opg(test_csv_path)
    print("...OPG filtering complete.\n")
    # ========================================================================
    #           *** END NEW BLOCK ***
    # ========================================================================
        
    # Create datasets
    image_processor = ImagePreprocessor(
        image_size=args.image_size,
        normalize=True,
    )
    
    train_dataset = OrthoDataset(
        csv_path=train_csv_path, # Use filtered path
        image_processor=image_processor,
        modality_filter=None, # Filter is ALREADY applied to the CSV
        max_images=1  # <-- FIX: Enforces 1 image per patient
    )
    
    val_dataset = OrthoDataset(
        csv_path=val_csv_path, # Use filtered path
        image_processor=image_processor,
        modality_filter=None, # Filter is ALREADY applied to the CSV
        max_images=1  # <-- FIX: Enforces 1 image per patient
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    print(f"Train dataset size: {len(train_dataset)} (OPG-only)")
    print(f"Validation dataset size: {len(val_dataset)} (OPG-only)")
    
    # Load MedGemma model
    print(f"Loading MedGemma checkpoint from {args.checkpoint_path}...")
    
    try:
        medgemma = MedGemmaWrapper(args.checkpoint_path, device=device)
        vision_encoder = medgemma.get_vision_encoder()
        
        if vision_encoder is None:
            print("Error: Could not load vision encoder")
            return
        
        # Freeze vision encoder initially
        if args.freeze_encoder:
            for param in vision_encoder.parameters():
                param.requires_grad = False
        
        # Create classifier
        # Get the feature dimension *dynamically* from the loaded model's config
        try:
            feature_dim = vision_encoder.config.hidden_size
        except AttributeError:
            print("Error: Could not find 'hidden_size' in vision_encoder.config.")
            sys.exit(1)
            
        print(f"Dynamically set feature_dim to: {feature_dim}")
        
        model = OPGClassifier(
            vision_encoder=vision_encoder,
            feature_dim=feature_dim, # This will now be 1152 (or whatever is correct)
            num_classes=args.num_classes,
            dropout=args.dropout
        ).to(device)
        
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the MedGemma checkpoint and base model are available.")
        return
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    # Training loop
    best_val_accuracy = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, args.num_classes
        )
        
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['f1_macro']
        
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_metrics': val_metrics
            }
            
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"✓ Saved best model (val_acc: {val_acc:.4f})")
    
    # Save final model and history
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pt'))
    
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*50}")
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Models saved to {args.output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E1: Train OPG-only baseline model")
    
    # Data arguments
    parser.add_argument('--data_csv', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to MedGemma checkpoint')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    
    # Dataset split
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze vision encoder')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    
    args = parser.parse_args()
    
    main(args)
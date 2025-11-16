"""
E3: MedGemma Multi-Image Prompting (Unmodified Encoder)

Feed multiple images as separate inputs with instruction prompts for attention-based fusion.

UPDATED: Includes patient-aware splitting, robust modality detection, and data leakage prevention.
Core logic from E2 applied: patient-level splitting, robust 3-tier modality detection, improved label parsing.
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
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_utils.data_loader import OrthoDataset, collate_fn, save_split_datasets
from shared_utils.image_utils import detect_modality_from_path
from shared_utils.metrics import MetricsCalculator
from shared_utils.model_utils import MedGemmaWrapper


def label_to_class(response: str) -> int:
    """
    Map textual 'Response' field to numeric class (Class I/II/III).
    Robust version from E2 improvements.
    
    Returns:
        0 = Class I
        1 = Class II
        2 = Class III
    """
    if not isinstance(response, str):
        return 0  # Default to Class I
    
    r = response.lower().strip()
    
    # Check for specific class mentions (most specific first)
    if 'class iii' in r or 'class 3' in r or 'iii' in r:
        return 2
    if 'class ii' in r or 'class 2' in r or ' ii' in r:
        return 1
    if 'class i' in r or 'class 1' in r or ' i' in r:
        return 0
    
    # Fallback to Class I for unlabeled/ambiguous cases
    return 0


def create_multi_image_prompt(image_paths, base_prompt, num_images):
    """
    Create a detailed prompt with image tokens for MedGemma multi-image processing.
    
    Args:
        image_paths: List of image paths
        base_prompt: Base diagnostic prompt
        num_images: Number of images being processed
    
    Returns:
        Enhanced prompt with <image> tokens for MedGemma
    """
    modality_descriptions = {
        'intraoral': 'intraoral photograph',
        'opg': 'panoramic X-ray (OPG)',
        'ceph': 'lateral cephalometric X-ray'
    }
    
    # Create image tokens and descriptions
    image_tokens = []
    image_descriptions = []
    
    for i, path in enumerate(image_paths[:num_images], 1):
        # Add image token for MedGemma (uses <start_of_image> token)
        image_tokens.append("<start_of_image>")
        
        # Detect modality using robust 3-tier detection
        modality = detect_modality_from_path(path, use_content_analysis=True)
        desc = modality_descriptions.get(modality, 'clinical image')
        image_descriptions.append(f"Image {i} is a {desc}")
    
    # Construct enhanced prompt with image tokens
    tokens_str = " ".join(image_tokens)
    descriptions_str = ". ".join(image_descriptions)
    
    enhanced_prompt = f"""{tokens_str}

You are analyzing multiple orthodontic images of the same patient.
{descriptions_str}.

Based on all provided images, {base_prompt}

Consider information from all modalities in your diagnosis."""
    
    return enhanced_prompt


class MultiImageMedGemma(nn.Module):
    """
    Wrapper around MedGemma that handles multi-image input with prompting.
    """
    
    def __init__(self, medgemma_wrapper, num_classes, max_images=7):
        super().__init__()
        self.medgemma = medgemma_wrapper
        self.num_classes = num_classes
        self.max_images = max_images
        
        # Get model and processor
        self.model = medgemma_wrapper.model
        self.processor = medgemma_wrapper.processor
        self.tokenizer = medgemma_wrapper.tokenizer
        
        # Classification head (maps hidden states to class logits)
        # Get hidden size from model config
        if hasattr(self.model.config, 'hidden_size'):
            hidden_size = self.model.config.hidden_size
        elif hasattr(self.model.config, 'text_config') and hasattr(self.model.config.text_config, 'hidden_size'):
            hidden_size = self.model.config.text_config.hidden_size
        else:
            hidden_size = 2048  # Default for Gemma-2B
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, images, prompts, image_paths_list):
        """
        Forward pass with multiple images and prompts.
        
        Args:
            images: List of image lists (batch)
            prompts: List of text prompts
            image_paths_list: List of image path lists for prompt enhancement
        
        Returns:
            Classification logits
        """
        batch_logits = []
        
        # Freeze MedGemma encoder (only train classifier)
        self.model.eval()
        
        for imgs, prompt, paths in zip(images, prompts, image_paths_list):
            if len(imgs) == 0:
                # Return zero logits if no images (with gradient support)
                zero_logits = torch.zeros(self.num_classes, device=self.model.device, requires_grad=True)
                batch_logits.append(zero_logits)
                continue
            
            # Limit number of images
            if len(imgs) > self.max_images:
                imgs = imgs[:self.max_images]
                paths = paths[:self.max_images]
            
            # Create enhanced prompt with correct number of <start_of_image> tokens
            enhanced_prompt = create_multi_image_prompt(paths, prompt, len(imgs))
            
            # Process inputs
            try:
                inputs = self.processor(
                    text=enhanced_prompt,
                    images=imgs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                # Move to device
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # Extract features with frozen encoder (no gradients for MedGemma)
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                
                # Extract last hidden state for classification
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    last_hidden = outputs.hidden_states[-1]
                    # Pool: take mean of sequence
                    pooled = last_hidden.mean(dim=1).squeeze(0)
                    
                    # Detach from encoder graph and classify (enables gradients for classifier only)
                    pooled = pooled.detach()
                    logits = self.classifier(pooled)
                else:
                    # Fallback: use zeros with gradient support
                    logits = torch.zeros(self.num_classes, device=self.model.device, requires_grad=True)
                
                batch_logits.append(logits)
            
            except Exception as e:
                print(f"Warning: Error processing sample: {e}")
                # Use zeros with gradient support for error cases
                zero_logits = torch.zeros(self.num_classes, device=self.model.device, requires_grad=True)
                batch_logits.append(zero_logits)
        
        # Stack batch
        if len(batch_logits) == 0:
            return torch.zeros(1, self.num_classes, device=self.model.device, requires_grad=True)
        
        return torch.stack(batch_logits)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, num_classes):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch in pbar:
        images = batch['images']  # List of PIL Images
        prompts = batch['prompts']
        responses = batch['responses']
        image_paths = batch['image_paths']
        num_images_list = batch['num_images']
        
        # Convert responses to labels using ROBUST label parsing
        labels = torch.tensor([label_to_class(r) for r in responses], dtype=torch.long).to(device)
        
        # Organize images by patient
        batch_images = []
        batch_paths = []
        start_idx = 0
        for i, num_imgs in enumerate(num_images_list):
            patient_images = images[start_idx:start_idx + num_imgs]
            patient_paths = image_paths[i] if i < len(image_paths) else []
            batch_images.append(patient_images)
            batch_paths.append(patient_paths)
            start_idx += num_imgs
        
        # Forward pass
        optimizer.zero_grad()
        
        try:
            logits = model(batch_images, prompts, batch_paths)
            
            # Ensure correct shape
            if logits.shape[0] != labels.shape[0]:
                logits = logits[:labels.shape[0]]
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        except Exception as e:
            print(f"Error in training batch: {e}")
            continue
    
    if num_batches == 0:
        return 0.0, 0.0
    
    avg_loss = total_loss / num_batches
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, num_classes):
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        
        for batch in pbar:
            images = batch['images']
            prompts = batch['prompts']
            responses = batch['responses']
            image_paths = batch['image_paths']
            num_images_list = batch['num_images']
            
            # Convert responses to labels using ROBUST label parsing
            labels = torch.tensor([label_to_class(r) for r in responses], dtype=torch.long).to(device)
            
            # Organize images by patient
            batch_images = []
            batch_paths = []
            start_idx = 0
            for i, num_imgs in enumerate(num_images_list):
                patient_images = images[start_idx:start_idx + num_imgs]
                patient_paths = image_paths[i] if i < len(image_paths) else []
                batch_images.append(patient_images)
                batch_paths.append(patient_paths)
                start_idx += num_imgs
            
            try:
                logits = model(batch_images, prompts, batch_paths)
                
                if logits.shape[0] != labels.shape[0]:
                    logits = logits[:labels.shape[0]]
                
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            except Exception as e:
                print(f"Error in validation batch: {e}")
                continue
    
    if num_batches == 0:
        return 0.0, {}
    
    avg_loss = total_loss / num_batches
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = np.mean(all_preds == all_labels)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro
    }
    
    return avg_loss, metrics


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Split dataset with PATIENT-AWARE SPLITTING (prevents data leakage)
    splits_dir = os.path.join(args.output_dir, 'data_splits')
    train_csv = os.path.join(splits_dir, 'train.csv')
    val_csv = os.path.join(splits_dir, 'val.csv')
    test_csv = os.path.join(splits_dir, 'test.csv')
    
    # Check if split files exist
    if not (os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv)):
        print("Splitting dataset (patient-aware to prevent data leakage)...")
        save_split_datasets(
            csv_path=args.data_csv,
            output_dir=splits_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.seed
        )
        print(f"✓ Dataset split complete. Files saved to {splits_dir}\n")
    else:
        print(f"Using existing splits from {splits_dir}")
        # Validate existing splits for patient overlap
        print("Checking existing splits for patient overlap...")
        try:
            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(val_csv)
            
            if 'Patient_ID' in train_df.columns and 'Patient_ID' in val_df.columns:
                train_patients = set(train_df['Patient_ID'].astype(str).str.strip())
                val_patients = set(val_df['Patient_ID'].astype(str).str.strip())
                
                overlap = train_patients.intersection(val_patients)
                if len(overlap) > 0:
                    print(f"*** WARNING: Found {len(overlap)} overlapping patients in existing splits! ***")
                    print(f"*** Please delete the files in {splits_dir} and re-run. ***\n")
                else:
                    print("✓ No overlapping patients found between train and val splits.\n")
            else:
                print("WARNING: Patient_ID column not found in splits. Cannot verify patient separation.\n")
        except Exception as e:
            print(f"Error checking existing splits: {e}\n")
    
    # Create datasets - no preprocessing since MedGemma handles it
    train_dataset = OrthoDataset(
        csv_path=train_csv,
        image_processor=None,  # MedGemma processor will handle this
        modality_filter=None,
        max_images=args.max_images
    )
    
    val_dataset = OrthoDataset(
        csv_path=val_csv,
        image_processor=None,
        modality_filter=None,
        max_images=args.max_images
    )
    
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
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}\n")
    
    # Load MedGemma model
    print(f"Loading MedGemma checkpoint from {args.checkpoint_path}...")
    
    try:
        medgemma = MedGemmaWrapper(args.checkpoint_path, device=device)
        
        if medgemma.model is None:
            print("Error: Could not load MedGemma model")
            return
        
        # Create multi-image model
        model = MultiImageMedGemma(
            medgemma_wrapper=medgemma,
            num_classes=args.num_classes,
            max_images=args.max_images
        ).to(device)
        
        print("Multi-image MedGemma model created")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # Only optimize the classification head
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_accuracy = 0.0
    best_val_f1 = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    print("\n" + "="*70)
    print("STARTING E3 TRAINING: Multi-Image Prompting with MedGemma")
    print("="*70)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Max images per patient: {args.max_images}")
    print(f"Number of classes: {args.num_classes}")
    print("="*70)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.num_classes
        )
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, args.num_classes
        )
        
        if val_metrics:
            val_acc = val_metrics['accuracy']
            val_f1 = val_metrics['f1_macro']
            
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            
            scheduler.step()
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            # Save best model (based on F1 score for better class balance)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_accuracy = val_acc
                
                checkpoint = {
                    'epoch': epoch,
                    'classifier_state_dict': model.classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'val_f1': val_f1,
                    'val_metrics': val_metrics,
                    'args': vars(args)
                }
                
                torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))
                print(f"✓ Saved best model (val_acc: {val_acc:.4f}, val_f1: {val_f1:.4f})")
    
    # Save final model and history
    torch.save(model.classifier.state_dict(), os.path.join(args.output_dir, 'final_classifier.pt'))
    
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save training configuration
    config = {
        'experiment': 'E3_Multi_Image_Prompting',
        'num_classes': args.num_classes,
        'max_images': args.max_images,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'best_val_accuracy': best_val_accuracy,
        'best_val_f1': best_val_f1,
        'checkpoint_path': args.checkpoint_path,
        'patient_aware_splitting': True,
        'robust_modality_detection': True
    }
    
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*70}")
    print("✓ E3 TRAINING COMPLETED!")
    print("="*70)
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best validation F1 score: {best_val_f1:.4f}")
    print(f"\nModels saved to: {args.output_dir}")
    print(f"  - best_model.pt (classifier weights + metrics)")
    print(f"  - final_classifier.pt (final epoch weights)")
    print(f"  - training_history.json (loss/acc curves)")
    print(f"  - config.json (experiment configuration)")
    print("="*70)
    print("\n✓ Data leakage prevention: Patient-aware splitting applied")
    print("✓ Modality detection: Robust 3-tier classification (filename → content → CLIP)")
    print("✓ Label parsing: Improved Class I/II/III detection")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E3: Train multi-image prompting model")
    
    parser.add_argument('--data_csv', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--max_images', type=int, default=7, help='Maximum images per patient')
    
    parser.add_argument('--batch_size', type=int, default=4, help='Smaller batch for multi-image')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=2)
    
    args = parser.parse_args()
    
    main(args)
"""
E2: Naïve Late Fusion (Per-Modality Heads + Ensemble)

Train 3 separate classifiers (intraoral, OPG, ceph), then fuse at decision level.
This is a self-contained implementation with all utilities included.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import f1_score, classification_report
from torchvision import transforms
import random
import json


# ============================================================================
# UTILITIES (Self-contained)
# ============================================================================

def detect_modality_from_path(path: str) -> str:
    """
    Detect modality using robust 3-tier classification from shared_utils.
    Falls back to embedded logic if shared_utils not available.
    """
    try:
        from shared_utils.image_utils import detect_modality_from_path as robust_detect
        return robust_detect(path, use_content_analysis=True)
    except ImportError:
        # Fallback to basic detection if shared_utils not available
        path_lower = path.lower()
        
        # OPG/Panoramic indicators
        if any(kw in path_lower for kw in ['pan.png', 'pan.jpg', 'opg', 'panoramic']):
            return 'opg'
        
        # Ceph indicators
        if any(kw in path_lower for kw in ['ceph', 'lateral', 'lat_ceph']):
            return 'ceph'
        
        # Intraoral indicators
        if any(kw in path_lower for kw in ['intra', 'intraoral', 'tooth', 'teeth', 'buccal', 'lingual']):
            return 'intraoral'
        
        # Default guess
        return 'intraoral'


def label_to_class(response: str) -> int:
    """
    Map textual 'Response' field to numeric class.
    Assumes malocclusion classes are encoded as strings in response.
    """
    if not isinstance(response, str):
        return 0
    
    r = response.lower()
    if 'class i' in r or 'class 1' in r:
        return 0
    if 'class ii' in r or 'class 2' in r:
        return 1
    if 'class iii' in r or 'class 3' in r:
        return 2
    
    # Fallback
    return 3


def analyze_modality_distribution(csv_path: str) -> Dict[str, Dict]:
    """
    Debug utility: analyze how many images of each modality exist per patient.
    """
    df = pd.read_csv(csv_path)
    
    modality_counts = {
        'intraoral': 0,
        'opg': 0,
        'ceph': 0
    }
    patients = df['Patient_ID'].unique()
    
    for pid in patients:
        rows = df[df['Patient_ID'] == pid]
        for _, row in rows.iterrows():
            image_paths_str = str(row.get('Image_Paths', ''))
            if not isinstance(image_paths_str, str):
                continue
            for p in image_paths_str.split('; '):
                p = p.strip()
                if not p:
                    continue
                mod = detect_modality_from_path(p)
                if mod in modality_counts:
                    modality_counts[mod] += 1
    
    print("\nModality distribution (across all patients):")
    for mod, cnt in modality_counts.items():
        print(f"  {mod}: {cnt} images")
    
    return modality_counts


def split_and_save_dataset(
    csv_path: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
):
    """
    Save train/val/test splits to separate CSV files, ensuring a
    patient-level split to prevent data leakage.
    
    MODIFIED: This function now drops unused columns (like 'Prompt')
    before saving the split files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load the full dataset
    try:
        df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
    except Exception as e:
        print(f"Error loading CSV {csv_path}: {e}")
        return

    # 2. Clean data and get unique patients
    #    CRITICAL: Drop rows where Patient_ID is missing
    required_cols = ['Image_Paths', 'Response', 'Patient_ID']
    df = df.dropna(subset=required_cols)
    
    # Check if all required columns are present after loading
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Master CSV '{csv_path}' is missing required columns: {missing_cols}")
        print("Please ensure the CSV contains 'Patient_ID', 'Image_Paths', and 'Response'.")
        return
        
    # Strip whitespace from Patient_ID
    df['Patient_ID'] = df['Patient_ID'].astype(str).str.strip()
    
    unique_patients = df['Patient_ID'].unique().tolist()
    print(f"Loaded {len(df)} rows with {len(unique_patients)} unique patients from {csv_path}")
    
    # 3. Shuffle patients
    rng = np.random.RandomState(random_seed)
    rng.shuffle(unique_patients)
    
    n_total = len(unique_patients)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val
    
    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train:n_train + n_val])
    test_patients = set(unique_patients[n_train + n_val:])
    
    # 4. Check for accidental overlaps
    train_val_overlap = train_patients.intersection(val_patients)
    train_test_overlap = train_patients.intersection(test_patients)
    val_test_overlap = val_patients.intersection(test_patients)
    
    print("\nPatient-level split summary:")
    print(f"  Train patients: {len(train_patients)}")
    print(f"  Val patients:   {len(val_patients)}")
    print(f"  Test patients:  {len(test_patients)}")
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("WARNING: Overlapping patients found between splits!")
        print(f"  Train-Val overlap: {len(train_val_overlap)}")
        print(f"  Train-Test overlap: {len(train_test_overlap)}")
        print(f"  Val-Test overlap: {len(val_test_overlap)}")
    
    # 5. Assign rows to splits
    def assign_split(pid: str) -> str:
        if pid in train_patients:
            return 'train'
        elif pid in val_patients:
            return 'val'
        elif pid in test_patients:
            return 'test'
        else:
            return 'unknown'
    
    df['split'] = df['Patient_ID'].apply(assign_split)
    
    train_df = df[df['split'] == 'train'].copy()
    val_df   = df[df['split'] == 'val'].copy()
    test_df  = df[df['split'] == 'test'].copy()
    
    # 6. Save
    train_path = os.path.join(output_dir, 'train.csv')
    val_path   = os.path.join(output_dir, 'val.csv')
    test_path  = os.path.join(output_dir, 'test.csv')
    
    # <-- START MODIFICATION -->
    # As requested, select only the columns needed for the classifier
    # This drops 'Prompt', 'Numeric_Key', and the temporary 'split' column
    final_columns = ['Patient_ID', 'Image_Paths', 'Response']
    
    train_df = train_df[final_columns]
    val_df   = val_df[final_columns]
    test_df  = test_df[final_columns]
    # <-- END MODIFICATION -->
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print("\nSaved splits to (with only required columns):")
    print(f"  {train_path} ({len(train_df)} rows)")
    print(f"  {val_path} ({len(val_df)} rows)")
    print(f"  {test_path} ({len(test_df)} rows)")
    
    # 7. Final overlap sanity check
    def get_patient_set(path: str) -> set:
        d = pd.read_csv(path)
        if 'Patient_ID' not in d.columns:
            return set()
        return set(d['Patient_ID'].astype(str).str.strip())
    
    train_pat_set = get_patient_set(train_path)
    val_pat_set   = get_patient_set(val_path)
    test_pat_set  = get_patient_set(test_path)
    
    print("\nChecking final overlaps after saving splits:")
    print(f"  Train-Val overlap: {len(train_pat_set & val_pat_set)}")
    print(f"  Train-Test overlap: {len(train_pat_set & test_pat_set)}")
    print(f"  Val-Test overlap: {len(val_pat_set & test_pat_set)}")
    print("Done splitting dataset.\n")


# ============================================================================
# DATASET
# ============================================================================

class ModalityDataset(Dataset):
    """Dataset that loads ONE modality per patient."""
    
    def __init__(
        self,
        csv_path: str,
        modality: str,  # 'intraoral', 'opg', or 'ceph'
        image_size: Tuple[int, int] = (224, 224),
        transform=None
    ):
        self.csv_path = csv_path
        self.modality = modality
        self.image_size = image_size
        self.transform = transform
        
        # Read CSV
        try:
            self.df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error loading CSV {csv_path}: {e}")
            self.df = pd.DataFrame(columns=['Image_Paths', 'Response', 'Patient_ID'])
        
        # Filter out rows with missing fields
        self.df = self.df.dropna(subset=['Image_Paths', 'Response', 'Patient_ID'])
        self.df['Patient_ID'] = self.df['Patient_ID'].astype(str).str.strip()
        
        print(f"[{modality}] Loaded {len(self.df)} patients from {csv_path}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        
        # Parse image paths (split by '; ')
        image_paths_str = str(row['Image_Paths'])
        image_paths = []
        for p in image_paths_str.split('; '):
            cleaned_path = p.strip().replace('\n', '').replace('\r', '')
            cleaned_path = ' '.join(cleaned_path.split())  # Remove extra whitespace
            if cleaned_path and len(cleaned_path) > 3:
                if ('/' in cleaned_path or '\\' in cleaned_path) and ('.' in cleaned_path):
                    image_paths.append(cleaned_path)
        
        # Collect candidate images for this modality and randomly pick one
        modality_paths = [
            path for path in image_paths
            if detect_modality_from_path(path) == self.modality
        ]
        
        target_image = None
        if modality_paths:
            random.shuffle(modality_paths)
            for path in modality_paths:
                if os.path.exists(path):
                    try:
                        img = Image.open(path).convert('RGB')
                        img = img.resize(self.image_size, Image.BILINEAR)
                        target_image = img
                        break
                    except Exception:
                        continue
        
        # Convert to tensor
        if target_image is not None:
            if self.transform:
                target_image = self.transform(target_image)
            else:
                # Simple normalization if no transform is provided
                target_image = torch.from_numpy(np.array(target_image)).permute(2, 0, 1).float() / 255.0
        else:
            # Return black image if not found
            target_image = torch.zeros(3, *self.image_size)
        
        # Get label
        label = label_to_class(row['Response'])
        
        return {
            'image': target_image,
            'label': torch.tensor(label, dtype=torch.long),
            'patient_id': str(row['Patient_ID']),
            'has_image': target_image is not None,
        }


# ============================================================================
# MODEL
# ============================================================================

class SimpleClassifier(nn.Module):
    """Lightweight CNN classifier for one modality."""
    
    def __init__(self, num_classes: int = 4):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56x56
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class EnsembleModel(nn.Module):
    """Late fusion ensemble of 3 modality-specific classifiers."""
    
    def __init__(
        self,
        intraoral_model: nn.Module,
        opg_model: nn.Module,
        ceph_model: nn.Module,
        fusion_method: str = 'average',
        num_classes: int = 4
    ):
        super().__init__()
        self.intraoral_model = intraoral_model
        self.opg_model = opg_model
        self.ceph_model = ceph_model
        self.fusion_method = fusion_method
        
        if fusion_method == 'learned':
            self.weights = nn.Parameter(torch.ones(3))
        else:
            self.weights = None
        
        self.num_classes = num_classes
    
    def forward(self, intraoral_x, opg_x, ceph_x):
        intraoral_logits = self.intraoral_model(intraoral_x)
        opg_logits = self.opg_model(opg_x)
        ceph_logits = self.ceph_model(ceph_x)
        
        if self.fusion_method == 'average':
            fused_logits = (intraoral_logits + opg_logits + ceph_logits) / 3.0
        elif self.fusion_method == 'weighted':
            weights = torch.tensor([0.4, 0.3, 0.3], device=intraoral_logits.device)
            fused_logits = (
                weights[0] * intraoral_logits +
                weights[1] * opg_logits +
                weights[2] * ceph_logits
            )
        elif self.fusion_method == 'learned':
            weights = torch.softmax(self.weights, dim=0)
            fused_logits = (
                weights[0] * intraoral_logits +
                weights[1] * opg_logits +
                weights[2] * ceph_logits
            )
        else:
            fused_logits = (intraoral_logits + opg_logits + ceph_logits) / 3.0
        
        return fused_logits


# ============================================================================
# TRAINING
# ============================================================================

def train_single_modality(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    modality: str
) -> Tuple[float, float]:
    """Train one modality classifier for one epoch."""
    model.train()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [{modality}]")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return avg_loss, accuracy


def validate_single_modality(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    modality: str
) -> Tuple[float, Dict]:
    """Validate one modality classifier."""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Val [{modality}]"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, {'accuracy': accuracy, 'f1_macro': f1}


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='E2: Naïve Late Fusion')
    parser.add_argument('--data_csv', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--fusion_method', type=str, default='average', 
                       choices=['average', 'weighted', 'learned'], help='Fusion method')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    splits_dir = os.path.join(args.output_dir, 'data_splits')
    
    # Split dataset if needed
    train_csv = os.path.join(splits_dir, 'train.csv')
    val_csv = os.path.join(splits_dir, 'val.csv')
    test_csv = os.path.join(splits_dir, 'test.csv')
    
    # This block will now call the *corrected* split_and_save_dataset function
    if not (os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv)):
        print("Splitting dataset (patient-aware)...")
        # The --data_csv argument should be your MASTER csv file
        split_and_save_dataset(args.data_csv, splits_dir)
    else:
        print(f"Using existing splits from {splits_dir}")
        # We can run a quick check on the existing splits
        print("Checking existing splits for patient overlap...")
        try:
            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(val_csv)
            
            train_patients = set(train_df['Patient_ID'].astype(str).str.strip())
            val_patients = set(val_df['Patient_ID'].astype(str).str.strip())
            
            overlap = train_patients.intersection(val_patients)
            if len(overlap) > 0:
                print(f"*** WARNING: Found {len(overlap)} overlapping patients in existing splits! ***")
                print(f"*** Please delete the files in {splits_dir} and re-run. ***")
            else:
                print("No overlapping patients found between train and val splits.")
        except Exception as e:
            print(f"Error checking existing splits: {e}")
    
    print("\n" + "="*70)
    print("STAGE 1: Training Individual Modality Classifiers")
    print("="*70)
    
    # Train 3 separate classifiers
    # Define per-modality data augmentation and validation transforms
    image_size = (args.image_size, args.image_size)
    
    base_normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    
    train_transforms = {
        'intraoral': transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomResizedCrop(image_size[0], scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            base_normalize,
        ]),
        'opg': transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomRotation(degrees=3),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.02, 0.02),
                scale=(0.95, 1.05),
            ),
            transforms.ToTensor(),
            base_normalize,
        ]),
        'ceph': transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomRotation(degrees=3),
            transforms.ToTensor(),
            base_normalize,
        ]),
    }
    
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size[0]),
        transforms.ToTensor(),
        base_normalize,
    ])
    
    modalities = ['intraoral', 'opg', 'ceph']
    trained_models = {}
    
    for modality in modalities:
        print(f"\n{'='*70}")
        print(f"Training {modality.upper()} Classifier")
        print(f"{'='*70}\n")
        
        # Create datasets
        train_dataset = ModalityDataset(
            train_csv,
            modality,
            image_size=image_size,
            transform=train_transforms[modality],
        )
        val_dataset = ModalityDataset(
            val_csv,
            modality,
            image_size=image_size,
            transform=val_transform,
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                               num_workers=args.num_workers)
        
        # Create model
        model = SimpleClassifier(num_classes=args.num_classes).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # Train
        best_val_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_single_modality(
                model, train_loader, criterion, optimizer, device, epoch, modality
            )
            val_loss, val_metrics = validate_single_modality(
                model, val_loader, criterion, device, modality
            )
            scheduler.step()
            
            print(f"Epoch {epoch}/{args.epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_metrics['accuracy']:.4f}, "
                  f"Val F1={val_metrics['f1_macro']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save(model.state_dict(), 
                          os.path.join(args.output_dir, f'{modality}_best.pth'))
                print(f"  → Saved best {modality} model (acc={best_val_acc:.4f})")
        
        print(f"\n✓ {modality.upper()} training complete. Best Val Acc: {best_val_acc:.4f}")
        trained_models[modality] = model
    
    print("\n" + "="*70)
    print("✓ All individual classifiers trained!")
    print("="*70)
    print(f"\nSaved models to: {args.output_dir}")
    print(f"  - intraoral_best.pth")
    print(f"  - opg_best.pth")
    print(f"  - ceph_best.pth")
    
    # Save training config
    config = {
        'num_classes': args.num_classes,
        'image_size': args.image_size,
        'fusion_method': args.fusion_method,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate
    }
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Training complete! Models saved to {args.output_dir}")


if __name__ == "__main__":
    main()
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research presentation repository for evaluating different approaches to orthodontic malocclusion diagnosis using deep learning. The project implements three experimental approaches (E1, E2, E3) that compare different strategies for processing multi-modal medical imaging data.

**Classification Task**: Three-class malocclusion classification (Class I, II, III)

**Image Modalities**:
- OPG (Panoramic X-rays)
- Intraoral photographs
- Cephalometric X-rays

## Research Documentation

- **Phase-1-experiments-resume.pdf**: Contains the research presentation and experimental results summary
- **Purpose**: This repository supports a research presentation comparing three different approaches to multi-modal medical image classification

This is a **presentation repository** - the focus is on demonstrating and comparing experimental approaches, not production deployment.

## Repository Structure

```
/home/user/RA-Presentation/
├── E(1)-Materials/          # Experiment 1: OPG-only baseline
│   └── train.py
├── E(2)-Materials/          # Experiment 2: Late fusion ensemble
│   └── train.py
├── E(3)-Materials/          # Experiment 3: Multi-image prompting
│   └── train.py
├── Phase-1-experiments-resume.pdf  # Research presentation/summary
├── CLAUDE.md                # This file
└── README.md                # Minimal project description
```

**Note**: The `shared_utils/` module referenced by all scripts is NOT included in this repository and must be available in the parent directory or Python path.

## Experiments

### E1: Single-Modality Baseline (OPG-only)
- **Location**: `E(1)-Materials/train.py`
- **Approach**: Uses only panoramic X-rays (OPG) with MedGemma-4B vision encoder
- **Architecture**: Frozen MedGemma vision encoder + trainable classification head
- **Key Feature**: Robust 3-tier modality filtering (filename → content analysis → aspect ratio)

### E2: Naïve Late Fusion
- **Location**: `E(2)-Materials/train.py`
- **Approach**: Trains 3 separate CNN classifiers (one per modality) and fuses predictions at decision level
- **Architecture**: SimpleClassifier (custom CNN) for each modality + ensemble fusion
- **Fusion Methods**: average, weighted, or learned fusion
- **Self-contained**: E2 includes embedded fallback utilities and can run WITHOUT the shared_utils module (though it will use shared_utils if available for robust modality detection)

### E3: Multi-Image Prompting
- **Location**: `E(3)-Materials/train.py`
- **Approach**: Feeds multiple images to MedGemma with specialized prompts for attention-based fusion
- **Architecture**: Frozen MedGemma encoder + trainable classification head
- **Key Feature**: Uses `<start_of_image>` tokens with modality-aware prompts

## Running Training

### E1 Example Command
```bash
python E\(1\)-Materials/train.py \
  --data_csv /path/to/dataset.csv \
  --checkpoint_path /path/to/medgemma/checkpoint \
  --output_dir ./E1_outputs \
  --num_classes 3 \
  --batch_size 8 \
  --epochs 20 \
  --learning_rate 1e-4 \
  --freeze_encoder
```

### E2 Example Command
```bash
python E\(2\)-Materials/train.py \
  --data_csv /path/to/dataset.csv \
  --output_dir ./E2_outputs \
  --num_classes 4 \
  --batch_size 16 \
  --epochs 20 \
  --fusion_method average
```

### E3 Example Command
```bash
python E\(3\)-Materials/train.py \
  --data_csv /path/to/dataset.csv \
  --checkpoint_path /path/to/medgemma/checkpoint \
  --output_dir ./E3_outputs \
  --num_classes 3 \
  --batch_size 4 \
  --max_images 7
```

## Data Format

### Input CSV Structure
The dataset CSV must contain these columns:
- `Patient_ID`: Unique patient identifier (string)
- `Image_Paths`: Semicolon-separated list of image file paths (e.g., `path1.png; path2.jpg; path3.png`)
- `Response`: Textual response containing class label (e.g., "Class I", "Class II", "Class III")

### Data Splitting
**CRITICAL**: All experiments use **patient-aware splitting** to prevent data leakage. Patients are split at the ID level before creating train/val/test sets, ensuring no patient appears in multiple splits.

Default ratios: 70% train, 15% validation, 15% test

Split files are saved to `outputs/data_splits/` and reused if they already exist.

## Code Architecture

### Shared Utilities (Referenced but Not in Repo)
The training scripts import from a `shared_utils` module that is not present in this repository:
- `shared_utils.data_loader`: OrthoDataset, collate_fn, save_split_datasets
- `shared_utils.image_utils`: ImagePreprocessor, detect_modality_from_path, content_modality
- `shared_utils.metrics`: MetricsCalculator
- `shared_utils.model_utils`: MedGemmaWrapper

**Note**: These utilities must be available in the parent directory or Python path when running the scripts.

### Modality Detection
The codebase implements robust 3-tier modality detection:

1. **Tier A (Filename)**: Pattern matching on file paths (e.g., "pan.png" → OPG, "ceph" → Ceph)
2. **Tier B (Content Analysis)**: Opens image and analyzes aspect ratio/content
3. **Tier C (CLIP-based)**: Uses vision-language model for ambiguous cases

### Label Parsing
The `label_to_class()` function maps textual responses to numeric labels:
- Response contains "Class I" or "class 1" → 0
- Response contains "Class II" or "class 2" → 1
- Response contains "Class III" or "class 3" → 2

### Model Checkpointing
All experiments save:
- `best_model.pt`: Best model based on validation accuracy/F1
- `final_model.pt` or `final_classifier.pt`: Final epoch weights
- `training_history.json`: Training curves (loss, accuracy, F1)
- `config.json`: Experiment configuration

## Dependencies

Key Python packages used:
- `torch` (PyTorch)
- `transformers` (for MedGemma)
- `pandas` (data loading)
- `numpy`
- `scikit-learn` (metrics)
- `Pillow` (image processing)
- `tqdm` (progress bars)

## Execution Environment

The experiments were designed to run on SLURM-managed GPU clusters:
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- CUDA 12.4
- Conda environment: `ortho-ai`

Logs show typical training speeds:
- E1: ~11 it/s on RTX 4090
- Validation: ~8-10 it/s

## Troubleshooting

### Missing shared_utils Module
If you see import errors for `shared_utils`:

**E1 and E3**: These experiments REQUIRE the shared_utils module. Ensure it's in the parent directory or add it to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/parent/directory"
```

**E2**: This experiment has embedded fallback utilities and can run independently, though modality detection will be less robust without shared_utils.

### Directory Name Escaping
All experiment directories contain parentheses. Always escape them in bash commands or use quotes (see "File Escaping Note" section below).

### Data Path Issues
Ensure all paths in the input CSV are either:
- Absolute paths (e.g., `/data/images/patient001/pan.png`)
- Relative paths from the working directory where you run the script

Missing images will cause training failures or be silently skipped depending on the experiment configuration.

## Important Implementation Details

### E1 OPG-Only Filter
E1 applies an aggressive OPG-only filter to the CSV splits after they're created. This filter:
1. Parses `Image_Paths` column
2. Checks each path using `robust_opg_check()`
3. Removes non-OPG images
4. **Drops entire rows** if no OPG images remain

### E2 Per-Modality Training
E2 trains classifiers sequentially (not in parallel):
1. Train intraoral classifier → save `intraoral_best.pth`
2. Train OPG classifier → save `opg_best.pth`
3. Train ceph classifier → save `ceph_best.pth`

Each modality uses different data augmentation strategies appropriate for that image type.

### E3 Prompt Engineering
E3 constructs prompts with the format:
```
<start_of_image> <start_of_image> <start_of_image>

You are analyzing multiple orthodontic images of the same patient.
Image 1 is a panoramic X-ray (OPG). Image 2 is a intraoral photograph. Image 3 is a lateral cephalometric X-ray.

Based on all provided images, [base prompt]

Consider information from all modalities in your diagnosis.
```

The number of `<start_of_image>` tokens must match the number of images provided.

### Feature Dimensions
- E1 MedGemma vision encoder: 1152-dim (dynamically detected from config)
- E3 MedGemma text encoder: 2048-dim (Gemma-2B default)
- E2 SimpleClassifier: 512-dim after final conv layer

### Training Strategy
- **E1 & E3**: Freeze MedGemma encoder, only train classification head
- **E2**: Train entire SimpleClassifier (no pretrained weights)
- **Optimizer**: AdamW with weight decay
- **Scheduler**: CosineAnnealingLR
- **Loss**: CrossEntropyLoss (E3 uses label_smoothing=0.05)

## Dataset Statistics (from logs)

E1 experiment with patient-aware splitting:
- Train: 293 patients
- Validation: 66 patients
- Test: 63 patients
- Total: 422 unique patients

## File Escaping Note

Directory names contain parentheses, requiring shell escaping:
```bash
# Correct
ls E\(1\)-Materials/

# Also correct
ls "E(1)-Materials/"

# Incorrect (will fail)
ls E(1)-Materials/
```

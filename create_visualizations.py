"""
Create visualizations for all three experiments using log data.
Generates loss and accuracy vs epoch graphs for E1, E2, and E3.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def parse_e1_log(log_path):
    """Parse E1 (OPG-only) log file."""
    with open(log_path, 'r') as f:
        content = f.read()

    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    val_f1 = []

    # Pattern: Train Loss: 0.8864, Train Acc: 0.5290
    train_pattern = r'Train Loss: ([\d.]+), Train Acc: ([\d.]+)'
    # Pattern: Val Loss: 0.8422, Val Acc: 0.6667, Val F1: 0.2667
    val_pattern = r'Val Loss: ([\d.]+), Val Acc: ([\d.]+), Val F1: ([\d.]+)'

    train_matches = re.findall(train_pattern, content)
    val_matches = re.findall(val_pattern, content)

    for i, (t_match, v_match) in enumerate(zip(train_matches, val_matches), 1):
        epochs.append(i)
        train_loss.append(float(t_match[0]))
        train_acc.append(float(t_match[1]))
        val_loss.append(float(v_match[0]))
        val_acc.append(float(v_match[1]))
        val_f1.append(float(v_match[2]))

    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_f1': val_f1
    }


def parse_e2_log(log_path):
    """Parse E2 (Late Fusion) log file."""
    with open(log_path, 'r') as f:
        content = f.read()

    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    val_f1 = []

    # Pattern: Epoch 1/45: Train Loss=0.5144, Train Acc=0.9143 | Val Loss=0.4111, Val Acc=0.9351, Val F1=0.4832
    pattern = r'Epoch (\d+)/\d+: Train Loss=([\d.]+), Train Acc=([\d.]+) \| Val Loss=([\d.]+), Val Acc=([\d.]+), Val F1=([\d.]+)'

    matches = re.findall(pattern, content)

    for match in matches:
        epochs.append(int(match[0]))
        train_loss.append(float(match[1]))
        train_acc.append(float(match[2]))
        val_loss.append(float(match[3]))
        val_acc.append(float(match[4]))
        val_f1.append(float(match[5]))

    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_f1': val_f1
    }


def parse_e3_log(log_path):
    """Parse E3 (Multi-Image Prompting) log file."""
    with open(log_path, 'r') as f:
        content = f.read()

    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    val_f1 = []

    # Pattern: Train Loss: 1.0388, Train Acc: 0.5429
    train_pattern = r'Train Loss: ([\d.]+), Train Acc: ([\d.]+)'
    # Pattern: Val Loss: 0.8591, Val Acc: 0.7273, Val F1: 0.3243
    val_pattern = r'Val Loss: ([\d.]+), Val Acc: ([\d.]+), Val F1: ([\d.]+)'

    train_matches = re.findall(train_pattern, content)
    val_matches = re.findall(val_pattern, content)

    for i, (t_match, v_match) in enumerate(zip(train_matches, val_matches), 1):
        epochs.append(i)
        train_loss.append(float(t_match[0]))
        train_acc.append(float(t_match[1]))
        val_loss.append(float(v_match[0]))
        val_acc.append(float(v_match[1]))
        val_f1.append(float(v_match[2]))

    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_f1': val_f1
    }


def plot_experiment(data, exp_name, output_dir):
    """Create individual plots for a single experiment."""

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss
    ax1.plot(data['epochs'], data['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    ax1.plot(data['epochs'], data['val_loss'], 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title(f'{exp_name}: Loss vs Epoch', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy
    ax2.plot(data['epochs'], data['train_acc'], 'b-', label='Train Accuracy', linewidth=2, marker='o', markersize=4)
    ax2.plot(data['epochs'], data['val_acc'], 'r-', label='Val Accuracy', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title(f'{exp_name}: Accuracy vs Epoch', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.0])

    plt.tight_layout()

    # Save figure
    output_path = output_dir / f'{exp_name.replace(" ", "_").replace(":", "")}_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_comparison(e1_data, e2_data, e3_data, output_dir):
    """Create comparison plots for all three experiments."""

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Training Loss Comparison
    ax = axes[0, 0]
    ax.plot(e1_data['epochs'], e1_data['train_loss'], 'b-', label='E1: OPG-only', linewidth=2, marker='o', markersize=3)
    if e2_data['epochs']:
        # Limit to first 20 epochs for comparison
        e2_epochs = e2_data['epochs'][:20]
        e2_train_loss = e2_data['train_loss'][:20]
        ax.plot(e2_epochs, e2_train_loss, 'g-', label='E2: Late Fusion', linewidth=2, marker='s', markersize=3)
    ax.plot(e3_data['epochs'], e3_data['train_loss'], 'r-', label='E3: Multi-Image', linewidth=2, marker='^', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Validation Loss Comparison
    ax = axes[0, 1]
    ax.plot(e1_data['epochs'], e1_data['val_loss'], 'b-', label='E1: OPG-only', linewidth=2, marker='o', markersize=3)
    if e2_data['epochs']:
        e2_val_loss = e2_data['val_loss'][:20]
        ax.plot(e2_epochs, e2_val_loss, 'g-', label='E2: Late Fusion', linewidth=2, marker='s', markersize=3)
    ax.plot(e3_data['epochs'], e3_data['val_loss'], 'r-', label='E3: Multi-Image', linewidth=2, marker='^', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 3: Training Accuracy Comparison
    ax = axes[1, 0]
    ax.plot(e1_data['epochs'], e1_data['train_acc'], 'b-', label='E1: OPG-only', linewidth=2, marker='o', markersize=3)
    if e2_data['epochs']:
        e2_train_acc = e2_data['train_acc'][:20]
        ax.plot(e2_epochs, e2_train_acc, 'g-', label='E2: Late Fusion', linewidth=2, marker='s', markersize=3)
    ax.plot(e3_data['epochs'], e3_data['train_acc'], 'r-', label='E3: Multi-Image', linewidth=2, marker='^', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])

    # Plot 4: Validation Accuracy Comparison
    ax = axes[1, 1]
    ax.plot(e1_data['epochs'], e1_data['val_acc'], 'b-', label='E1: OPG-only', linewidth=2, marker='o', markersize=3)
    if e2_data['epochs']:
        e2_val_acc = e2_data['val_acc'][:20]
        ax.plot(e2_epochs, e2_val_acc, 'g-', label='E2: Late Fusion', linewidth=2, marker='s', markersize=3)
    ax.plot(e3_data['epochs'], e3_data['val_acc'], 'r-', label='E3: Multi-Image', linewidth=2, marker='^', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'all_experiments_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    # Define paths
    base_dir = Path(__file__).parent
    e1_log = base_dir / 'E(1)-Materials' / 'e1_opg_only_98684.log'
    e2_log = base_dir / 'E(2)-Materials' / 'slurm-98598.out'
    e3_log = base_dir / 'E(3)-Materials' / 'e3_multi_image_98696.log'

    output_dir = base_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Parsing experiment logs and creating visualizations")
    print("=" * 60)

    # Parse logs
    print("\n[1/3] Parsing E1 (OPG-only) log...")
    e1_data = parse_e1_log(e1_log)
    print(f"  - Found {len(e1_data['epochs'])} epochs")

    print("\n[2/3] Parsing E2 (Late Fusion) log...")
    e2_data = parse_e2_log(e2_log)
    print(f"  - Found {len(e2_data['epochs'])} epochs")

    print("\n[3/3] Parsing E3 (Multi-Image Prompting) log...")
    e3_data = parse_e3_log(e3_log)
    print(f"  - Found {len(e3_data['epochs'])} epochs")

    # Create individual plots
    print("\n" + "=" * 60)
    print("Creating individual experiment plots")
    print("=" * 60)

    plot_experiment(e1_data, 'E1: OPG-only Baseline', output_dir)
    plot_experiment(e2_data, 'E2: Late Fusion', output_dir)
    plot_experiment(e3_data, 'E3: Multi-Image Prompting', output_dir)

    # Create comparison plots
    print("\n" + "=" * 60)
    print("Creating comparison plots")
    print("=" * 60)

    plot_comparison(e1_data, e2_data, e3_data, output_dir)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    print("\nE1 (OPG-only):")
    print(f"  Best Val Acc: {max(e1_data['val_acc']):.4f} at epoch {e1_data['epochs'][e1_data['val_acc'].index(max(e1_data['val_acc']))]}")
    print(f"  Best Val F1:  {max(e1_data['val_f1']):.4f} at epoch {e1_data['epochs'][e1_data['val_f1'].index(max(e1_data['val_f1']))]}")
    print(f"  Final Val Acc: {e1_data['val_acc'][-1]:.4f}")

    print("\nE2 (Late Fusion):")
    print(f"  Best Val Acc: {max(e2_data['val_acc']):.4f} at epoch {e2_data['epochs'][e2_data['val_acc'].index(max(e2_data['val_acc']))]}")
    print(f"  Best Val F1:  {max(e2_data['val_f1']):.4f} at epoch {e2_data['epochs'][e2_data['val_f1'].index(max(e2_data['val_f1']))]}")
    print(f"  Final Val Acc: {e2_data['val_acc'][-1]:.4f}")

    print("\nE3 (Multi-Image Prompting):")
    print(f"  Best Val Acc: {max(e3_data['val_acc']):.4f} at epoch {e3_data['epochs'][e3_data['val_acc'].index(max(e3_data['val_acc']))]}")
    print(f"  Best Val F1:  {max(e3_data['val_f1']):.4f} at epoch {e3_data['epochs'][e3_data['val_f1'].index(max(e3_data['val_f1']))]}")
    print(f"  Final Val Acc: {e3_data['val_acc'][-1]:.4f}")

    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

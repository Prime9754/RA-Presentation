"""
Create a clean architecture comparison table without overlapping elements.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

fig = plt.figure(figsize=(18, 12))
ax = fig.add_subplot(111)
ax.axis('off')

# Title
fig.text(0.5, 0.97, 'Architecture Comparison: Three Experimental Approaches',
         ha='center', fontsize=22, fontweight='bold')

# Define comparison data
comparison_data = {
    'Aspect': [
        'Approach',
        'Model Backbone',
        'Input Modalities',
        'Feature Dimensions',
        'Trainable Components',
        'Frozen Components',
        'Training Strategy',
        'Batch Size',
        'Epochs',
        'Optimizer',
        'Learning Rate',
        'Scheduler',
        'Loss Function',
        'Key Innovation',
        'Val Accuracy',
        'Val F1 Score',
        'Training Time',
        'Complexity'
    ],
    'E1: OPG-only Baseline': [
        'Single-Modality',
        'MedGemma-4B Vision',
        'OPG only',
        '1152-dim',
        'Classification head',
        'MedGemma encoder',
        'End-to-end',
        '8',
        '20',
        'AdamW',
        '1e-4',
        'CosineAnnealingLR',
        'CrossEntropyLoss',
        '3-tier modality filter',
        '66.67%',
        '0.3246',
        '~30 min',
        'Low'
    ],
    'E2: Late Fusion': [
        'Multi-Modal Ensemble',
        '3× SimpleClassifier CNNs',
        'Intraoral + OPG + Ceph',
        '512-dim (per CNN)',
        'All 3 CNNs + fusion',
        'None (train from scratch)',
        'Sequential per modality',
        '16',
        '45 (per modality)',
        'AdamW',
        'Custom per modality',
        'CosineAnnealingLR',
        'CrossEntropyLoss',
        'Decision-level fusion',
        '93.51%',
        '0.4832',
        '~2 hours',
        'Medium'
    ],
    'E3: Multi-Image Prompting': [
        'Multi-Modal VLM',
        'MedGemma-4B (Vision+Text)',
        'Intraoral + OPG + Ceph',
        '2048-dim (text)',
        'Classification head',
        'MedGemma encoder',
        'Multi-image prompting',
        '4',
        '20',
        'AdamW',
        '1e-4',
        'CosineAnnealingLR',
        'CrossEntropyLoss + LS',
        'Modality-aware prompts',
        '72.73%',
        '0.3924',
        '~50 min',
        'High'
    ]
}

# Colors
header_color = '#4472C4'
row_colors = ['#F2F2F2', '#FFFFFF']
highlight_color = '#90EE90'
aspect_color = '#E7E6E6'

# Table dimensions
cell_height = 0.038
cell_width = 0.23
start_y = 0.90
start_x = 0.05

# Draw headers
headers = ['Aspect', 'E1: OPG-only', 'E2: Late Fusion', 'E3: Multi-Image']
for i, header in enumerate(headers):
    # Header box
    box = FancyBboxPatch(
        (start_x + i * cell_width, start_y),
        cell_width, cell_height,
        boxstyle="round,pad=0.005",
        facecolor=header_color,
        edgecolor='black',
        linewidth=1.5,
        transform=fig.transFigure
    )
    fig.add_artist(box)

    # Header text
    fig.text(
        start_x + i * cell_width + cell_width/2,
        start_y + cell_height/2,
        header,
        ha='center', va='center',
        fontsize=13, fontweight='bold',
        color='white',
        transform=fig.transFigure
    )

# Draw data rows
for row_idx, aspect in enumerate(comparison_data['Aspect']):
    y_pos = start_y - (row_idx + 1) * cell_height

    # Aspect column
    box_color = aspect_color
    box = FancyBboxPatch(
        (start_x, y_pos),
        cell_width, cell_height,
        boxstyle="round,pad=0.005",
        facecolor=box_color,
        edgecolor='gray',
        linewidth=0.5,
        transform=fig.transFigure
    )
    fig.add_artist(box)

    fig.text(
        start_x + cell_width/2,
        y_pos + cell_height/2,
        aspect,
        ha='center', va='center',
        fontsize=10, fontweight='bold',
        transform=fig.transFigure
    )

    # Data columns
    for col_idx, exp_name in enumerate(['E1: OPG-only Baseline', 'E2: Late Fusion', 'E3: Multi-Image Prompting']):
        x_pos = start_x + (col_idx + 1) * cell_width
        value = comparison_data[exp_name][row_idx]

        # Highlight best performance
        if aspect == 'Val Accuracy' and '93.51%' in value:
            box_color = highlight_color
        elif aspect == 'Val F1 Score' and value == '0.4832':
            box_color = highlight_color
        else:
            box_color = row_colors[row_idx % 2]

        box = FancyBboxPatch(
            (x_pos, y_pos),
            cell_width, cell_height,
            boxstyle="round,pad=0.005",
            facecolor=box_color,
            edgecolor='gray',
            linewidth=0.5,
            transform=fig.transFigure
        )
        fig.add_artist(box)

        # Adjust font size for long text
        font_size = 9 if len(str(value)) > 25 else 10

        fig.text(
            x_pos + cell_width/2,
            y_pos + cell_height/2,
            value,
            ha='center', va='center',
            fontsize=font_size,
            transform=fig.transFigure
        )

# Add legend - positioned below the table with more space
legend_y = start_y - (len(comparison_data['Aspect']) + 1) * cell_height - 0.04
fig.text(
    0.5, legend_y,
    'Best performer  |  Green highlighting indicates best performance metrics',
    fontsize=11, style='italic', ha='center',
    transform=fig.transFigure,
    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8, edgecolor='gray')
)

# Add footer notes - positioned at the bottom
footer_text = """Notes:
• E1 uses only panoramic X-rays with frozen MedGemma encoder
• E2 trains separate CNNs for each modality and fuses predictions at decision level
• E3 uses multi-image prompting with frozen MedGemma encoder
• All experiments use patient-aware splitting (70/15/15) to prevent data leakage
• LS = Label Smoothing (0.05)"""

fig.text(
    0.5, 0.03,
    footer_text,
    fontsize=9,
    ha='center',
    verticalalignment='bottom',
    transform=fig.transFigure,
    bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5, edgecolor='gray')
)

plt.savefig('visualizations/architecture_comparison_table.png',
            dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
print("✓ Saved: visualizations/architecture_comparison_table.png")

plt.close()
print("\n✓ Clean architecture comparison table created successfully!")

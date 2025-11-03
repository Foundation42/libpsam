#!/usr/bin/env python3
"""
Visualize PSAM Composition Weight Sweep Results

Creates plots showing how vocabulary distribution changes with layer weights.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
import argparse


def create_ascii_plot(data_points, width=60, height=20):
    """
    Create an ASCII line plot.
    
    Args:
        data_points: List of (x, y_dict) where y_dict maps series_name -> y_value
        width: Width of plot in characters
        height: Height of plot in characters
    """
    if not data_points:
        return "No data to plot"
    
    # Extract all series names
    all_series = set()
    for _, y_dict in data_points:
        all_series.update(y_dict.keys())
    
    series_list = sorted(all_series)
    
    # Find min/max for scaling
    x_values = [x for x, _ in data_points]
    y_values = []
    for _, y_dict in data_points:
        y_values.extend(y_dict.values())
    
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    
    # Add padding
    y_range = y_max - y_min
    y_min -= y_range * 0.1
    y_max += y_range * 0.1
    
    # Create canvas
    canvas = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Plot each series
    markers = ['*', '+', 'x', 'o', '#']
    
    for series_idx, series_name in enumerate(series_list):
        marker = markers[series_idx % len(markers)]
        
        for x, y_dict in data_points:
            if series_name not in y_dict:
                continue
            
            y = y_dict[series_name]
            
            # Scale to canvas coordinates
            canvas_x = int((x - x_min) / (x_max - x_min) * (width - 1))
            canvas_y = height - 1 - int((y - y_min) / (y_max - y_min) * (height - 1))
            
            if 0 <= canvas_x < width and 0 <= canvas_y < height:
                canvas[canvas_y][canvas_x] = marker
    
    # Build plot string
    lines = []
    
    # Y-axis labels and plot
    for i, row in enumerate(canvas):
        y_val = y_max - (i / (height - 1)) * (y_max - y_min)
        label = f"{y_val:6.1f} |"
        lines.append(label + ''.join(row))
    
    # X-axis
    x_axis = " " * 8 + "+" + "-" * (width - 1)
    lines.append(x_axis)
    
    # X-axis labels
    x_label = " " * 8
    for i in range(0, width, width // 5):
        x_val = x_min + (i / width) * (x_max - x_min)
        x_label += f"{x_val:6.2f}".ljust(width // 5)
    lines.append(x_label)
    
    # Legend
    lines.append("")
    lines.append("Legend:")
    for series_idx, series_name in enumerate(series_list):
        marker = markers[series_idx % len(markers)]
        lines.append(f"  {marker} = {series_name}")
    
    return '\n'.join(lines)


def load_sweep_results(results_dir: Path):
    """
    Load all sweep results from the results directory.
    
    Returns:
        List of (weight_a, vocab_distributions) tuples
    """
    results = []
    
    for result_file in sorted(results_dir.glob('sweep_*_predictions.txt')):
        # Parse filename to get weights
        parts = result_file.stem.split('_')
        if len(parts) < 3:
            continue
        
        try:
            weight_a = float(parts[1])
        except ValueError:
            continue
        
        # Parse predictions
        vocab_mass = defaultdict(float)
        
        with open(result_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    token = parts[0]
                    try:
                        prob = float(parts[1])
                        
                        # Determine vocab
                        vocab = token_to_vocab(token)
                        vocab_mass[vocab] += prob
                    except ValueError:
                        continue
        
        results.append((weight_a, dict(vocab_mass)))
    
    return results


def token_to_vocab(token: str) -> str:
    """Determine which vocabulary set a token belongs to"""
    VOCAB_SETS = {
        'A': ['apple', 'ant', 'arrow', 'anchor', 'atlas', 'axe', 'angel', 'arch'],
        'B': ['ball', 'bat', 'bear', 'boat', 'bell', 'bird', 'bone', 'bread'],
        'C': ['cat', 'car', 'cave', 'coin', 'crown', 'cloud', 'cup', 'cliff'],
        'D': ['dog', 'drum', 'duck', 'door', 'desk', 'dirt', 'dove', 'dune']
    }
    
    for vocab_name, vocab_tokens in VOCAB_SETS.items():
        if token in vocab_tokens:
            return vocab_name
    return 'UNKNOWN'


def main():
    parser = argparse.ArgumentParser(description='Visualize composition weight sweep')
    parser.add_argument('--results-dir', default='composition_test_results',
                        help='Directory containing sweep results')
    parser.add_argument('--output', help='Save plot to file')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    # Load results
    results = load_sweep_results(results_dir)
    
    if not results:
        print("No sweep results found")
        return 1
    
    print("\nWeight Sweep Visualization")
    print("=" * 70)
    
    # Normalize to percentages
    normalized_results = []
    for weight_a, vocab_mass in results:
        total = sum(vocab_mass.values())
        normalized = {k: (v / total * 100) for k, v in vocab_mass.items()}
        normalized_results.append((weight_a, normalized))
    
    # Create plot
    plot = create_ascii_plot(normalized_results, width=60, height=20)
    
    print("\nVocabulary Distribution vs Weight")
    print("(X-axis: Weight of Layer A, Y-axis: Probability Mass %)")
    print()
    print(plot)
    
    # Summary table
    print("\n\nDetailed Results:")
    print("-" * 70)
    print("Weight A | Weight B | Vocab A% | Vocab B% | Expected A% | Δ")
    print("-" * 70)
    
    for weight_a, vocab_pcts in normalized_results:
        weight_b = 1.0 - weight_a
        pct_a = vocab_pcts.get('A', 0)
        pct_b = vocab_pcts.get('B', 0)
        
        # Calculate expected percentage (assuming linear blending)
        total_weight = weight_a + weight_b
        expected_a = (weight_a / total_weight * 100) if total_weight > 0 else 0
        
        delta = pct_a - expected_a
        
        status = "✓" if abs(delta) < 15 else "✗"
        
        print(f"  {weight_a:5.2f}  |   {weight_b:5.2f}  |  {pct_a:6.1f}% |  {pct_b:6.1f}% |    {expected_a:6.1f}% | {delta:+6.1f}% {status}")
    
    print("-" * 70)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(plot)
        print(f"\nPlot saved to: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
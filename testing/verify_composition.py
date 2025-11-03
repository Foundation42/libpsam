#!/usr/bin/env python3
"""
Composition Test Verification Script

Analyze predictions from composed models and verify they match expected behavior.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse


# Vocabulary definitions (must match synthetic_composition_test.py)
VOCAB_SETS = {
    'A': ['apple', 'ant', 'arrow', 'anchor', 'atlas', 'axe', 'angel', 'arch'],
    'B': ['ball', 'bat', 'bear', 'boat', 'bell', 'bird', 'bone', 'bread'],
    'C': ['cat', 'car', 'cave', 'coin', 'crown', 'cloud', 'cup', 'cliff'],
    'D': ['dog', 'drum', 'duck', 'door', 'desk', 'dirt', 'dove', 'dune']
}


def token_to_vocab(token: str) -> str:
    """Determine which vocabulary set a token belongs to"""
    for vocab_name, vocab_tokens in VOCAB_SETS.items():
        if token in vocab_tokens:
            return vocab_name
    return 'UNKNOWN'


def analyze_predictions(predictions: List[Tuple[str, float]]) -> Dict[str, float]:
    """
    Analyze predictions and compute probability mass per vocabulary.
    
    Args:
        predictions: List of (token, probability) tuples
    
    Returns:
        Dict mapping vocab_name -> total probability mass
    """
    vocab_mass = defaultdict(float)
    
    for token, prob in predictions:
        vocab = token_to_vocab(token)
        vocab_mass[vocab] += prob
    
    return dict(vocab_mass)


def parse_psam_predictions(output: str) -> List[Tuple[str, float]]:
    """
    Parse PSAM prediction output.
    
    Expected format (example):
    ant     0.4523
    arrow   0.2341
    apple   0.1234
    ...
    """
    predictions = []
    
    for line in output.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split()
        if len(parts) >= 2:
            token = parts[0]
            try:
                prob = float(parts[1])
                predictions.append((token, prob))
            except ValueError:
                continue
    
    return predictions


def print_analysis(predictions: List[Tuple[str, float]], 
                   expected_distribution: Dict[str, float] = None):
    """
    Print detailed analysis of predictions.
    
    Args:
        predictions: List of (token, probability) tuples
        expected_distribution: Optional dict of expected vocab distributions
    """
    vocab_mass = analyze_predictions(predictions)
    
    print("\n" + "=" * 60)
    print("PREDICTION ANALYSIS")
    print("=" * 60)
    
    print("\nTop Predictions:")
    for token, prob in predictions[:10]:
        vocab = token_to_vocab(token)
        print(f"  {token:12s} {prob:8.4f}  [{vocab}]")
    
    print("\nProbability Mass by Vocabulary:")
    total_mass = sum(vocab_mass.values())
    
    for vocab_name in sorted(vocab_mass.keys()):
        mass = vocab_mass[vocab_name]
        percentage = (mass / total_mass * 100) if total_mass > 0 else 0
        
        bar_length = int(percentage / 2)  # Scale to fit terminal
        bar = "█" * bar_length
        
        print(f"  {vocab_name:8s}: {mass:6.4f} ({percentage:5.1f}%) {bar}")
    
    if expected_distribution:
        print("\nExpected vs Actual:")
        for vocab_name, expected_pct in expected_distribution.items():
            actual_pct = (vocab_mass.get(vocab_name, 0) / total_mass * 100) if total_mass > 0 else 0
            diff = actual_pct - expected_pct
            status = "✓" if abs(diff) < 10 else "✗"  # Within 10% tolerance
            print(f"  {status} {vocab_name}: Expected {expected_pct:5.1f}%, Got {actual_pct:5.1f}% (Δ {diff:+5.1f}%)")
    
    print("=" * 60 + "\n")
    
    return vocab_mass


def verify_test_case(test_name: str, predictions: List[Tuple[str, float]]) -> bool:
    """
    Verify predictions match expected behavior for a specific test case.
    
    Returns:
        True if test passes, False otherwise
    """
    expected_distributions = {
        'test_01_pure_dominance': {'A': 100.0, 'B': 0.0},
        'test_02_equal_blend': {'A': 50.0, 'B': 50.0},
        'test_03_weighted_blend': {'A': 70.0, 'B': 30.0},
        'test_04_three_layer': {'A': 50.0, 'B': 30.0, 'C': 20.0},
        'test_06_zero_weight': {'A': 100.0, 'B': 0.0},
    }
    
    if test_name not in expected_distributions:
        print(f"No expected distribution defined for {test_name}")
        return None
    
    expected = expected_distributions[test_name]
    vocab_mass = print_analysis(predictions, expected)
    
    # Check if actual distribution is within tolerance
    total_mass = sum(vocab_mass.values())
    passed = True
    
    for vocab_name, expected_pct in expected.items():
        actual_pct = (vocab_mass.get(vocab_name, 0) / total_mass * 100) if total_mass > 0 else 0
        diff = abs(actual_pct - expected_pct)
        
        if diff > 15:  # 15% tolerance
            passed = False
            print(f"FAIL: {vocab_name} distribution off by {diff:.1f}%")
    
    return passed


def compare_weight_sweep(results: List[Dict]):
    """
    Compare results from a weight sweep experiment.
    
    Args:
        results: List of dicts with keys: 'weight_a', 'weight_b', 'predictions'
    """
    print("\n" + "=" * 60)
    print("WEIGHT SWEEP ANALYSIS")
    print("=" * 60)
    
    print("\nWeight_A  Weight_B  |  Vocab A%  Vocab B%  |  Status")
    print("-" * 60)
    
    for result in results:
        weight_a = result['weight_a']
        weight_b = result['weight_b']
        predictions = result['predictions']
        
        vocab_mass = analyze_predictions(predictions)
        total = sum(vocab_mass.values())
        
        pct_a = (vocab_mass.get('A', 0) / total * 100) if total > 0 else 0
        pct_b = (vocab_mass.get('B', 0) / total * 100) if total > 0 else 0
        
        # Check if distribution matches weights (with tolerance)
        expected_a = (weight_a / (weight_a + weight_b)) * 100 if (weight_a + weight_b) > 0 else 0
        expected_b = (weight_b / (weight_a + weight_b)) * 100 if (weight_a + weight_b) > 0 else 0
        
        diff_a = abs(pct_a - expected_a)
        diff_b = abs(pct_b - expected_b)
        
        status = "✓" if (diff_a < 15 and diff_b < 15) else "✗"
        
        print(f"  {weight_a:5.2f}     {weight_b:5.2f}    |  {pct_a:6.1f}%  {pct_b:6.1f}%  | {status}")
    
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Verify PSAM composition test results')
    parser.add_argument('--test', help='Test case name (e.g., test_01_pure_dominance)')
    parser.add_argument('--input', help='File containing prediction output from PSAM')
    parser.add_argument('--stdin', action='store_true', help='Read predictions from stdin')
    
    args = parser.parse_args()
    
    # Read predictions
    if args.stdin:
        prediction_output = sys.stdin.read()
    elif args.input:
        with open(args.input) as f:
            prediction_output = f.read()
    else:
        print("Error: Must provide --input or --stdin")
        return 1
    
    # Parse predictions
    predictions = parse_psam_predictions(prediction_output)
    
    if not predictions:
        print("Error: No valid predictions found in input")
        return 1
    
    # Verify if test case provided
    if args.test:
        passed = verify_test_case(args.test, predictions)
        if passed is not None:
            return 0 if passed else 1
    else:
        # Just print analysis
        print_analysis(predictions)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
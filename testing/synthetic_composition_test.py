#!/usr/bin/env python3
"""
Synthetic Data Generator for PSAM Composition Testing

Creates ultra-simple vocabularies and training sequences to test
compositional behavior in a completely transparent way.
"""

import random
from pathlib import Path
from typing import List, Dict, Tuple
import json


class SimpleVocabGenerator:
    """Generate simple, non-overlapping vocabularies for testing"""
    
    # Ultra-simple vocab sets - completely disjoint
    VOCAB_SETS = {
        'A': ['apple', 'ant', 'arrow', 'anchor', 'atlas', 'axe', 'angel', 'arch'],
        'B': ['ball', 'bat', 'bear', 'boat', 'bell', 'bird', 'bone', 'bread'],
        'C': ['cat', 'car', 'cave', 'coin', 'crown', 'cloud', 'cup', 'cliff'],
        'D': ['dog', 'drum', 'duck', 'door', 'desk', 'dirt', 'dove', 'dune']
    }
    
    @staticmethod
    def get_vocab(name: str) -> List[str]:
        """Get a vocabulary set by name"""
        return SimpleVocabGenerator.VOCAB_SETS[name].copy()
    
    @staticmethod
    def create_unified_vocab(*vocab_names: str) -> List[str]:
        """Create a unified vocabulary from multiple sets"""
        unified = []
        for name in vocab_names:
            unified.extend(SimpleVocabGenerator.VOCAB_SETS[name])
        return unified


class PatternGenerator:
    """Generate training sequences with clear statistical patterns"""
    
    @staticmethod
    def repeating_pattern(vocab: List[str], pattern: List[int], length: int) -> List[str]:
        """
        Generate sequence following a repeating pattern.
        
        Args:
            vocab: List of tokens to use
            pattern: Indices into vocab, e.g., [0, 1, 0, 2] means vocab[0], vocab[1], vocab[0], vocab[2]
            length: Total number of tokens to generate
        
        Example:
            vocab = ['apple', 'ant', 'arrow']
            pattern = [0, 1, 0, 2]  # apple, ant, apple, arrow
            Generates: apple ant apple arrow apple ant apple arrow ...
        """
        sequence = []
        for i in range(length):
            sequence.append(vocab[pattern[i % len(pattern)]])
        return sequence
    
    @staticmethod
    def markov_simple(vocab: List[str], transitions: Dict[str, List[Tuple[str, float]]], 
                      length: int, seed: str = None) -> List[str]:
        """
        Generate sequence using simple Markov transitions.
        
        Args:
            vocab: List of tokens
            transitions: Dict mapping token -> [(next_token, probability), ...]
            length: Number of tokens to generate
            seed: Starting token (random if None)
        
        Example:
            transitions = {
                'apple': [('ant', 0.7), ('arrow', 0.3)],
                'ant': [('apple', 0.5), ('anchor', 0.5)],
                ...
            }
        """
        if seed is None:
            current = random.choice(vocab)
        else:
            current = seed
        
        sequence = [current]
        
        for _ in range(length - 1):
            if current in transitions:
                next_tokens, probs = zip(*transitions[current])
                current = random.choices(next_tokens, weights=probs)[0]
            else:
                current = random.choice(vocab)
            sequence.append(current)
        
        return sequence
    
    @staticmethod
    def biased_random(vocab: List[str], weights: List[float], length: int) -> List[str]:
        """
        Generate sequence with biased random selection.
        
        Args:
            vocab: List of tokens
            weights: Probability weight for each token
            length: Number of tokens to generate
        """
        return random.choices(vocab, weights=weights, k=length)
    
    @staticmethod
    def alternating(vocab_a: List[str], vocab_b: List[str], 
                   chunk_size: int, num_chunks: int) -> List[str]:
        """
        Alternate between two vocabularies in chunks.
        
        Useful for testing domain-switching behavior.
        """
        sequence = []
        for i in range(num_chunks):
            if i % 2 == 0:
                sequence.extend(random.choices(vocab_a, k=chunk_size))
            else:
                sequence.extend(random.choices(vocab_b, k=chunk_size))
        return sequence


class CompositionTestCase:
    """A complete test case with training data and expected behavior"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.layers = {}  # layer_name -> training_sequence
        self.prompts = []  # List of (prompt, expected_behavior)
    
    def add_layer(self, layer_name: str, vocab_name: str, sequence: List[str]):
        """Add a training sequence for a layer"""
        self.layers[layer_name] = {
            'vocab_name': vocab_name,
            'sequence': sequence
        }
    
    def add_test_prompt(self, prompt: str, expected: str):
        """Add a test prompt with expected behavior"""
        self.prompts.append({
            'prompt': prompt,
            'expected': expected
        })
    
    def save(self, output_dir: Path):
        """Save test case to disk"""
        test_dir = output_dir / self.name
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            'name': self.name,
            'description': self.description,
            'layers': list(self.layers.keys()),
            'prompts': self.prompts
        }
        
        with open(test_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save each layer's training data
        for layer_name, layer_data in self.layers.items():
            with open(test_dir / f'{layer_name}.txt', 'w') as f:
                f.write(' '.join(layer_data['sequence']))
        
        print(f"✓ Saved test case '{self.name}' to {test_dir}")


def create_test_suite() -> List[CompositionTestCase]:
    """Create the complete test suite"""
    
    tests = []
    
    # ================================================================
    # TEST 1: Pure Layer Dominance
    # ================================================================
    test1 = CompositionTestCase(
        name='test_01_pure_dominance',
        description='Verify that a layer with weight=1.0 completely dominates predictions'
    )
    
    vocab_a = SimpleVocabGenerator.get_vocab('A')
    vocab_b = SimpleVocabGenerator.get_vocab('B')
    
    # Layer A: Simple repeating pattern
    seq_a = PatternGenerator.repeating_pattern(vocab_a, [0, 1, 2, 0, 1, 2], 1000)
    test1.add_layer('layer_a', 'A', seq_a)
    
    # Layer B: Different pattern
    seq_b = PatternGenerator.repeating_pattern(vocab_b, [0, 2, 1, 0, 2, 1], 1000)
    test1.add_layer('layer_b', 'B', seq_b)
    
    test1.add_test_prompt(
        'apple',
        'Should predict ONLY A-vocabulary tokens (ant, arrow, etc). Zero B-vocab.'
    )
    test1.add_test_prompt(
        'apple ant',
        'Should continue A-vocabulary pattern'
    )
    
    tests.append(test1)
    
    # ================================================================
    # TEST 2: Equal Blend
    # ================================================================
    test2 = CompositionTestCase(
        name='test_02_equal_blend',
        description='Verify 50/50 blend shows equal representation from both layers'
    )
    
    # Same training data as test 1
    test2.add_layer('layer_a', 'A', seq_a)
    test2.add_layer('layer_b', 'B', seq_b)
    
    test2.add_test_prompt(
        'apple',
        'Should predict BOTH A and B vocabulary tokens with roughly equal probability'
    )
    test2.add_test_prompt(
        'ball',
        'Should predict from both vocabularies'
    )
    
    tests.append(test2)
    
    # ================================================================
    # TEST 3: Weighted Blend (70/30)
    # ================================================================
    test3 = CompositionTestCase(
        name='test_03_weighted_blend',
        description='Verify 70/30 blend shows proportional representation'
    )
    
    test3.add_layer('layer_a', 'A', seq_a)
    test3.add_layer('layer_b', 'B', seq_b)
    
    test3.add_test_prompt(
        'apple',
        'Should predict ~70% A-vocabulary, ~30% B-vocabulary tokens'
    )
    
    tests.append(test3)
    
    # ================================================================
    # TEST 4: Three-Layer Blend
    # ================================================================
    test4 = CompositionTestCase(
        name='test_04_three_layer',
        description='Verify three-layer composition with weights (0.5, 0.3, 0.2)'
    )
    
    vocab_c = SimpleVocabGenerator.get_vocab('C')
    seq_c = PatternGenerator.repeating_pattern(vocab_c, [0, 1, 0, 2, 3], 1000)
    
    test4.add_layer('layer_a', 'A', seq_a)
    test4.add_layer('layer_b', 'B', seq_b)
    test4.add_layer('layer_c', 'C', seq_c)
    
    test4.add_test_prompt(
        'apple',
        'Should predict ~50% A-vocab, ~30% B-vocab, ~20% C-vocab'
    )
    
    tests.append(test4)
    
    # ================================================================
    # TEST 5: Markov Transition Test
    # ================================================================
    test5 = CompositionTestCase(
        name='test_05_markov_transitions',
        description='Test blending with clear transition patterns'
    )
    
    # Layer A: Strong apple -> ant transition
    transitions_a = {
        'apple': [('ant', 0.9), ('arrow', 0.1)],
        'ant': [('arrow', 0.9), ('apple', 0.1)],
        'arrow': [('apple', 0.9), ('anchor', 0.1)],
        'anchor': [('apple', 0.7), ('atlas', 0.3)],
        'atlas': [('apple', 0.8), ('ant', 0.2)]
    }
    
    random.seed(42)
    seq_markov_a = PatternGenerator.markov_simple(vocab_a[:5], transitions_a, 1000, 'apple')
    
    # Layer B: Strong ball -> bat transition
    transitions_b = {
        'ball': [('bat', 0.9), ('bear', 0.1)],
        'bat': [('bear', 0.9), ('ball', 0.1)],
        'bear': [('ball', 0.9), ('boat', 0.1)],
        'boat': [('ball', 0.7), ('bell', 0.3)],
        'bell': [('ball', 0.8), ('bat', 0.2)]
    }
    
    random.seed(43)
    seq_markov_b = PatternGenerator.markov_simple(vocab_b[:5], transitions_b, 1000, 'ball')
    
    test5.add_layer('layer_a', 'A', seq_markov_a)
    test5.add_layer('layer_b', 'B', seq_markov_b)
    
    test5.add_test_prompt(
        'apple',
        'Pure A layer should strongly predict ant (0.9). Blended should show bat appearing too.'
    )
    test5.add_test_prompt(
        'ball',
        'Pure B layer should strongly predict bat (0.9). Blended should show ant appearing too.'
    )
    
    tests.append(test5)
    
    # ================================================================
    # TEST 6: Zero Weight Verification
    # ================================================================
    test6 = CompositionTestCase(
        name='test_06_zero_weight',
        description='Verify that layers with weight=0 have no effect'
    )
    
    test6.add_layer('layer_a', 'A', seq_a)
    test6.add_layer('layer_b', 'B', seq_b)
    
    test6.add_test_prompt(
        'apple',
        'With B-layer weight=0, should be identical to test_01 (pure A)'
    )
    
    tests.append(test6)
    
    # ================================================================
    # TEST 7: Domain Alternation
    # ================================================================
    test7 = CompositionTestCase(
        name='test_07_domain_alternation',
        description='Test handling of sequences that switch between domains'
    )
    
    # Create training data that alternates between domains
    seq_alternating = PatternGenerator.alternating(vocab_a, vocab_b, chunk_size=20, num_chunks=50)
    
    test7.add_layer('layer_mixed', 'A+B', seq_alternating)
    
    test7.add_test_prompt(
        'apple ant arrow',
        'Should continue in A-domain (we are in an A-chunk)'
    )
    test7.add_test_prompt(
        'ball bat bear',
        'Should continue in B-domain (we are in a B-chunk)'
    )
    
    tests.append(test7)
    
    # ================================================================
    # TEST 8: Uniform Distribution Baseline
    # ================================================================
    test8 = CompositionTestCase(
        name='test_08_uniform_baseline',
        description='Baseline test with uniform random sequences'
    )
    
    # Completely uniform - no patterns
    random.seed(100)
    seq_uniform_a = random.choices(vocab_a, k=1000)
    random.seed(101)
    seq_uniform_b = random.choices(vocab_b, k=1000)
    
    test8.add_layer('layer_a', 'A', seq_uniform_a)
    test8.add_layer('layer_b', 'B', seq_uniform_b)
    
    test8.add_test_prompt(
        'apple',
        'With uniform data, predictions should be roughly uniform across vocab'
    )
    
    tests.append(test8)
    
    return tests


def generate_vocab_files(output_dir: Path):
    """Generate vocabulary TSV files for PSAM"""
    
    vocab_dir = output_dir / 'vocabs'
    vocab_dir.mkdir(parents=True, exist_ok=True)
    
    # Individual vocab files
    for name, tokens in SimpleVocabGenerator.VOCAB_SETS.items():
        with open(vocab_dir / f'vocab_{name}.tsv', 'w') as f:
            for idx, token in enumerate(tokens):
                f.write(f'{idx}\t{token}\n')
    
    # Unified vocab files for different combinations
    combos = [
        ('A', 'B'),
        ('A', 'B', 'C'),
        ('A', 'B', 'C', 'D')
    ]
    
    for combo in combos:
        unified = SimpleVocabGenerator.create_unified_vocab(*combo)
        name = '_'.join(combo)
        with open(vocab_dir / f'vocab_{name}.tsv', 'w') as f:
            for idx, token in enumerate(unified):
                f.write(f'{idx}\t{token}\n')
    
    print(f"✓ Generated vocabulary files in {vocab_dir}")


def generate_readme(output_dir: Path):
    """Generate README explaining the test suite"""
    
    readme = """# PSAM Composition Test Suite - Synthetic Data

This test suite uses ultra-simple vocabularies to verify compositional behavior
in a completely transparent way.

## Vocabulary Sets

- **Vocab A**: apple, ant, arrow, anchor, atlas, axe, angel, arch
- **Vocab B**: ball, bat, bear, boat, bell, bird, bone, bread  
- **Vocab C**: cat, car, cave, coin, crown, cloud, cup, cliff
- **Vocab D**: dog, drum, duck, door, desk, dirt, dove, dune

All vocabularies are completely disjoint - no overlapping tokens.

## Test Cases

### Test 1: Pure Layer Dominance
Verify that a single layer with weight=1.0 completely dominates predictions.
- Train models A and B on their respective vocabularies
- Compose with weights (1.0, 0.0)
- Prompt: "apple"
- **Expected**: Should ONLY predict A-vocabulary tokens

### Test 2: Equal Blend  
Verify 50/50 blend shows equal representation.
- Same models as Test 1
- Compose with weights (0.5, 0.5)
- **Expected**: Roughly equal probability mass on A-vocab and B-vocab

### Test 3: Weighted Blend
Verify proportional blending with 70/30 weights.
- **Expected**: ~70% probability mass on A-vocab, ~30% on B-vocab

### Test 4: Three-Layer Blend
Test three-layer composition with weights (0.5, 0.3, 0.2).
- **Expected**: Probability mass proportional to weights

### Test 5: Markov Transitions
Test blending with clear state transition patterns.
- Layer A trained on: apple -> ant (0.9), ant -> arrow (0.9), etc.
- Layer B trained on: ball -> bat (0.9), bat -> bear (0.9), etc.
- **Expected**: Blending should mix both transition patterns

### Test 6: Zero Weight
Verify layers with weight=0 have no effect.
- **Expected**: Identical behavior to pure single-layer

### Test 7: Domain Alternation
Test sequences that alternate between domains.
- **Expected**: Model learns domain boundaries

### Test 8: Uniform Baseline
Baseline with completely uniform random sequences.
- **Expected**: Roughly uniform predictions

## Usage

1. **Generate synthetic data:**
   ```bash
   python synthetic_composition_test.py
   ```

2. **Train models** (example for test_01):
   ```bash
   psam train --input test_01_pure_dominance/layer_a.txt \\
              --vocab vocabs/vocab_A.tsv \\
              --output models/test01_layer_a.psam
   
   psam train --input test_01_pure_dominance/layer_b.txt \\
              --vocab vocabs/vocab_B.tsv \\
              --output models/test01_layer_b.psam
   ```

3. **Create composition:**
   ```bash
   psam compose --layer models/test01_layer_a.psam 1.0 \\
                --layer models/test01_layer_b.psam 0.0 \\
                --output compositions/test01_pure_a.psamc
   ```

4. **Test predictions:**
   ```bash
   psam predict --model compositions/test01_pure_a.psamc \\
                --prompt "apple" \\
                --top-k 10
   ```

5. **Verify behavior:**
   - Check that predictions match expected vocabulary distributions
   - Plot probability mass across vocabulary groups
   - Run weight sweeps and verify smooth transitions

## Verification Metrics

For each test, measure:
- **Vocabulary coverage**: % of predictions from each vocab set
- **Probability mass**: Sum of probabilities for each vocab group
- **Transition fidelity**: Whether learned patterns are preserved
- **Blend linearity**: Whether weight changes produce proportional effects

## Expected Results

If composition is working correctly:
- Pure layers (weight 1.0) → 100% predictions from that layer's vocab
- Equal blend (0.5, 0.5) → ~50/50 split
- Weighted blend (0.7, 0.3) → ~70/30 split
- Zero weight → Complete elimination of that layer's influence

Any deviation from these expectations indicates a bug in the composition logic.
"""
    
    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme)
    
    print(f"✓ Generated README.md")


def main():
    output_dir = Path('synthetic_test_suite')
    output_dir.mkdir(exist_ok=True)
    
    print("Generating PSAM Composition Test Suite...")
    print("=" * 60)
    
    # Generate vocabulary files
    generate_vocab_files(output_dir)
    
    # Generate test cases
    tests = create_test_suite()
    
    for test in tests:
        test.save(output_dir)
    
    # Generate documentation
    generate_readme(output_dir)
    
    print("=" * 60)
    print(f"\n✓ Complete! Generated {len(tests)} test cases in {output_dir}/")
    print("\nNext steps:")
    print("1. Train models for each layer using the generated .txt files")
    print("2. Create compositions using psam compose")
    print("3. Run predictions and verify behavior matches expectations")
    print("\nSee README.md for detailed instructions.")


if __name__ == '__main__':
    main()
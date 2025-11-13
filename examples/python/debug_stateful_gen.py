#!/usr/bin/env python3
"""
Debug stateful generator to see what's happening with residuals
"""

from psam import PSAM, PSAMGenerator, ResidualConfig, is_library_available
import numpy as np

def main():
    if not is_library_available():
        print("❌ Native library not available")
        return 1

    # Simple vocab
    vocab = {
        'what': 1, 'is': 2, 'the': 3, 'dog': 4, 'doing': 5, '?': 6,
        'rolling': 7, 'on': 8, 'rug': 9, '.': 10
    }
    reverse_vocab = {v: k for k, v in vocab.items()}

    psam = PSAM(vocab_size=20, window=8, top_k=20)

    # Train: "What is the dog doing? The dog is rolling on the rug."
    sequence = [1, 2, 3, 4, 5, 6, 3, 4, 2, 7, 8, 3, 9, 10]
    psam.train_batch(sequence)
    psam.finalize_training()

    print("=" * 70)
    print("STATEFUL GENERATOR DEBUG")
    print("=" * 70)

    residual_config = ResidualConfig(
        max_lookahead=5,
        residual_decay=0.85,
        residual_blend=0.6,
        enable=True
    )

    # Create stateful generator
    generator = PSAMGenerator(psam, residual_config=residual_config)

    # Start with question
    context = [1, 2, 3, 4, 5, 6]  # "what is the dog doing ?"
    print(f"\nStarting context: {' '.join([reverse_vocab[t] for t in context])}")
    print()

    # Generate step by step
    for step in range(5):
        print(f"Step {step + 1}:")
        print(f"  Context ({len(context)} tokens): {' '.join([reverse_vocab.get(t, f'<{t}>') for t in context])}")

        pred_ids, scores, _, _, _ = generator.predict(context, max_predictions=10)

        print(f"  Top 5 predictions:")
        for i in range(min(5, len(pred_ids))):
            word = reverse_vocab.get(pred_ids[i], f'<{pred_ids[i]}>')
            print(f"    {i+1}. '{word}' (score: {scores[i]:.2f})")

        if len(pred_ids) == 0:
            print("  No predictions!")
            break

        # Sample deterministically (top-1)
        next_token = pred_ids[0]
        word = reverse_vocab.get(next_token, f'<{next_token}>')
        print(f"  → Selected: '{word}'")
        print()

        context.append(next_token)

        if word == '.':
            break

    generated_text = ' '.join([reverse_vocab.get(t, f'<{t}>') for t in context])
    print(f"Final output: {generated_text}")

    generator.destroy()
    psam.destroy()
    return 0

if __name__ == "__main__":
    exit(main())

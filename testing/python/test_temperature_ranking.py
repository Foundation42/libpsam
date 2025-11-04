"""
Test to see if temperature SHOULD affect ranking.

With z-score normalization, we convert scores to a standard normal distribution.
Then temperature scaling divides by temperature before softmax.

If we have scores like [5.0, 0.0, -1.0] after z-score:
- temp=0.1: [50.0, 0.0, -10.0] -> very peaked on first
- temp=1.0: [5.0, 0.0, -1.0] -> moderate distribution
- temp=10.0: [0.5, 0.0, -0.1] -> very flat distribution

The ORDER shouldn't change just from temperature scaling... unless the original
scores are very close together. Let me check with actual PSAM scores.
"""
from __future__ import annotations

import re
from typing import List

import numpy as np

from psam import LogitTransform, PSAM, SamplerConfig


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[.,!?;]", text.lower())


def build_vocab(tokens):
    ordered = list(dict.fromkeys(tokens))
    token_to_id = {token: idx for idx, token in enumerate(ordered)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return token_to_id, id_to_token


def main():
    # Simple example where we can reason about the scores
    text = "a b c. a b d. a b c. a b d. a b e."

    tokens = tokenize(text)
    token_to_id, id_to_token = build_vocab(tokens)
    sequence = [token_to_id[token] for token in tokens]

    model = PSAM(vocab_size=len(token_to_id), window=8, top_k=32)
    model.train_batch(sequence)
    model.finalize_training()

    context = "a b"
    context_ids = [token_to_id[token] for token in tokenize(context)]

    print("Training data: a b c (2x), a b d (2x), a b e (1x)")
    print(f"Context: '{context}' -> expecting c and d to have similar scores")
    print("=" * 100)
    print()

    # Test different temperatures
    temperatures = [0.1, 0.5, 1.0, 2.0, 10.0]

    for temp in temperatures:
        sampler = SamplerConfig(
            transform=LogitTransform.ZSCORE,
            temperature=temp,
            top_k=0,
            top_p=1.0,  # No top-p filtering
            seed=1234,
        )

        ids, scores, raw_strengths, support_counts, probabilities = model.predict(
            context_ids[-model.window:],
            max_predictions=model.top_k,
            sampler=sampler,
        )

        print(f"Temperature: {temp}")
        print(f"  Token ranking (by returned order):")
        for i in range(min(10, len(ids))):
            token = id_to_token[ids[i]]
            print(f"    {i+1}. {token:<5}  prob={probabilities[i]:.6f}  score={scores[i]:8.4f}  raw={raw_strengths[i]:8.4f}  support={support_counts[i]}")

        # Now let's manually check if we re-sorted by probability
        sorted_indices = np.argsort(-np.array(probabilities))
        print(f"  Token ranking (if sorted by probability):")
        for i in range(min(10, len(ids))):
            idx = sorted_indices[i]
            token = id_to_token[ids[idx]]
            print(f"    {i+1}. {token:<5}  prob={probabilities[idx]:.6f}  score={scores[idx]:8.4f}  raw={raw_strengths[idx]:8.4f}  support={support_counts[idx]}")
        print()


if __name__ == "__main__":
    main()

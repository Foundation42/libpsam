"""
Test to understand how temperature affects the probability distribution.
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


def train_demo_model(text: str, window: int = 8, top_k: int = 32):
    tokens = tokenize(text)
    token_to_id, id_to_token = build_vocab(tokens)
    sequence = [token_to_id[token] for token in tokens]

    model = PSAM(vocab_size=len(token_to_id), window=window, top_k=top_k)
    model.train_batch(sequence)
    model.finalize_training()
    return model, token_to_id, id_to_token, sequence


def main():
    text = (
        "the cat sat on the mat. "
        "the dog sat on the rug. "
        "the bird sat on the branch. "
        "the frog sat on the log."
    )

    model, token_to_id, id_to_token, _ = train_demo_model(text)

    context = "the cat"
    context_ids = [token_to_id[token] for token in tokenize(context)]

    print(f"Context: '{context}'")
    print("=" * 100)
    print()

    # Test different temperatures
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0]

    for temp in temperatures:
        sampler = SamplerConfig(
            transform=LogitTransform.ZSCORE,
            temperature=temp,
            top_k=0,
            top_p=0.95,
            seed=1234,
        )

        ids, scores, raw_strengths, support_counts, probabilities = model.predict(
            context_ids[-model.window:],
            max_predictions=model.top_k,
            sampler=sampler,
        )

        print(f"Temperature: {temp}")
        print(f"  Top prediction: '{id_to_token[ids[0]]}' (id={ids[0]}, prob={probabilities[0]:.6f})")
        print(f"  Distribution of top 5:")
        for i in range(min(5, len(ids))):
            token = id_to_token[ids[i]]
            print(f"    {i+1}. {token:<10} id={ids[i]:<3}  prob={probabilities[i]:.6f}  score={scores[i]:8.4f}  raw={raw_strengths[i]:8.4f}  support={support_counts[i]}")

        # Calculate entropy of distribution
        probs = np.asarray(probabilities, dtype=np.float32)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        print(f"  Entropy: {entropy:.4f}")
        print()

    print("=" * 100)
    print("\nNow let's generate a sequence at different temperatures:")
    print("=" * 100)
    print()

    for temp in [0.1, 1.0, 10.0, 100.0]:
        sampler = SamplerConfig(
            transform=LogitTransform.ZSCORE,
            temperature=temp,
            top_k=0,
            top_p=0.95,
            seed=1234,
        )

        tokens = list(context_ids)
        generated = []

        for _ in range(10):
            ids, scores, raw_strengths, support_counts, probabilities = model.predict(
                tokens[-model.window:],
                max_predictions=model.top_k,
                sampler=sampler,
            )

            if not ids:
                break

            # Greedy selection
            selected_token = ids[0]
            generated.append(id_to_token[selected_token])
            tokens.append(selected_token)

        print(f"Temperature {temp:6.1f}: {' '.join(generated)}")


if __name__ == "__main__":
    main()

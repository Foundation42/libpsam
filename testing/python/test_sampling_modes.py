"""
Test to verify greedy vs stochastic sampling behavior.
"""
from __future__ import annotations

import re
from typing import List

from psam import LogitTransform, PSAM, SamplerConfig


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[.,!?;]", text.lower())


def build_vocab(tokens):
    ordered = list(dict.fromkeys(tokens))
    token_to_id = {token: idx for idx, token in enumerate(ordered)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return token_to_id, id_to_token


def main():
    text = (
        "the cat sat on the mat. "
        "the dog sat on the rug. "
        "the bird sat on the branch. "
        "the frog sat on the log."
    )

    tokens = tokenize(text)
    token_to_id, id_to_token = build_vocab(tokens)
    sequence = [token_to_id[token] for token in tokens]

    model = PSAM(vocab_size=len(token_to_id), window=8, top_k=32)
    model.train_batch(sequence)
    model.finalize_training()

    context = "the cat"
    context_ids = [token_to_id[token] for token in tokenize(context)]

    print("Testing Greedy vs Stochastic Sampling")
    print("=" * 80)
    print()

    # Test with different temperatures
    for temp in [0.5, 1.0, 2.0]:
        print(f"Temperature: {temp}")
        print()

        sampler = SamplerConfig(
            transform=LogitTransform.ZSCORE,
            temperature=temp,
            top_k=0,
            top_p=0.95,
            seed=1234,
        )

        # Greedy mode - run multiple times, should always be the same
        print("  GREEDY MODE (should be deterministic):")
        for run in range(3):
            tokens_gen = list(context_ids)
            generated = []

            for _ in range(10):
                ids, _scores, _raw, _support, probs = model.predict(
                    tokens_gen[-model.window:],
                    max_predictions=model.top_k,
                    sampler=sampler,
                )

                # Greedy: always pick index 0
                selected_token = ids[0]
                generated.append(id_to_token[selected_token])
                tokens_gen.append(selected_token)

            print(f"    Run {run + 1}: {' '.join(generated)}")

        print()

        # Stochastic mode - run multiple times, should vary
        print("  STOCHASTIC MODE (should vary between runs):")
        for run in range(3):
            tokens_gen = list(context_ids)
            generated = []

            for _ in range(10):
                ids, _scores, _raw, _support, probs = model.predict(
                    tokens_gen[-model.window:],
                    max_predictions=model.top_k,
                    sampler=sampler,
                )

                # Stochastic: sample from distribution
                import random
                total_prob = sum(probs)
                target = random.random() * total_prob
                cumsum = 0.0
                selected_idx = 0

                for idx, prob in enumerate(probs):
                    cumsum += prob
                    if target <= cumsum:
                        selected_idx = idx
                        break

                selected_token = ids[selected_idx]
                generated.append(id_to_token[selected_token])
                tokens_gen.append(selected_token)

            print(f"    Run {run + 1}: {' '.join(generated)}")

        print()
        print("-" * 80)
        print()


if __name__ == "__main__":
    main()

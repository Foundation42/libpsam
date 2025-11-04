"""
Regression tests mirroring the browser WASM demo auto-generation loop.

These ensure that sampling via `predict(...).probabilities` stays aligned with
the returned token IDs across iterative steps—matching the logic used in
`PSAMWasmDemo.handleGenerate` in the web UI.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

import numpy as np
import pytest

from psam import LogitTransform, PSAM, SamplerConfig, is_library_available


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[.,!?;]", text.lower())


def build_vocab(tokens: Iterable[str]) -> tuple[dict[str, int], dict[int, str]]:
    ordered = list(dict.fromkeys(tokens))
    token_to_id = {token: idx for idx, token in enumerate(ordered)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return token_to_id, id_to_token


def train_demo_model(text: str, window: int = 8, top_k: int = 32) -> tuple[PSAM, dict[str, int], dict[int, str], List[int]]:
    tokens = tokenize(text)
    token_to_id, id_to_token = build_vocab(tokens)
    sequence = [token_to_id[token] for token in tokens]

    model = PSAM(vocab_size=len(token_to_id), window=window, top_k=top_k)
    model.train_batch(sequence)
    model.finalize_training()
    return model, token_to_id, id_to_token, sequence


def auto_generate(
    model: PSAM,
    context_ids: Sequence[int],
    sampler: SamplerConfig,
    num_tokens: int,
) -> List[int]:
    """
    Mirror the WASM demo's handleGenerate loop with greedy selection.
    Always picks the top prediction (index 0).
    """
    tokens = list(context_ids)
    generated: List[int] = []

    for _ in range(num_tokens):
        ids, _scores, _raw_strengths, _support_counts, probabilities = model.predict(
            tokens[-model.window :],
            max_predictions=model.top_k,
            sampler=sampler,
        )

        assert ids, "Model returned no predictions"
        assert probabilities is not None, "Sampler must produce calibrated probabilities"
        probs = np.asarray(probabilities, dtype=np.float32)

        # Guard against missing or malformed probability mass: should be ~1.0
        total = float(np.sum(probs))
        assert total == pytest.approx(1.0, rel=1e-5, abs=1e-6), f"Unexpected probability mass {total}"

        # Greedy selection - always pick the top prediction
        selected_idx = 0
        selected_token = ids[selected_idx]
        generated.append(selected_token)
        tokens.append(selected_token)

    return generated


@pytest.mark.skipif(not is_library_available(), reason="libpsam native library not available")
@pytest.mark.parametrize(
    ("context", "expected_greedy"),
    [
        (
            "the cat sat on the",
            ["mat", ".", "the", "dog", "sat", "on", "the", "rug", ".", "the"],
        ),
        (
            "the cat",
            ["sat", "on", "the", "mat", ".", "the", "dog", "sat", "on", "the"],
        ),
    ],
)
def test_auto_generate_matches_expected_sequence(context, expected_greedy):
    text = (
        "the cat sat on the mat. "
        "the dog sat on the rug. "
        "the bird sat on the branch. "
        "the frog sat on the log."
    )

    model, token_to_id, id_to_token, _ = train_demo_model(text)

    sampler = SamplerConfig(
        transform=LogitTransform.ZSCORE,
        temperature=1.0,
        top_k=0,
        top_p=0.95,
        seed=1234,
    )

    context_ids = [token_to_id[token] for token in tokenize(context)]

    # Generate 10 tokens using greedy selection
    generated_ids = auto_generate(model, context_ids, sampler, num_tokens=10)
    generated_tokens = [id_to_token[token_id] for token_id in generated_ids]

    expected_greedy_ids = [token_to_id[token] for token in expected_greedy]
    assert generated_ids == expected_greedy_ids
    assert generated_tokens == expected_greedy

    print(f"Context '{context}' — Greedy sequence:", " ".join(generated_tokens))

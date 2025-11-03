"""
Integration tests for PSAM composition using the Python bindings.

These tests operate on synthetic vocabularies with disjoint token sets so that
expected distributions are easy to reason about. They are intended to run in CI
to guard the end-to-end composition workflow (training → composition → inference).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import pytest

from psam import (
    LayeredComposite,
    LogitTransform,
    PSAM,
    SamplerConfig,
    is_library_available,
)

# Synthetic vocabularies. Must stay in sync with the JavaScript tests.
VOCAB_SETS: Dict[str, List[str]] = {
    "A": ["apple", "ant", "arrow", "anchor", "atlas", "axe", "angel", "arch"],
    "B": ["ball", "bat", "bear", "boat", "bell", "bird", "bone", "bread"],
    "C": ["cat", "car", "cave", "coin", "crown", "cloud", "cup", "cliff"],
}


@dataclass(frozen=True)
class VocabInfo:
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]
    token_to_vocab: Dict[str, str]


def build_vocab(*sets: str) -> VocabInfo:
    tokens: List[str] = []
    token_to_vocab: Dict[str, str] = {}

    for name in sets:
        subset = VOCAB_SETS[name]
        tokens.extend(subset)
        token_to_vocab.update({token: name for token in subset})

    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}

    return VocabInfo(token_to_id=token_to_id, id_to_token=id_to_token, token_to_vocab=token_to_vocab)


def repeating_sequence(vocab: List[str], pattern: List[int], length: int) -> List[str]:
    seq: List[str] = []
    pattern_len = len(pattern)
    for i in range(length):
        seq.append(vocab[pattern[i % pattern_len]])
    return seq


def encode(sequence: Iterable[str], token_to_id: Dict[str, int]) -> List[int]:
    return [token_to_id[token] for token in sequence]


def train_model(tokens: List[int], vocab_size: int, window: int = 4, top_k: int = 32) -> PSAM:
    model = PSAM(vocab_size=vocab_size, window=window, top_k=top_k)
    model.train_batch(tokens)
    model.finalize_training()
    return model


def build_distribution(
    composite: LayeredComposite,
    context: List[int],
    id_to_token: Dict[int, str],
    token_to_vocab: Dict[str, str],
    sampler: SamplerConfig,
    top_k: int = 64,
) -> Dict[str, float]:
    ids, _scores, probabilities = composite.predict(context, max_predictions=top_k, sampler=sampler)
    assert probabilities is not None, "Sampler must produce probability mass"

    mass: Dict[str, float] = {}

    for idx, prob in zip(ids, probabilities):
        vocab = token_to_vocab[id_to_token[idx]]
        mass[vocab] = mass.get(vocab, 0.0) + float(prob)

    # Normalize to guard against slight truncation when top_k < vocab size
    total = sum(mass.values())
    if total > 0:
        mass = {k: (v / total) * 100.0 for k, v in mass.items()}
    return mass


@pytest.fixture(scope="module")
def sampler() -> SamplerConfig:
    return SamplerConfig(transform=LogitTransform.ZSCORE, temperature=1.0, top_k=0, top_p=0.95, seed=1234)


@pytest.fixture(scope="module")
def models():
    if not is_library_available():
        pytest.skip("libpsam native library not available")

    vocab = build_vocab("A", "B", "C")
    length = 512

    seq_a = encode(repeating_sequence(VOCAB_SETS["A"], [0, 1, 2, 0, 1, 2], length), vocab.token_to_id)
    seq_b = encode(repeating_sequence(VOCAB_SETS["B"], [0, 2, 1, 0, 2, 1], length), vocab.token_to_id)
    seq_c = encode(repeating_sequence(VOCAB_SETS["C"], [0, 1, 2, 3, 4, 5], length), vocab.token_to_id)

    model_a = train_model(seq_a, len(vocab.token_to_id))
    model_b = train_model(seq_b, len(vocab.token_to_id))
    model_c = train_model(seq_c, len(vocab.token_to_id))

    yield vocab, model_a, model_b, model_c

    model_c.destroy()
    model_b.destroy()
    model_a.destroy()


def test_pure_dominance(models, sampler: SamplerConfig):
    vocab, model_a, model_b, _model_c = models
    composite = model_a.create_layered_composite()
    try:
        composite.set_base_weight(1.0)
        composite.add_layer("overlay_b", model_b, 0.0)

        context = [vocab.token_to_id["apple"]]
        mass = build_distribution(composite, context, vocab.id_to_token, vocab.token_to_vocab, sampler)

        total = sum(mass.values())
        assert abs(total - 100.0) < 1e-3
        assert mass.get("A", 0.0) >= 80.0, f"Expected A to dominate, got {mass}"
        assert mass.get("B", 0.0) <= 15.0, f"Unexpected B leakage: {mass}"
        assert mass.get("C", 0.0) <= 15.0, f"Unexpected C leakage: {mass}"
    finally:
        composite.destroy()


def test_equal_blend(models, sampler: SamplerConfig):
    vocab, model_a, model_b, _model_c = models
    composite = model_a.create_layered_composite()
    try:
        composite.set_base_weight(0.5)
        composite.add_layer("overlay_b", model_b, 0.5)

        context = [vocab.token_to_id["apple"]]
        mass = build_distribution(composite, context, vocab.id_to_token, vocab.token_to_vocab, sampler)

        total = sum(mass.values())
        assert abs(total - 100.0) < 1e-3
        assert mass.get("A", 0.0) >= 45.0, f"Expected A share near parity, got {mass}"
        assert mass.get("B", 0.0) >= 30.0, f"Expected B contribution, got {mass}"
        assert mass.get("C", 0.0) <= 15.0, f"C should remain secondary, got {mass}"
    finally:
        composite.destroy()


def test_weighted_blend(models, sampler: SamplerConfig):
    vocab, model_a, model_b, _model_c = models
    composite = model_a.create_layered_composite()
    try:
        composite.set_base_weight(0.7)
        composite.add_layer("overlay_b", model_b, 0.3)

        context = [vocab.token_to_id["apple"]]
        mass = build_distribution(composite, context, vocab.id_to_token, vocab.token_to_vocab, sampler)

        total = sum(mass.values())
        assert abs(total - 100.0) < 1e-3
        assert mass.get("A", 0.0) >= 70.0, f"A should lead, got {mass}"
        assert 10.0 <= mass.get("B", 0.0) <= 35.0, f"B should contribute meaningfully, got {mass}"
        assert mass.get("C", 0.0) <= 15.0, f"C should be minor, got {mass}"
    finally:
        composite.destroy()


def test_three_layer_blend(models, sampler: SamplerConfig):
    vocab, model_a, model_b, model_c = models
    composite = model_a.create_layered_composite()
    try:
        composite.set_base_weight(0.5)
        composite.add_layer("overlay_b", model_b, 0.3)
        composite.add_layer("overlay_c", model_c, 0.2)

        context = [vocab.token_to_id["apple"]]
        mass = build_distribution(composite, context, vocab.id_to_token, vocab.token_to_vocab, sampler)

        total = sum(mass.values())
        assert abs(total - 100.0) < 1e-3, f"Distribution should sum to 100%, got {total:.2f}%"
        assert mass.get("A", 0.0) > mass.get("B", 0.0) > mass.get("C", 0.0)
        assert mass.get("B", 0.0) >= 10.0, f"Expected at least 10% contribution from vocab B, got {mass.get('B', 0.0):.2f}%"
        assert mass.get("C", 0.0) >= 10.0, f"Expected at least 10% contribution from vocab C, got {mass.get('C', 0.0):.2f}%"
    finally:
        composite.destroy()

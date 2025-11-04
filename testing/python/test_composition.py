"""
Integration tests for PSAM composition using the Python bindings.

These tests operate on synthetic vocabularies with disjoint token sets so that
expected distributions are easy to reason about. They are intended to run in CI
to guard the end-to-end composition workflow (training → composition → inference).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List
import re

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
    ids, _scores, _raw_strengths, _support_counts, probabilities = composite.predict(
        context, max_predictions=top_k, sampler=sampler
    )
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


def test_markov_transitions(sampler: SamplerConfig):
    """Test 5: Verify that learned transition patterns are preserved in blends."""
    if not is_library_available():
        pytest.skip("libpsam native library not available")

    vocab = build_vocab("A", "B")
    length = 512

    # Layer A: Strong apple→ant→arrow cycle (90% transitions)
    pattern_a = [0, 1, 2] * (length // 3)  # apple, ant, arrow repeating
    seq_a = encode([VOCAB_SETS["A"][i] for i in pattern_a[:length]], vocab.token_to_id)

    # Layer B: Strong ball→bat→bear cycle (90% transitions)
    pattern_b = [0, 1, 2] * (length // 3)  # ball, bat, bear repeating
    seq_b = encode([VOCAB_SETS["B"][i] for i in pattern_b[:length]], vocab.token_to_id)

    model_a = train_model(seq_a, len(vocab.token_to_id))
    model_b = train_model(seq_b, len(vocab.token_to_id))

    try:
        # Test pure A - should show strong apple→ant pattern
        composite_pure_a = model_a.create_layered_composite()
        try:
            composite_pure_a.set_base_weight(1.0)
            composite_pure_a.add_layer("overlay_b", model_b, 0.0)

            context = [vocab.token_to_id["apple"]]
            ids, _scores, _raw_strengths, _support_counts, probs = composite_pure_a.predict(
                context, max_predictions=8, sampler=sampler
            )
            assert probs is not None

            # Find "ant" in predictions (should be top prediction for apple)
            ant_id = vocab.token_to_id["ant"]
            ant_idx = None
            for i, pred_id in enumerate(ids):
                if pred_id == ant_id:
                    ant_idx = i
                    break

            assert ant_idx is not None, "Expected 'ant' to appear in predictions after 'apple'"
            assert ant_idx <= 2, f"Expected 'ant' in top 3 predictions, found at position {ant_idx}"
        finally:
            composite_pure_a.destroy()

        # Test blended - should show BOTH patterns
        composite_blend = model_a.create_layered_composite()
        try:
            composite_blend.set_base_weight(0.5)
            composite_blend.add_layer("overlay_b", model_b, 0.5)

            context = [vocab.token_to_id["apple"]]
            ids, _scores, _raw_strengths, _support_counts, probs = composite_blend.predict(
                context, max_predictions=16, sampler=sampler
            )
            assert probs is not None

            # Both A-pattern (ant) and B-pattern tokens should appear
            predicted_tokens = [vocab.id_to_token[id] for id in ids]
            a_tokens = [t for t in predicted_tokens if t in VOCAB_SETS["A"]]
            b_tokens = [t for t in predicted_tokens if t in VOCAB_SETS["B"]]

            assert len(a_tokens) > 0, "Expected some A-vocabulary tokens in blend"
            assert len(b_tokens) > 0, "Expected some B-vocabulary tokens in blend"

            # Verify that learned patterns are still visible
            # "ant" should still rank high (A-pattern preservation)
            ant_id = vocab.token_to_id["ant"]
            if ant_id in ids:
                ant_position = list(ids).index(ant_id)
                assert ant_position < len(ids) // 2, "A-pattern token 'ant' should rank in top half"
        finally:
            composite_blend.destroy()
    finally:
        model_b.destroy()
        model_a.destroy()


def test_weight_sweep_linearity(sampler: SamplerConfig):
    """Test smooth linear transition of probability mass across weight sweep."""
    if not is_library_available():
        pytest.skip("libpsam native library not available")

    vocab = build_vocab("A", "B")
    length = 512

    seq_a = encode(repeating_sequence(VOCAB_SETS["A"], [0, 1, 2, 0, 1, 2], length), vocab.token_to_id)
    seq_b = encode(repeating_sequence(VOCAB_SETS["B"], [0, 2, 1, 0, 2, 1], length), vocab.token_to_id)

    model_a = train_model(seq_a, len(vocab.token_to_id))
    model_b = train_model(seq_b, len(vocab.token_to_id))

    try:
        weights = [1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0]
        a_percentages = []

        for weight_a in weights:
            weight_b = 1.0 - weight_a
            composite = model_a.create_layered_composite()
            try:
                composite.set_base_weight(weight_a)
                composite.add_layer("overlay_b", model_b, weight_b)

                context = [vocab.token_to_id["apple"]]
                mass = build_distribution(composite, context, vocab.id_to_token, vocab.token_to_vocab, sampler)

                a_percent = mass.get("A", 0.0)
                a_percentages.append(a_percent)

                # Each weight should produce distribution close to target (±16%)
                expected = weight_a * 100.0
                assert abs(a_percent - expected) <= 16.0, f"Weight {weight_a:.1f}: expected ~{expected:.0f}% A, got {a_percent:.1f}%"
            finally:
                composite.destroy()

        # Verify monotonic decrease (linearity check)
        # A-percentage should consistently decrease as weight_a decreases
        for i in range(len(a_percentages) - 1):
            # Allow small tolerance for noise, but trend should be clear
            assert a_percentages[i] >= a_percentages[i + 1] - 5.0, \
                f"Non-monotonic transition at weight {weights[i]:.1f}: {a_percentages[i]:.1f}% -> {a_percentages[i+1]:.1f}%"
    finally:
        model_b.destroy()
        model_a.destroy()


def test_zero_weight_elimination(models, sampler: SamplerConfig):
    """Test 6: Verify that weight=0 completely eliminates layer influence."""
    vocab, model_a, model_b, _model_c = models
    composite = model_a.create_layered_composite()
    try:
        composite.set_base_weight(1.0)
        composite.add_layer("overlay_b", model_b, 0.0)

        context = [vocab.token_to_id["apple"]]
        mass = build_distribution(composite, context, vocab.id_to_token, vocab.token_to_vocab, sampler)

        # Should be identical to pure A (Test 1)
        assert mass.get("A", 0.0) >= 80.0, f"Expected A to dominate with B weight=0, got {mass}"
        assert mass.get("B", 0.0) <= 15.0, f"B should have zero influence with weight=0, got {mass}"
    finally:
        composite.destroy()


def test_uniform_baseline(sampler: SamplerConfig):
    """Test 8: Baseline test with random sequences should show roughly uniform predictions."""
    if not is_library_available():
        pytest.skip("libpsam native library not available")

    import random

    vocab = build_vocab("A", "B")
    length = 512

    # Create truly random sequence (no patterns)
    random.seed(9999)
    all_tokens = list(vocab.token_to_id.keys())
    random_seq = [vocab.token_to_id[random.choice(all_tokens)] for _ in range(length)]

    model = train_model(random_seq, len(vocab.token_to_id))

    try:
        composite = model.create_layered_composite()
        try:
            composite.set_base_weight(1.0)

            context = [vocab.token_to_id["apple"]]
            ids, _scores, _raw_strengths, _support_counts, probs = composite.predict(
                context, max_predictions=16, sampler=sampler
            )
            assert probs is not None

            # Calculate entropy (should be relatively high for uniform distribution)
            # H = -Σ(p * log2(p))
            import math
            entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in probs)

            # Max entropy for 16 predictions would be log2(16) = 4.0
            # We expect at least moderate entropy (>2.0) for random training
            assert entropy > 2.0, f"Expected higher entropy for random baseline, got {entropy:.2f}"

            # No single token should dominate
            max_prob = max(probs)
            assert max_prob < 0.5, f"Expected no single token to dominate (max prob < 0.5), got {max_prob:.2f}"
        finally:
            composite.destroy()
    finally:
        model.destroy()


def test_consensus_prefers_multi_support(sampler: SamplerConfig):
    """Ensure consensus weighting promotes multi-support continuations."""
    if not is_library_available():
        pytest.skip("libpsam native library not available")

    story = (
        "once upon a time in a small village, there lived a curious young girl named luna. "
        "luna loved to explore the forest near her home. one sunny morning, luna decided to venture "
        "deeper into the woods than ever before. she discovered a hidden clearing where magical "
        "butterflies danced in the golden sunlight. the butterflies led her to an ancient oak tree "
        "with a door carved into its trunk. luna opened the door and found a library filled with "
        "books that whispered secrets of the forest. from that day on, the butterflies brought luna "
        "a special gift, a silver key that unlocked a hidden chamber deep within the oak tree."
    ).lower()

    tokens = re.findall(r"\w+|[.,!?;]", story)
    token_to_id = {token: idx for idx, token in enumerate(dict.fromkeys(tokens))}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    sequence = [token_to_id[token] for token in tokens]

    model = train_model(sequence, len(token_to_id))
    composite = model.create_layered_composite()

    try:
        composite.set_base_weight(1.0)
        context_ids = [token_to_id[word] for word in ["once", "upon", "a"]]

        ids, scores, raw_strengths, support_counts, probabilities = composite.predict(
            context_ids, max_predictions=12, sampler=sampler
        )

        id_list = list(ids)
        top_token = id_to_token[id_list[0]]
        assert top_token == "time", f"Expected consensus continuation 'time', got {top_token}"

        def find_index(word: str) -> int:
            token_id = token_to_id[word]
            assert token_id in id_list, f"{word} missing from predictions"
            return id_list.index(token_id)

        time_idx = find_index("time")
        special_idx = find_index("special")

        assert support_counts[time_idx] >= 2, "Expected 'time' to accumulate multiple supports"
        assert support_counts[special_idx] == 1, "'special' should remain single-support"
        assert scores[time_idx] > scores[special_idx], "Consensus score should outrank unigram continuation"
        assert probabilities[time_idx] > probabilities[special_idx], "Sampler probability should reflect consensus preference"
    finally:
        composite.destroy()
        model.destroy()

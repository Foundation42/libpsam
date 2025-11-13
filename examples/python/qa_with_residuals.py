#!/usr/bin/env python3
"""
Prototype: Residual/Deferred Activation for PSAM Q&A

The idea: When predicting position j, also compute what associations
would fire at j+1, j+2, etc., and carry those forward as "residuals"
to boost predictions at future positions.
"""

from psam import PSAM, is_library_available
import numpy as np
from collections import defaultdict

QA_PAIRS = [
    ("What is the dog doing?", "The dog is rolling on the rug."),
    ("What is the horse doing?", "The horse is galloping across the field."),
    ("What is the cat doing?", "The cat is sleeping on the mat."),
]

def tokenize_simple(text):
    return text.lower().replace('?', ' ?').replace('.', ' .').split()

def build_vocab(qa_pairs):
    vocab = {}
    idx = 1
    for question, answer in qa_pairs:
        for word in tokenize_simple(question) + tokenize_simple(answer):
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

def encode_text(text, vocab):
    return [vocab[word] for word in tokenize_simple(text)]

def decode_tokens(tokens, vocab):
    reverse_vocab = {v: k for k, v in vocab.items()}
    return ' '.join([reverse_vocab.get(t, f'<UNK:{t}>') for t in tokens])

def predict_with_residuals(psam, context, vocab, max_lookahead=3, residual_decay=0.7, verbose=False):
    """
    Predict next token considering deferred activations from context.

    Args:
        psam: PSAM model
        context: List of token IDs
        vocab: Vocabulary dict
        max_lookahead: How many positions ahead to compute residuals for
        residual_decay: How much to decay residuals at each future position
        verbose: Print debug info

    Returns:
        (token_id, detailed_scores) where detailed_scores shows base + residual contributions
    """
    reverse_vocab = {v: k for k, v in vocab.items()}

    # Get base predictions for current position
    pred_ids, base_scores, _, _, _ = psam.predict(context, max_predictions=20)

    if verbose:
        print(f"\n  Context (last 4): {[reverse_vocab[t] for t in context[-4:]]}")
        print(f"  Predicting position {len(context)}")

    # Build a score dictionary for easier manipulation
    score_dict = {tid: score for tid, score in zip(pred_ids, base_scores)}

    # Compute residuals: for each context token, what would it activate at future positions?
    residual_scores = defaultdict(lambda: defaultdict(float))  # residual_scores[lookahead][token_id] = score

    for lookahead in range(1, max_lookahead + 1):
        # Simulate predicting at position: current + lookahead
        # This means we're looking for associations with offset increased by lookahead

        # To simulate this, we create a "fake" extended context
        # We can't actually predict future positions without knowing what tokens will be there,
        # but we CAN see what associations from CURRENT context would fire at those offsets

        # For each token in context, check if it has associations that would fire at +lookahead
        for ctx_pos, ctx_token in enumerate(context):
            # The offset from this context token to the future position
            future_offset = (len(context) + lookahead - 1) - ctx_pos

            # We need to query: what does ctx_token associate with at this future_offset?
            # Unfortunately, PSAM's Python API doesn't expose per-offset lookups directly
            # So we'll use a workaround: predict from a synthetic context

            # Workaround: Create a minimal context that puts ctx_token at the right relative position
            # This is a hack, but demonstrates the concept
            if future_offset > 0 and future_offset <= psam.window:
                # Create a synthetic context: just the single token at the right distance
                synthetic_context = [ctx_token]

                try:
                    # Predict what this token would activate at the offset
                    future_pred_ids, future_scores, _, _, _ = psam.predict(synthetic_context, max_predictions=20)

                    # These are associations at offset +1 from ctx_token
                    # We need to adjust for the actual offset we care about
                    # This is a simplification - ideally we'd query the graph directly

                    if future_offset == 1:
                        # These scores are directly applicable
                        for tid, score in zip(future_pred_ids, future_scores):
                            # Apply decay based on lookahead distance
                            decayed_score = score * (residual_decay ** lookahead)
                            residual_scores[lookahead][tid] += decayed_score
                except:
                    pass

    # Apply residuals to base scores
    enhanced_scores = score_dict.copy()

    # For position 0 (immediate next), add residuals from lookahead=1
    if 1 in residual_scores:
        for tid, residual in residual_scores[1].items():
            enhanced_scores[tid] = enhanced_scores.get(tid, -10.0) + residual

    if verbose:
        print(f"\n  Base predictions:")
        for i, (tid, score) in enumerate(sorted(score_dict.items(), key=lambda x: -x[1])[:5]):
            word = reverse_vocab.get(tid, f'<{tid}>')
            print(f"    {i+1}. '{word}' base: {score:.2f}")

        if residual_scores[1]:
            print(f"\n  Top residual contributions (lookahead=1):")
            for i, (tid, residual) in enumerate(sorted(residual_scores[1].items(), key=lambda x: -x[1])[:5]):
                word = reverse_vocab.get(tid, f'<{tid}>')
                print(f"    {i+1}. '{word}' residual: {residual:.2f}")

        print(f"\n  Enhanced predictions (base + residuals):")
        for i, (tid, score) in enumerate(sorted(enhanced_scores.items(), key=lambda x: -x[1])[:5]):
            word = reverse_vocab.get(tid, f'<{tid}>')
            base = score_dict.get(tid, 0)
            residual = residual_scores[1].get(tid, 0)
            print(f"    {i+1}. '{word}' enhanced: {score:.2f} (base: {base:.2f}, residual: {residual:.2f})")

    # Return top prediction
    best_token = max(enhanced_scores.items(), key=lambda x: x[1])[0]
    return best_token, enhanced_scores

def generate_with_residuals(psam, question_tokens, vocab, max_length=15, verbose=False):
    """Generate an answer using residual-enhanced predictions"""
    reverse_vocab = {v: k for k, v in vocab.items()}

    generated = question_tokens.copy()

    if verbose:
        print(f"\nGenerating with residuals:")
        print(f"Starting context: {[reverse_vocab[t] for t in question_tokens]}")

    for step in range(max_length):
        if verbose:
            print(f"\n--- Step {step + 1} ---")

        next_token, _ = predict_with_residuals(
            psam, generated, vocab,
            max_lookahead=3,
            residual_decay=0.7,
            verbose=verbose
        )

        generated.append(next_token)

        word = reverse_vocab.get(next_token, f'<{next_token}>')
        if verbose:
            print(f"  → Generated: '{word}'")

        # Stop at period
        if word == '.':
            break

    return generated

def main():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     PSAM with Residual/Deferred Activation Prototype      ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    if not is_library_available():
        print("❌ Native library not available")
        return 1

    vocab = build_vocab(QA_PAIRS)
    reverse_vocab = {v: k for k, v in vocab.items()}

    print("📚 Training data:")
    for q, a in QA_PAIRS:
        print(f"   Q: {q}")
        print(f"   A: {a}")
    print()

    VOCAB_SIZE = len(vocab) + 10
    WINDOW = 8
    TOP_K = 20

    psam = PSAM(vocab_size=VOCAB_SIZE, window=WINDOW, top_k=TOP_K)

    # Train
    print("🎓 Training...")
    for question, answer in QA_PAIRS:
        q_tokens = encode_text(question, vocab)
        a_tokens = encode_text(answer, vocab)
        sequence = q_tokens + a_tokens
        psam.train_batch(sequence)

    psam.finalize_training()
    print("✓ Training complete\n")

    # Test with and without residuals
    test_q = "What is the dog doing?"
    print("="*70)
    print(f"🧪 TEST: {test_q}")
    print("="*70)

    q_tokens = encode_text(test_q, vocab)

    # Standard generation (baseline)
    print("\n📊 BASELINE: Standard generation (no residuals)")
    print("-" * 70)

    generated_baseline = q_tokens.copy()
    for i in range(15):
        pred_ids, scores, _, _, _ = psam.predict(generated_baseline, max_predictions=psam.top_k)
        if len(pred_ids) == 0:
            break

        # Sample with low temperature
        temperature = 0.3
        logits = scores / temperature
        logits = logits - np.max(logits)
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores)
        next_token = int(np.random.choice(pred_ids, p=probs))

        generated_baseline.append(next_token)
        if reverse_vocab.get(next_token) == '.':
            break

    baseline_text = decode_tokens(generated_baseline, vocab)
    print(f"Generated: {baseline_text}\n")

    # Check if it got "dog"
    answer_part = decode_tokens(generated_baseline[len(q_tokens):], vocab)
    if "dog" in answer_part and "rolling" in answer_part:
        print("✅ CORRECT: Mentioned 'dog' and 'rolling'")
    else:
        print("❌ INCORRECT: Did not mention 'dog' or 'rolling'")

    # Generation with residuals
    print("\n📊 ENHANCED: Generation with residuals")
    print("-" * 70)

    generated_residual = generate_with_residuals(psam, q_tokens, vocab, max_length=15, verbose=True)

    residual_text = decode_tokens(generated_residual, vocab)
    print(f"\nFinal generated: {residual_text}\n")

    # Check if it got "dog"
    answer_part = decode_tokens(generated_residual[len(q_tokens):], vocab)
    if "dog" in answer_part and "rolling" in answer_part:
        print("✅ CORRECT: Mentioned 'dog' and 'rolling'")
    else:
        print("❌ INCORRECT: Did not mention 'dog' or 'rolling'")

    print("\n" + "="*70)
    print("\n💡 ANALYSIS:")
    print("="*70)
    print()
    print("This prototype demonstrates the CONCEPT of residual activation,")
    print("but has limitations due to the Python API:")
    print()
    print("  • Can't directly query associations at specific offsets")
    print("  • Uses synthetic contexts as a workaround")
    print("  • Simplified lookahead computation")
    print()
    print("For a PROPER implementation, you'd want to:")
    print()
    print("  1. Add to the C API: query associations by (source, target, offset)")
    print("  2. At each prediction step, compute ALL future activations")
    print("     within the window (offsets +1, +2, ..., +window)")
    print("  3. Store these in a residual buffer indexed by future position")
    print("  4. When reaching each position, blend current + residual scores")
    print("  5. Apply decay to residuals based on distance")
    print()
    print("This would let context tokens continue to influence predictions")
    print("even when their exact offset doesn't match the current position!")

    psam.destroy()
    return 0

if __name__ == "__main__":
    exit(main())

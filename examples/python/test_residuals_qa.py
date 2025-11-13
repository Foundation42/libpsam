#!/usr/bin/env python3
"""
Test residual/deferred activation on Q&A dataset

This tests whether residuals can solve the offset mismatch problem!
"""

from psam import PSAM, ResidualConfig, is_library_available
import numpy as np

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

def generate_answer(psam, question_tokens, vocab, max_length=15, use_residuals=False):
    """Generate answer using standard or residual prediction"""
    reverse_vocab = {v: k for k, v in vocab.items()}
    generated = question_tokens.copy()

    residual_config = None
    if use_residuals:
        residual_config = ResidualConfig(
            max_lookahead=4,
            residual_decay=0.85,
            residual_blend=0.5,
            enable=True
        )

    for _ in range(max_length):
        pred_ids, scores, _, _, _ = psam.predict(
            generated,
            max_predictions=psam.top_k,
            residual_config=residual_config
        )

        if len(pred_ids) == 0:
            break

        # Sample with low temperature
        temperature = 0.3
        logits = scores / temperature
        logits = logits - np.max(logits)
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores)
        next_token = int(np.random.choice(pred_ids, p=probs))

        generated.append(next_token)
        if reverse_vocab.get(next_token) == '.':
            break

    return generated

def main():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     PSAM Residual Activation - Q&A Test                   ║")
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
    print("🎓 Training on concatenated Q+A pairs...")
    for question, answer in QA_PAIRS:
        q_tokens = encode_text(question, vocab)
        a_tokens = encode_text(answer, vocab)
        sequence = q_tokens + a_tokens
        psam.train_batch(sequence)

    psam.finalize_training()
    print("✓ Training complete\n")

    # Test question
    test_q = "What is the dog doing?"
    expected_keywords = ["dog", "rolling"]

    print("="*70)
    print(f"🧪 TEST QUESTION: {test_q}")
    print(f"Expected keywords: {expected_keywords}")
    print("="*70)

    q_tokens = encode_text(test_q, vocab)

    # Test WITHOUT residuals (baseline)
    print("\n📊 WITHOUT RESIDUALS (Baseline)")
    print("-" * 70)

    np.random.seed(42)  # For reproducibility
    generated_baseline = generate_answer(psam, q_tokens, vocab, use_residuals=False)
    baseline_text = decode_tokens(generated_baseline, vocab)

    print(f"Generated: {baseline_text}\n")

    answer_part = decode_tokens(generated_baseline[len(q_tokens):], vocab)
    baseline_correct = all(kw in answer_part for kw in expected_keywords)

    if baseline_correct:
        print("✅ CORRECT: Mentioned all expected keywords")
    else:
        print(f"❌ INCORRECT: Missing keywords from {expected_keywords}")
        print(f"   Answer was: {answer_part}")

    # Test WITH residuals
    print("\n📊 WITH RESIDUALS (Enhanced)")
    print("-" * 70)

    np.random.seed(42)  # Same seed for fair comparison
    generated_residual = generate_answer(psam, q_tokens, vocab, use_residuals=True)
    residual_text = decode_tokens(generated_residual, vocab)

    print(f"Generated: {residual_text}\n")

    answer_part = decode_tokens(generated_residual[len(q_tokens):], vocab)
    residual_correct = all(kw in answer_part for kw in expected_keywords)

    if residual_correct:
        print("✅ CORRECT: Mentioned all expected keywords")
    else:
        print(f"❌ INCORRECT: Missing keywords from {expected_keywords}")
        print(f"   Answer was: {answer_part}")

    # Summary
    print("\n" + "="*70)
    print("📈 RESULTS SUMMARY")
    print("="*70)
    print(f"Baseline (no residuals):  {'✅ PASS' if baseline_correct else '❌ FAIL'}")
    print(f"Enhanced (with residuals): {'✅ PASS' if residual_correct else '❌ FAIL'}")

    if residual_correct and not baseline_correct:
        print("\n🎉 SUCCESS! Residuals fixed the prediction!")
    elif residual_correct and baseline_correct:
        print("\n✓ Both methods work (residuals didn't break anything)")
    elif not residual_correct and baseline_correct:
        print("\n⚠️  WARNING: Residuals made it worse!")
    else:
        print("\n❌ Both methods failed - may need tuning")

    psam.destroy()

    return 0

if __name__ == "__main__":
    exit(main())

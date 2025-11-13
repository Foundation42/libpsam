#!/usr/bin/env python3
"""
Potential solutions for improving Q&A predictions with PSAM
"""

from psam import PSAM, is_library_available
import numpy as np

# Test dataset
QA_PAIRS = [
    ("Where is the cat?", "The cat is on the mat."),
    ("What is the dog doing?", "The dog is rolling on the rug."),
    ("Where is the rabbit hiding?", "The rabbit is hiding in the burrow."),
    ("What is the horse doing?", "The horse is galloping across the field."),
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

def test_approach(name, description, train_fn, test_fn, qa_pairs):
    """Test a specific approach"""
    print(f"\n{'='*70}")
    print(f"🧪 {name}")
    print(f"{'='*70}")
    print(f"Description: {description}\n")

    vocab = build_vocab(qa_pairs)
    VOCAB_SIZE = len(vocab) + 10
    WINDOW = 8
    TOP_K = 20

    psam = PSAM(vocab_size=VOCAB_SIZE, window=WINDOW, top_k=TOP_K)

    # Train using the approach
    train_fn(psam, qa_pairs, vocab)
    psam.finalize_training()

    # Test on a question
    test_question = "What is the dog doing?"
    expected = "The dog is rolling on the rug."

    print(f"Test Question: {test_question}")
    print(f"Expected: {expected}\n")

    result = test_fn(psam, test_question, vocab)
    print(f"Generated: {result}\n")

    # Check if correct
    if "dog" in result.lower() and "rolling" in result.lower():
        print("✅ CORRECT!")
    else:
        print("❌ INCORRECT")

    psam.destroy()
    return result

def approach_1_baseline(psam, qa_pairs, vocab):
    """Baseline: concatenate Q+A as single sequence"""
    print("Training: Concatenating Q+A as single sequences...")
    for question, answer in qa_pairs:
        q_tokens = encode_text(question, vocab)
        a_tokens = encode_text(answer, vocab)
        sequence = q_tokens + a_tokens
        psam.train_batch(sequence)

def test_baseline(psam, question, vocab):
    """Generate answer by sampling"""
    q_tokens = encode_text(question, vocab)
    generated = q_tokens.copy()

    for _ in range(15):
        pred_ids, scores, _, _, _ = psam.predict(generated, max_predictions=psam.top_k)
        if len(pred_ids) == 0:
            break

        temperature = 0.3
        logits = scores / temperature
        logits = logits - np.max(logits)
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores)
        next_token = int(np.random.choice(pred_ids, p=probs))

        generated.append(next_token)
        if decode_tokens([next_token], vocab) == '.':
            break

    return decode_tokens(generated, vocab)

def approach_2_separator(psam, qa_pairs, vocab):
    """Add a special separator token between Q and A"""
    print("Training: Using <SEP> separator between Q and A...")

    # Add <SEP> to vocab if not present
    if '<sep>' not in vocab:
        vocab['<sep>'] = max(vocab.values()) + 1

    for question, answer in qa_pairs:
        q_tokens = encode_text(question, vocab)
        a_tokens = encode_text(answer, vocab)
        sep_token = vocab['<sep>']

        # Q ? <SEP> A .
        sequence = q_tokens + [sep_token] + a_tokens
        psam.train_batch(sequence)

def test_separator(psam, question, vocab):
    """Generate answer with separator"""
    if '<sep>' not in vocab:
        vocab['<sep>'] = max(vocab.values()) + 1

    q_tokens = encode_text(question, vocab)
    sep_token = vocab['<sep>']

    # Start generation after separator
    generated = q_tokens + [sep_token]

    for _ in range(15):
        pred_ids, scores, _, _, _ = psam.predict(generated, max_predictions=psam.top_k)
        if len(pred_ids) == 0:
            break

        temperature = 0.3
        logits = scores / temperature
        logits = logits - np.max(logits)
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores)
        next_token = int(np.random.choice(pred_ids, p=probs))

        generated.append(next_token)
        if decode_tokens([next_token], vocab) == '.':
            break

    # Return only the answer part (after separator)
    answer_start = len(q_tokens) + 1
    return decode_tokens(generated[answer_start:], vocab)

def approach_3_repeat(psam, qa_pairs, vocab):
    """Repeat Q+A pairs multiple times to strengthen associations"""
    print("Training: Repeating each Q+A pair 5 times...")
    for question, answer in qa_pairs:
        q_tokens = encode_text(question, vocab)
        a_tokens = encode_text(answer, vocab)
        sequence = q_tokens + a_tokens

        # Train 5 times
        for _ in range(5):
            psam.train_batch(sequence)

def main():
    if not is_library_available():
        print("❌ Native library not available")
        return 1

    print("╔════════════════════════════════════════════════════════════╗")
    print("║     PSAM Q&A Solutions - Testing Different Approaches     ║")
    print("╚════════════════════════════════════════════════════════════╝")

    # Test Approach 1: Baseline
    test_approach(
        "Approach 1: Baseline (Concatenated Q+A)",
        "Simply concatenate questions and answers as training sequences",
        approach_1_baseline,
        test_baseline,
        QA_PAIRS
    )

    # Test Approach 2: Separator token
    test_approach(
        "Approach 2: Separator Token",
        "Insert a special <SEP> token between question and answer",
        approach_2_separator,
        test_separator,
        QA_PAIRS
    )

    # Test Approach 3: Repetition
    test_approach(
        "Approach 3: Repeated Training",
        "Train on each Q+A pair multiple times to strengthen patterns",
        approach_3_repeat,
        test_baseline,
        QA_PAIRS
    )

    print("\n" + "="*70)
    print("\n💡 SUMMARY OF FINDINGS:")
    print("\nThe core issue: PSAM learns that '?' is followed by 'the' across")
    print("ALL questions, but struggles to maintain which specific subject")
    print("(dog, cat, rabbit, etc.) should follow.")
    print("\nThis is because:")
    print("  1. The window size (8 tokens) may not capture full question context")
    print("  2. Common patterns like '? the [animal]' overwhelm specific signals")
    print("  3. Position-specific associations decay as context grows")
    print("\nBest solutions:")
    print("  • Use separator tokens to create clearer boundaries")
    print("  • Increase training repetitions for important patterns")
    print("  • Consider larger window sizes for longer-range dependencies")
    print("  • Use model composition to blend question-specific models")

    return 0

if __name__ == "__main__":
    exit(main())

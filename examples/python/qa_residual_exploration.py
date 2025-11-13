#!/usr/bin/env python3
"""
Exploring residual/continuation mechanisms for Q&A
"""

from psam import PSAM, is_library_available
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

def main():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     Residual/Continuation Exploration for Q&A             ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    if not is_library_available():
        print("❌ Native library not available")
        return 1

    vocab = build_vocab(QA_PAIRS)
    reverse_vocab = {v: k for k, v in vocab.items()}

    print("📚 Vocabulary:")
    for word, idx in sorted(vocab.items(), key=lambda x: x[1]):
        print(f"   {idx:3d}: {word}")
    print()

    VOCAB_SIZE = len(vocab) + 10
    WINDOW = 8
    TOP_K = 20

    psam = PSAM(vocab_size=VOCAB_SIZE, window=WINDOW, top_k=TOP_K)

    print("🎓 Training on concatenated Q+A pairs:")
    for question, answer in QA_PAIRS:
        q_tokens = encode_text(question, vocab)
        a_tokens = encode_text(answer, vocab)
        sequence = q_tokens + a_tokens
        psam.train_batch(sequence)
        print(f"   {question} → {answer}")

    psam.finalize_training()
    print("\n✓ Training complete\n")

    # Test question
    test_q = "What is the dog doing?"
    print("="*70)
    print(f"🔬 TESTING: {test_q}")
    print("="*70)

    q_tokens = encode_text(test_q, vocab)
    print(f"\nQuestion tokens: {q_tokens}")
    print(f"Words: {[reverse_vocab[t] for t in q_tokens]}\n")

    # Approach 1: Standard prediction (what we've been doing)
    print("\n📊 APPROACH 1: Standard prediction at position 6")
    print("-" * 70)
    context = q_tokens.copy()
    pred_ids, scores, raw_strengths, support_counts, _ = psam.predict(context, max_predictions=10)

    print("Context (positions 0-5):", [reverse_vocab[t] for t in context])
    print("Predicting position 6:\n")
    print("Top predictions:")
    for i, (tid, score, support) in enumerate(zip(pred_ids[:5], scores[:5], support_counts[:5])):
        word = reverse_vocab[tid]
        print(f"   {i+1}. '{word}' (score: {score:.2f}, support: {support})")

    # Approach 2: Look at what happens when we extend context
    print("\n\n📊 APPROACH 2: Progressive context extension")
    print("-" * 70)
    print("Let's see how predictions change as we add the answer tokens:\n")

    # Manually extend with the correct answer to see what happens
    correct_answer = "The dog is rolling on the rug."
    answer_tokens = encode_text(correct_answer, vocab)

    contexts_to_test = [
        (q_tokens, "Question only"),
        (q_tokens + answer_tokens[:1], "Q + 'the'"),
        (q_tokens + answer_tokens[:2], "Q + 'the dog'"),
        (q_tokens + answer_tokens[:3], "Q + 'the dog is'"),
    ]

    for ctx, description in contexts_to_test:
        print(f"\n{description}:")
        print(f"  Context: {[reverse_vocab[t] for t in ctx[-WINDOW:]]}")  # Show last WINDOW tokens

        pred_ids, scores, _, support_counts, _ = psam.predict(ctx, max_predictions=5)

        print(f"  Top 3 predictions:")
        for i, (tid, score, support) in enumerate(zip(pred_ids[:3], scores[:3], support_counts[:3])):
            word = reverse_vocab[tid]
            print(f"     {i+1}. '{word}' (score: {score:.2f}, support: {support})")

    # Approach 3: Explain how each token in the answer gets predicted
    print("\n\n📊 APPROACH 3: Token-by-token explanation")
    print("-" * 70)
    print("Let's trace how PSAM predicts each answer token:\n")

    full_sequence = q_tokens + answer_tokens

    for i in range(len(q_tokens), min(len(q_tokens) + 5, len(full_sequence))):
        context = full_sequence[:i]
        actual_next = full_sequence[i]
        actual_word = reverse_vocab[actual_next]

        print(f"\nPosition {i}: Predicting '{actual_word}'")
        print(f"  Context: ...{[reverse_vocab[t] for t in context[-4:]]}")

        # Get predictions
        pred_ids, scores, _, support_counts, _ = psam.predict(context, max_predictions=8)

        # Find where the actual token ranks
        actual_rank = None
        actual_score = None
        for rank, (tid, score) in enumerate(zip(pred_ids, scores)):
            if tid == actual_next:
                actual_rank = rank + 1
                actual_score = score
                break

        if actual_rank:
            print(f"  ✓ '{actual_word}' ranked #{actual_rank} (score: {actual_score:.2f})")
        else:
            print(f"  ✗ '{actual_word}' not in top predictions!")

        print(f"  Top 3:")
        for j, (tid, score, support) in enumerate(zip(pred_ids[:3], scores[:3], support_counts[:3])):
            word = reverse_vocab[tid]
            marker = "←" if tid == actual_next else ""
            print(f"     {j+1}. '{word}' (score: {score:.2f}, support: {support}) {marker}")

        # Use explain to see why
        if actual_rank and actual_rank <= 3:
            try:
                explanation = psam.explain(context, actual_next, max_terms=3)
                if explanation.terms:
                    print(f"  Explanation for '{actual_word}':")
                    for term in explanation.terms[:3]:
                        src_word = reverse_vocab.get(term.source, f'<{term.source}>')
                        print(f"     '{src_word}' @ offset {term.offset}: contribution {term.contribution:.2f}")
            except:
                pass

    print("\n\n" + "="*70)
    print("💡 OBSERVATIONS:")
    print("="*70)
    print()
    print("Notice how predictions improve as we add more context:")
    print("  1. With just the question, offset mismatches prevent good predictions")
    print("  2. As we add answer tokens, the context window shifts")
    print("  3. Tokens already generated can help predict the next ones")
    print()
    print("RESIDUAL IDEA:")
    print("  What if we carried forward 'unfired' associations?")
    print()
    print("  Example: When predicting position 6, 'dog'@3 has associations")
    print("  with offset +4 (learned from training). These don't fire for")
    print("  position 6, but we could 'carry them forward' as residuals.")
    print()
    print("  Then at position 7, those offset +4 associations would fire!")
    print()
    print("  This would be like a 'deferred activation' mechanism:")
    print("  - Track associations that WOULD fire at future positions")
    print("  - Accumulate them as residuals")
    print("  - Apply them when the offset matches")
    print()
    print("  Essentially: 'dog' in the question is waiting to activate 'dog'")
    print("  in the answer, and residuals let it defer that activation.")

    psam.destroy()
    return 0

if __name__ == "__main__":
    exit(main())

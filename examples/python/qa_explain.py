#!/usr/bin/env python3
"""
Use PSAM's explain() feature to understand why predictions are failing
"""

from psam import PSAM, is_library_available

# Minimal test case
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
    print("║     PSAM Q&A Explain - Why is it picking the wrong        ║")
    print("║                       animal?                              ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    if not is_library_available():
        print("❌ Native library not available")
        return 1

    # Build vocabulary
    vocab = build_vocab(QA_PAIRS)
    reverse_vocab = {v: k for k, v in vocab.items()}

    print("📚 Vocabulary:")
    for word, idx in sorted(vocab.items(), key=lambda x: x[1]):
        print(f"   {idx:3d}: {word}")
    print()

    # Create PSAM model
    VOCAB_SIZE = len(vocab) + 10
    WINDOW = 8
    TOP_K = 20

    print("📦 Creating PSAM model...")
    print(f"   Window: {WINDOW}, Top-K: {TOP_K}\n")

    psam = PSAM(vocab_size=VOCAB_SIZE, window=WINDOW, top_k=TOP_K)

    # Train
    print("🎓 Training on Q&A pairs:")
    for i, (question, answer) in enumerate(QA_PAIRS):
        q_tokens = encode_text(question, vocab)
        a_tokens = encode_text(answer, vocab)
        sequence = q_tokens + a_tokens

        print(f"   [{i+1}] {question}")
        print(f"       {answer}")
        print(f"       Tokens: {sequence}")

        psam.train_batch(sequence)

    psam.finalize_training()
    print("\n✓ Training complete\n")

    # Get statistics
    stats = psam.stats()
    print("📊 Model Statistics:")
    print(f"   - Rows: {stats.row_count}")
    print(f"   - Edges: {stats.edge_count}")
    print(f"   - Total tokens: {stats.total_tokens}\n")

    # Test the problematic case
    print("="*70)
    print("🔬 EXPLAINING PREDICTIONS")
    print("="*70)

    test_q = "What is the dog doing?"
    print(f"\nQuestion: {test_q}")
    print(f"Expected answer should mention: 'dog' and 'rolling'\n")

    # Get the context
    q_tokens = encode_text(test_q, vocab)
    print(f"Context tokens: {q_tokens}")
    print(f"Context words: {[reverse_vocab[t] for t in q_tokens]}\n")

    # Get top predictions
    pred_ids, scores, raw_strengths, support_counts, _ = psam.predict(q_tokens, max_predictions=10)

    print("Top 10 predictions:")
    for i, (token_id, score, support) in enumerate(zip(pred_ids, scores, support_counts)):
        word = reverse_vocab.get(token_id, f'<UNK:{token_id}>')
        print(f"   {i+1}. '{word}' (token {token_id}, score: {score:.2f}, support: {support})")

    print("\n" + "="*70)
    print("🔍 DETAILED EXPLANATION FOR KEY CANDIDATES")
    print("="*70)

    # Explain why each animal was or wasn't predicted
    candidates_to_explain = ["dog", "horse", "cat", "the"]

    for word in candidates_to_explain:
        if word not in vocab:
            continue

        token_id = vocab[word]
        print(f"\n📋 Explaining candidate: '{word}' (token {token_id})")
        print("-" * 70)

        # Use the explain() function
        try:
            explanation = psam.explain(q_tokens, token_id, max_terms=10)

            print(f"   Total score: {explanation.total:.4f}")
            print(f"   Bias (unigram): {explanation.bias:.4f}")
            print(f"   Term count: {explanation.term_count}")

            if explanation.terms:
                print(f"\n   Top contributing associations:")
                for i, term in enumerate(explanation.terms):
                    source_word = reverse_vocab.get(term.source, f'<UNK:{term.source}>')
                    print(f"      [{i+1}] '{source_word}' @ offset {term.offset}")
                    print(f"          PPMI weight: {term.weight:.4f}")
                    print(f"          IDF: {term.idf:.4f}")
                    print(f"          Decay: {term.decay:.4f}")
                    print(f"          → Contribution: {term.contribution:.4f}")
            else:
                print("   No associations found!")

        except Exception as e:
            print(f"   Error explaining: {e}")

    print("\n" + "="*70)
    print("\n💡 ANALYSIS:")
    print("\nLook at the explanations above. For 'dog' to be predicted correctly,")
    print("we should see:")
    print("  1. Strong PPMI from 'dog' in question → 'dog' in answer")
    print("  2. High IDF for 'dog' (it's a rare, informative word)")
    print("  3. The contribution from 'dog' → 'dog' should dominate")
    print("\nIf 'horse' or 'cat' score higher, check:")
    print("  - Are their PPMI values higher? (shouldn't be)")
    print("  - Are the offsets correct?")
    print("  - Is the window size capturing 'dog' from the question?")

    # Cleanup
    psam.destroy()

    return 0

if __name__ == "__main__":
    exit(main())

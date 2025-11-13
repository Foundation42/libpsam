#!/usr/bin/env python3
"""
Detailed Q&A debugging to understand PSAM prediction behavior
"""

from psam import PSAM, is_library_available
import numpy as np

# Simplified Q&A dataset for debugging
QA_PAIRS = [
    ("Where is the cat?", "The cat is on the mat."),
    ("What is the dog doing?", "The dog is rolling on the rug."),
    ("Where is the rabbit hiding?", "The rabbit is hiding in the burrow."),
]

def tokenize_simple(text):
    """Simple word-level tokenization"""
    return text.lower().replace('?', ' ?').replace('.', ' .').split()

def build_vocab(qa_pairs):
    """Build vocabulary from Q&A pairs"""
    vocab = {}
    idx = 1  # Start from 1

    for question, answer in qa_pairs:
        for word in tokenize_simple(question) + tokenize_simple(answer):
            if word not in vocab:
                vocab[word] = idx
                idx += 1

    return vocab

def encode_text(text, vocab):
    """Encode text to token IDs"""
    return [vocab[word] for word in tokenize_simple(text)]

def decode_tokens(tokens, vocab):
    """Decode token IDs to text"""
    reverse_vocab = {v: k for k, v in vocab.items()}
    return ' '.join([reverse_vocab.get(t, f'<UNK:{t}>') for t in tokens])

def main():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║      PSAM Q&A Detailed Debug - Understanding Context      ║")
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
    print(f"   - Vocabulary size: {VOCAB_SIZE}")
    print(f"   - Window: {WINDOW} (context window for predictions)")
    print(f"   - Top-K: {TOP_K}\n")

    psam = PSAM(vocab_size=VOCAB_SIZE, window=WINDOW, top_k=TOP_K)

    # Train on Q&A pairs
    print("🎓 Training sequences:")
    print("="*70)
    for i, (question, answer) in enumerate(QA_PAIRS):
        q_tokens = encode_text(question, vocab)
        a_tokens = encode_text(answer, vocab)

        # This is key: we're training on CONCATENATED sequences
        # Q: Where is the cat ? A: The cat is on the mat .
        sequence = q_tokens + a_tokens

        print(f"\n[{i+1}] Q: {question}")
        print(f"    A: {answer}")
        print(f"    Token sequence: {sequence}")
        print(f"    Decoded: {decode_tokens(sequence, vocab)}")

        psam.train_batch(sequence)

    psam.finalize_training()
    print("\n" + "="*70)
    print("✓ Training complete\n")

    # Get statistics
    stats = psam.stats()
    print("📊 Model Statistics:")
    print(f"   - Rows: {stats.row_count}")
    print(f"   - Edges: {stats.edge_count} (learned associations)")
    print(f"   - Total tokens processed: {stats.total_tokens}\n")

    # Now let's debug what happens with different contexts
    print("🔍 DEBUGGING PREDICTIONS")
    print("="*70)

    # Test 1: Question "What is the dog doing?"
    test_q = "What is the dog doing?"
    print(f"\n🧪 Test Question: {test_q}")
    print(f"Expected Answer: The dog is rolling on the rug.\n")

    q_tokens = encode_text(test_q, vocab)
    print(f"Full question context: {q_tokens} = {decode_tokens(q_tokens, vocab)}")

    # Try different context lengths
    for context_len in [6, 5, 4, 3, 2, 1]:
        context = q_tokens[-context_len:]
        print(f"\n  Context (last {context_len} tokens): {context}")
        print(f"  = '{decode_tokens(context, vocab)}'")

        pred_ids, scores, raw_strengths, support_counts, _ = psam.predict(context, max_predictions=5)

        print(f"  Top 5 predictions:")
        for i, (token_id, score, support) in enumerate(zip(pred_ids, scores, support_counts)):
            word = reverse_vocab.get(token_id, f'<UNK:{token_id}>')
            print(f"    {i+1}. '{word}' (score: {score:.2f}, support: {support})")

    print("\n" + "="*70)

    # Test what PSAM learned after "dog doing ?"
    print("\n🔬 What did PSAM learn about the context 'dog doing ?'")
    context = encode_text("dog doing ?", vocab)
    print(f"Context: {context} = {decode_tokens(context, vocab)}")

    pred_ids, scores, raw_strengths, support_counts, _ = psam.predict(context, max_predictions=10)
    print(f"\nTop 10 predictions:")
    for i, (token_id, score, support) in enumerate(zip(pred_ids, scores, support_counts)):
        word = reverse_vocab.get(token_id, f'<UNK:{token_id}>')
        print(f"  {i+1}. '{word}' (score: {score:.2f}, support: {support})")

    print("\n" + "="*70)
    print("\n💡 KEY INSIGHT:")
    print("PSAM learns position-specific associations within the training window.")
    print("The window size ({}) determines how far back it looks for patterns.".format(WINDOW))
    print("\nWhen you ask 'What is the dog doing?', PSAM sees:")
    print("  - The pattern '? the' appears frequently (after ALL questions)")
    print("  - It may not have enough unique signal from 'dog doing' alone")
    print("  - So it predicts common continuations from the whole dataset")
    print("\nPotential solutions:")
    print("  1. Use a larger window to capture more question context")
    print("  2. Add more training data with distinct patterns")
    print("  3. Use question-specific markers or tokens")
    print("  4. Increase the weight of recent context tokens")

    # Cleanup
    psam.destroy()

    return 0

if __name__ == "__main__":
    exit(main())

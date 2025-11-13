#!/usr/bin/env python3
"""
Q&A dataset test for PSAM debugging

Tests PSAM's ability to answer questions based on context
"""

from psam import PSAM, is_library_available

# Q&A dataset
QA_PAIRS = [
    ("Where is the cat?", "The cat is on the mat."),
    ("What is the dog doing?", "The dog is rolling on the rug."),
    ("Where did the bird chirp?", "The bird chirped on the branch."),
    ("How did the frog move?", "The frog hopped on the log."),
    ("Where is the fox sleeping?", "The fox is sleeping in the den."),
    ("What is the fish doing in the pond?", "The fish is swimming in the pond."),
    ("Where did the bee buzz?", "The bee buzzed near the hive."),
    ("What is the mouse nibbling?", "The mouse is nibbling the cheese."),
    ("Where is the snake slithering?", "The snake is slithering through the grass."),
    ("What is the horse doing?", "The horse is galloping across the field."),
    ("Where did the squirrel scamper?", "The squirrel scampered up the tree."),
    ("What sound did the owl make?", "The owl hooted from the barn."),
    ("Where is the rabbit hiding?", "The rabbit is hiding in the burrow."),
    ("Who howled at the moon?", "The wolf howled at the moon."),
    ("What is basking on the rock?", "The lion is basking on the rock."),
    ("How does a penguin walk on the ice?", "The penguin waddles on the ice."),
    ("Where is the bear fishing?", "The bear is fishing in the stream."),
    ("What is bouncing through the outback?", "The kangaroo is bouncing through the outback."),
    ("What did the monkey swing from?", "The monkey swung from the vine."),
    ("Who is balancing on the ball?", "The seal is balancing on the ball."),
    ("Where is the dragon perched?", "The dragon is perched on the cliff."),
    ("What is grazing in the meadow?", "The unicorn is grazing in the meadow."),
    ("Where did the mermaid rest?", "The mermaid rested on the rock."),
    ("What is hiding in the coral?", "The octopus is hiding in the coral."),
    ("Who is munching on the bamboo?", "The panda is munching on the bamboo."),
]

def tokenize_simple(text):
    """Simple word-level tokenization"""
    return text.lower().replace('?', ' ?').replace('.', ' .').split()

def build_vocab(qa_pairs):
    """Build vocabulary from Q&A pairs"""
    vocab = {}
    idx = 1  # Start from 1, reserve 0 for special tokens

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
    print("║          PSAM Q&A Dataset Debug Test                      ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    if not is_library_available():
        print("❌ Native library not available")
        return 1

    # Build vocabulary
    print("📚 Building vocabulary...")
    vocab = build_vocab(QA_PAIRS)
    print(f"   Vocabulary size: {len(vocab)} unique words\n")

    # Show some vocab examples
    print("   Sample vocabulary:")
    for i, (word, idx) in enumerate(list(vocab.items())[:10]):
        print(f"   '{word}' -> {idx}")
    print()

    # Create PSAM model
    VOCAB_SIZE = len(vocab) + 10  # Add some padding
    WINDOW = 8
    TOP_K = 10

    print("📦 Creating PSAM model...")
    print(f"   - Vocabulary size: {VOCAB_SIZE}")
    print(f"   - Window: {WINDOW}")
    print(f"   - Top-K: {TOP_K}\n")

    psam = PSAM(vocab_size=VOCAB_SIZE, window=WINDOW, top_k=TOP_K)

    # Train on Q&A pairs
    print("🎓 Training on Q&A pairs...")
    for i, (question, answer) in enumerate(QA_PAIRS):
        # Encode as: Q: <question> ? A: <answer> .
        q_tokens = encode_text(question, vocab)
        a_tokens = encode_text(answer, vocab)

        # Combine into single sequence
        sequence = q_tokens + a_tokens

        psam.train_batch(sequence)

        if i < 3:  # Show first few
            print(f"   [{i+1}] Q: {question}")
            print(f"       A: {answer}")
            print(f"       Tokens: {sequence[:10]}{'...' if len(sequence) > 10 else ''}")

    psam.finalize_training()
    print(f"✓ Trained on {len(QA_PAIRS)} Q&A pairs\n")

    # Get statistics
    stats = psam.stats()
    print("📊 Model Statistics:")
    print(f"   - Vocabulary: {stats.vocab_size} tokens")
    print(f"   - Rows: {stats.row_count}")
    print(f"   - Edges: {stats.edge_count}")
    print(f"   - Memory: {stats.memory_bytes} bytes ({stats.memory_bytes / 1024:.1f} KB)\n")

    # Test predictions on several questions
    print("🔮 Testing Predictions...")
    print("=" * 60)

    test_questions = [
        "Where is the cat?",
        "What is the dog doing?",
        "Where is the rabbit hiding?",
        "Who howled at the moon?",
    ]

    for test_q in test_questions:
        print(f"\nQ: {test_q}")

        # Expected answer
        expected = next((a for q, a in QA_PAIRS if q == test_q), None)
        print(f"Expected: {expected}")

        # Encode question as context
        q_tokens = encode_text(test_q, vocab)
        print(f"Context tokens: {q_tokens}")

        # Predict next tokens
        print("\nTop predictions:")
        pred_ids, scores, raw_strengths, support_counts, _ = psam.predict(q_tokens, max_predictions=10)

        for i, (token_id, score, support) in enumerate(zip(pred_ids, scores, support_counts)):
            word = decode_tokens([token_id], vocab)
            print(f"   {i+1}. {word} (token {token_id}, score: {score:.4f}, support: {support})")

        # Try to generate answer by sampling
        print("\nGenerated answer (sampling):")
        generated = q_tokens.copy()
        for _ in range(15):  # Generate up to 15 tokens
            if len(generated) == 0:
                break

            # Manual sampling since the built-in sample() has the same bug
            pred_ids, scores, _, _, _ = psam.predict(generated, max_predictions=psam.top_k)
            if len(pred_ids) == 0:
                break

            # Apply temperature and sample
            import numpy as np
            temperature = 0.5
            logits = scores / temperature
            logits = logits - np.max(logits)
            exp_scores = np.exp(logits)
            probs = exp_scores / np.sum(exp_scores)
            next_token = int(np.random.choice(pred_ids, p=probs))

            generated.append(next_token)
            # Stop at period
            if decode_tokens([next_token], vocab) == '.':
                break

        generated_text = decode_tokens(generated, vocab)
        print(f"   {generated_text}")
        print("-" * 60)

    # Cleanup
    psam.destroy()

    print("\n🎉 Test complete!\n")
    return 0

if __name__ == "__main__":
    exit(main())

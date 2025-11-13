#!/usr/bin/env python3
"""
Debug specifically the "What is the dog doing?" question
"""

from psam import PSAM, PSAMGenerator, ResidualConfig, is_library_available
import numpy as np

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
    if not is_library_available():
        print("❌ Native library not available")
        return 1

    vocab = build_vocab(QA_PAIRS)
    reverse_vocab = {v: k for k, v in vocab.items()}

    VOCAB_SIZE = len(vocab) + 10
    WINDOW = 8
    TOP_K = 20

    psam = PSAM(vocab_size=VOCAB_SIZE, window=WINDOW, top_k=TOP_K)

    # Train
    print("Training...")
    for question, answer in QA_PAIRS:
        q_tokens = encode_text(question, vocab)
        a_tokens = encode_text(answer, vocab)
        sequence = q_tokens + a_tokens
        psam.train_batch(sequence)

    psam.finalize_training()
    print("Training complete\n")

    # Focus on the dog question
    test_q = "What is the dog doing?"
    expected_a = "The dog is rolling on the rug."

    print("="*70)
    print(f"Q: {test_q}")
    print(f"Expected: {expected_a}")
    print("="*70)

    q_tokens = encode_text(test_q, vocab)
    print(f"\nQuestion tokens: {q_tokens}")
    print(f"Question: {decode_tokens(q_tokens, vocab)}\n")

    residual_config = ResidualConfig(
        max_lookahead=5,
        residual_decay=0.85,
        residual_blend=0.6,
        enable=True
    )

    # Create stateful generator
    generator = PSAMGenerator(psam, residual_config=residual_config)

    context = q_tokens.copy()

    print("Generating step-by-step:\n")

    np.random.seed(42)

    for step in range(15):
        pred_ids, scores, _, _, _ = generator.predict(context, max_predictions=10)

        if len(pred_ids) == 0:
            break

        # Show top 5
        print(f"Step {step+1} - Context: ...{' '.join([reverse_vocab.get(t, f'<{t}>') for t in context[-3:]])}")
        print(f"  Top 5:")
        for i in range(min(5, len(pred_ids))):
            word = reverse_vocab.get(pred_ids[i], f'<{pred_ids[i]}>')
            print(f"    {i+1}. '{word}' (score: {scores[i]:.2f})")

        # Sample with temperature
        temperature = 0.2
        logits = scores / temperature
        logits = logits - np.max(logits)
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores)
        next_token = int(np.random.choice(pred_ids, p=probs))

        word = reverse_vocab.get(next_token, f'<{next_token}>')
        print(f"  → Sampled: '{word}' (prob: {probs[pred_ids.index(next_token)]:.4f})")
        print()

        context.append(next_token)

        if word == '.':
            break

    answer_tokens = context[len(q_tokens):]
    answer = decode_tokens(answer_tokens, vocab)

    print("="*70)
    print(f"Generated answer: {answer}")
    print(f"Expected answer:  the dog is rolling on the rug .")
    print("="*70)

    if 'dog' in answer:
        print("✅ Has 'dog'")
    else:
        print("❌ Missing 'dog'")

    if 'rolling' in answer:
        print("✅ Has 'rolling'")
    else:
        print("❌ Missing 'rolling'")

    generator.destroy()
    psam.destroy()
    return 0

if __name__ == "__main__":
    exit(main())

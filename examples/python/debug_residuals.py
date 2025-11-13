#!/usr/bin/env python3
"""
Debug residual activation to see what's happening
"""

from psam import PSAM, ResidualConfig, is_library_available

QA_PAIRS = [
    ("What is the dog doing?", "The dog is rolling on the rug."),
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
    print("Residual Activation Debug\n")

    if not is_library_available():
        print("❌ Native library not available")
        return 1

    vocab = build_vocab(QA_PAIRS)
    reverse_vocab = {v: k for k, v in vocab.items()}

    print("Vocabulary:")
    for word, idx in sorted(vocab.items(), key=lambda x: x[1]):
        print(f"  {idx}: {word}")
    print()

    psam = PSAM(vocab_size=len(vocab) + 10, window=8, top_k=20)

    # Train
    for question, answer in QA_PAIRS:
        q_tokens = encode_text(question, vocab)
        a_tokens = encode_text(answer, vocab)
        sequence = q_tokens + a_tokens
        psam.train_batch(sequence)
        print(f"Training sequence: {sequence}")
        print(f"  = {decode_tokens(sequence, vocab)}\n")

    psam.finalize_training()

    # Test different context lengths
    test_q = "What is the dog doing?"
    q_tokens = encode_text(test_q, vocab)

    print(f"Question: {test_q}")
    print(f"Tokens: {q_tokens}\n")

    residual_config = ResidualConfig(
        max_lookahead=4,
        residual_decay=0.9,
        residual_blend=1.0,  # Max blend
        enable=True
    )

    contexts_to_test = [
        (q_tokens, "Full question"),
        (q_tokens + [vocab['the']], "Q + 'the'"),
        (q_tokens + [vocab['the'], vocab['dog']], "Q + 'the dog'"),
    ]

    for context, desc in contexts_to_test:
        print(f"{'='*60}")
        print(f"Context: {desc}")
        print(f"Tokens: {context} = {decode_tokens(context, vocab)}")
        print(f"Predicting position: {len(context)}\n")

        # Without residuals
        pred_ids, scores, _, _, _ = psam.predict(context, max_predictions=10)
        print("WITHOUT residuals:")
        for i, (tid, score) in enumerate(zip(pred_ids[:5], scores[:5])):
            word = reverse_vocab[tid]
            print(f"  {i+1}. '{word}' (score: {score:.2f})")

        # With residuals
        pred_ids_res, scores_res, _, _, _ = psam.predict(
            context, max_predictions=10, residual_config=residual_config
        )
        print("\nWITH residuals:")
        for i, (tid, score) in enumerate(zip(pred_ids_res[:5], scores_res[:5])):
            word = reverse_vocab[tid]
            # Show if prediction changed
            marker = ""
            if i < len(pred_ids) and tid != pred_ids[i]:
                marker = " ← CHANGED!"
            print(f"  {i+1}. '{word}' (score: {score:.2f}){marker}")

        print()

    psam.destroy()
    return 0

if __name__ == "__main__":
    exit(main())

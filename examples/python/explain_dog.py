#!/usr/bin/env python3
"""
Use explain() to see what 'dog' from the question predicts
"""

from psam import PSAM, is_library_available

QA_PAIRS = [
    ("Where is the cat?", "The cat is on the mat."),
    ("What is the dog doing?", "The dog is rolling on the rug."),
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

def main():
    if not is_library_available():
        print("❌ Native library not available")
        return 1

    vocab = build_vocab(QA_PAIRS)
    reverse_vocab = {v: k for k, v in vocab.items()}

    psam = PSAM(vocab_size=len(vocab) + 10, window=8, top_k=20)

    # Train
    for question, answer in QA_PAIRS:
        q_tokens = encode_text(question, vocab)
        a_tokens = encode_text(answer, vocab)
        sequence = q_tokens + a_tokens
        psam.train_batch(sequence)
        print(f"Trained: {' '.join([reverse_vocab[t] for t in sequence])}")

    psam.finalize_training()

    # Check what associations 'dog' has
    dog_id = vocab['dog']
    horse_id = vocab['horse']

    print("\n" + "="*70)
    print("ANALYSIS: After question, predicting next word")
    print("="*70)

    # Context: "What is the dog doing ? the"
    # We want to predict the next word (should be 'dog' ideally)

    test_q = "What is the dog doing?"
    q_tokens = encode_text(test_q, vocab)
    context = q_tokens + [vocab['the']]  # Add 'the'

    print(f"\nContext: {' '.join([reverse_vocab[t] for t in context])}")
    print(f"Tokens: {context}")
    print()

    # Explain why we predict 'dog'
    print("Why does it predict 'dog'?")
    result_dog = psam.explain(context, vocab['dog'], max_terms=10)
    print(f"  Total score: {result_dog.total:.2f}")
    print(f"  Bias: {result_dog.bias:.2f}")
    print(f"  Terms ({result_dog.term_count}):")
    for term in result_dog.terms[:10]:
        src_word = reverse_vocab.get(term.source, f'<{term.source}>')
        print(f"    '{src_word}' @ offset {term.offset}: contrib={term.contribution:.2f} (weight={term.weight:.2f}, idf={term.idf:.2f}, decay={term.decay:.2f})")

    print()

    # Explain why we predict 'horse'
    print("Why does it predict 'horse'?")
    result_horse = psam.explain(context, vocab['horse'], max_terms=10)
    print(f"  Total score: {result_horse.total:.2f}")
    print(f"  Bias: {result_horse.bias:.2f}")
    print(f"  Terms ({result_horse.term_count}):")
    for term in result_horse.terms[:10]:
        src_word = reverse_vocab.get(term.source, f'<{term.source}>')
        print(f"    '{src_word}' @ offset {term.offset}: contrib={term.contribution:.2f} (weight={term.weight:.2f}, idf={term.idf:.2f}, decay={term.decay:.2f})")

    psam.destroy()
    return 0

if __name__ == "__main__":
    exit(main())

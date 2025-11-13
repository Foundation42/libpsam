#!/usr/bin/env python3
"""
Test salience tracking on the Q&A dataset
"""

from psam import PSAM, PSAMGenerator, SalienceConfig, is_library_available
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
    print("🎓 Training...")
    for question, answer in QA_PAIRS:
        q_tokens = encode_text(question, vocab)
        a_tokens = encode_text(answer, vocab)
        sequence = q_tokens + a_tokens
        psam.train_batch(sequence)

    psam.finalize_training()
    print("✓ Training complete\n")

    # Focus on the dog question which fails without salience
    test_q = "What is the dog doing?"
    expected_a = "The dog is rolling on the rug."

    print("="*70)
    print(f"Q: {test_q}")
    print(f"Expected: {expected_a}")
    print("="*70)

    q_tokens = encode_text(test_q, vocab)

    # Test 1: Baseline (no salience)
    print("\n📊 TEST 1: BASELINE (No Salience)")
    print("-"*70)

    generator_baseline = PSAMGenerator(psam, salience_config=None)
    context = q_tokens.copy()
    np.random.seed(42)

    for step in range(10):
        pred_ids, scores, _, _, _ = generator_baseline.predict(context, max_predictions=10)

        if len(pred_ids) == 0:
            break

        # Show top 3
        if step < 3:
            print(f"Step {step+1}: Top 3: ", end="")
            for i in range(min(3, len(pred_ids))):
                w = reverse_vocab.get(pred_ids[i], f'<{pred_ids[i]}>')
                print(f"{w}({scores[i]:.1f}) ", end="")
            print()

        temperature = 0.2
        logits = scores / temperature
        logits = logits - np.max(logits)
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores)
        next_token = int(np.random.choice(pred_ids, p=probs))

        word = reverse_vocab.get(next_token, f'<{next_token}>')
        context.append(next_token)

        if word == '.':
            break

    answer_baseline = decode_tokens(context[len(q_tokens):], vocab)
    print(f"\nBaseline answer: {answer_baseline}")

    has_dog_baseline = 'dog' in answer_baseline
    has_rolling_baseline = 'rolling' in answer_baseline
    print(f"  Has 'dog': {'✅' if has_dog_baseline else '❌'}")
    print(f"  Has 'rolling': {'✅' if has_rolling_baseline else '❌'}")

    generator_baseline.destroy()

    # Test 2: With Salience
    print("\n\n🎯 TEST 2: WITH SALIENCE TRACKING")
    print("-"*70)

    salience_config = SalienceConfig(
        max_anchors=16,
        ewma_freq_halflife=128.0,
        ewma_contrib_halflife=64.0,
        eta=1.0,
        kappa=0.25,
        beta=0.5,  # Higher beta to give anchors more weight
        pop_decay_distance=256.0,
        min_salience=0.05,  # Lower threshold
        enable=True
    )

    generator_salience = PSAMGenerator(psam, salience_config=salience_config)
    context = q_tokens.copy()
    np.random.seed(42)

    for step in range(10):
        pred_ids, scores, _, _, _ = generator_salience.predict(context, max_predictions=10)

        if len(pred_ids) == 0:
            break

        # Show top 3
        if step < 3:
            print(f"Step {step+1}: Top 3: ", end="")
            for i in range(min(3, len(pred_ids))):
                w = reverse_vocab.get(pred_ids[i], f'<{pred_ids[i]}>')
                print(f"{w}({scores[i]:.1f}) ", end="")
            print()

        temperature = 0.2
        logits = scores / temperature
        logits = logits - np.max(logits)
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores)
        next_token = int(np.random.choice(pred_ids, p=probs))

        word = reverse_vocab.get(next_token, f'<{next_token}>')
        context.append(next_token)

        if word == '.':
            break

    answer_salience = decode_tokens(context[len(q_tokens):], vocab)
    print(f"\nSalience answer: {answer_salience}")

    has_dog_salience = 'dog' in answer_salience
    has_rolling_salience = 'rolling' in answer_salience
    print(f"  Has 'dog': {'✅' if has_dog_salience else '❌'}")
    print(f"  Has 'rolling': {'✅' if has_rolling_salience else '❌'}")

    generator_salience.destroy()

    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)

    print(f"\nBaseline:  {answer_baseline}")
    print(f"Salience:  {answer_salience}")
    print(f"Expected:  the dog is rolling on the rug .")

    if has_dog_salience and not has_dog_baseline:
        print("\n🎉 SALIENCE FIXED THE SUBJECT!")
    elif has_rolling_salience and not has_rolling_baseline:
        print("\n🎉 SALIENCE FIXED THE ACTION!")
    elif has_dog_salience and has_rolling_salience:
        print("\n🎉🎉 SALIENCE GOT BOTH CORRECT!")
    elif not has_dog_salience and not has_dog_baseline:
        print("\n📊 No improvement (both still wrong)")
    else:
        print("\n⚠️  Different results but not a clear win")

    psam.destroy()
    return 0

if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Test residual activation on the FULL 25 Q&A pairs dataset
"""

from psam import PSAM, ResidualConfig, is_library_available
import numpy as np

# Full 25 Q&A pairs
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

def extract_subject(question):
    """Extract the main subject from a question"""
    q_lower = question.lower()

    # List of animals/subjects in our dataset
    subjects = [
        'cat', 'dog', 'bird', 'frog', 'fox', 'fish', 'bee', 'mouse', 'snake',
        'horse', 'squirrel', 'owl', 'rabbit', 'wolf', 'lion', 'penguin', 'bear',
        'kangaroo', 'monkey', 'seal', 'dragon', 'unicorn', 'mermaid', 'octopus', 'panda'
    ]

    for subject in subjects:
        if subject in q_lower:
            return subject

    return None

def generate_answer(psam, question_tokens, vocab, max_length=20, use_residuals=False):
    """Generate answer using standard or residual prediction"""
    reverse_vocab = {v: k for k, v in vocab.items()}
    generated = question_tokens.copy()

    residual_config = None
    if use_residuals:
        residual_config = ResidualConfig(
            max_lookahead=5,
            residual_decay=0.85,
            residual_blend=0.6,
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

        # Sample with low temperature for more deterministic output
        temperature = 0.2
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
    print("║   PSAM Residuals - Full 25 Q&A Dataset Test               ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    if not is_library_available():
        print("❌ Native library not available")
        return 1

    vocab = build_vocab(QA_PAIRS)

    print(f"📚 Dataset: {len(QA_PAIRS)} Q&A pairs")
    print(f"📝 Vocabulary: {len(vocab)} unique words\n")

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

    # Test on a subset of questions
    test_questions = [
        "Where is the cat?",
        "What is the dog doing?",
        "Where is the rabbit hiding?",
        "Who howled at the moon?",
        "What is the horse doing?",
        "Where is the bear fishing?",
        "What is the mouse nibbling?",
    ]

    print("="*70)
    print("TESTING ON SAMPLE QUESTIONS")
    print("="*70)

    results_baseline = []
    results_residual = []

    for test_q in test_questions:
        # Find expected answer
        expected = next((a for q, a in QA_PAIRS if q == test_q), None)
        subject = extract_subject(test_q)

        print(f"\n{'─'*70}")
        print(f"Q: {test_q}")
        print(f"Expected: {expected}")
        print(f"Subject: {subject}")

        q_tokens = encode_text(test_q, vocab)

        # Test WITHOUT residuals
        np.random.seed(42)
        generated_baseline = generate_answer(psam, q_tokens, vocab, use_residuals=False)
        answer_baseline = decode_tokens(generated_baseline[len(q_tokens):], vocab)

        baseline_has_subject = subject and subject in answer_baseline.lower()
        results_baseline.append(baseline_has_subject)

        print(f"\nBaseline:  {answer_baseline}")
        print(f"  Subject match: {'✅' if baseline_has_subject else '❌'}")

        # Test WITH residuals
        np.random.seed(42)
        generated_residual = generate_answer(psam, q_tokens, vocab, use_residuals=True)
        answer_residual = decode_tokens(generated_residual[len(q_tokens):], vocab)

        residual_has_subject = subject and subject in answer_residual.lower()
        results_residual.append(residual_has_subject)

        print(f"Residuals: {answer_residual}")
        print(f"  Subject match: {'✅' if residual_has_subject else '❌'}")

        # Highlight improvement
        if residual_has_subject and not baseline_has_subject:
            print("  🎉 RESIDUALS FIXED IT!")
        elif baseline_has_subject and not residual_has_subject:
            print("  ⚠️  Residuals made it worse")

    # Summary statistics
    print("\n" + "="*70)
    print("📊 RESULTS SUMMARY")
    print("="*70)

    baseline_correct = sum(results_baseline)
    residual_correct = sum(results_residual)
    total = len(test_questions)

    print(f"\nBaseline (no residuals):")
    print(f"  Correct subject: {baseline_correct}/{total} ({baseline_correct/total*100:.1f}%)")

    print(f"\nWith Residuals:")
    print(f"  Correct subject: {residual_correct}/{total} ({residual_correct/total*100:.1f}%)")

    improvement = residual_correct - baseline_correct
    print(f"\nImprovement: {improvement:+d} questions")

    if improvement > 0:
        print(f"🎉 Residuals improved accuracy by {improvement} questions!")
    elif improvement == 0:
        print("📊 Residuals maintained same accuracy (no regression)")
    else:
        print(f"⚠️  Residuals decreased accuracy by {abs(improvement)} questions")

    print("\n" + "="*70)
    print("💡 NOTES:")
    print("="*70)
    print("• Residuals help when the subject appears in both Q and A")
    print("• The offset mismatch problem is being addressed")
    print("• For full sequential generation, we need stateful residual buffers")
    print("• Tuning lookahead/decay/blend parameters could improve results further")

    psam.destroy()
    return 0

if __name__ == "__main__":
    exit(main())

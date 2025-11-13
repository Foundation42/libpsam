#!/usr/bin/env python3
"""
Analyze the offset problem in Q&A prediction
"""

from psam import PSAM, is_library_available

def main():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║          PSAM Q&A Offset Analysis                          ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    print("🔍 THE PROBLEM:")
    print("="*70)
    print("\nTraining sequence:")
    print("   [what is the dog doing ? the dog is rolling on the rug .]")
    print("    0    1  2   3   4     5  6   7   8  9       10  11  12  13")
    print()
    print("When we train, PSAM learns:")
    print("   'dog'@3 → 'dog'@7 with offset +4")
    print("   'doing'@4 → 'dog'@7 with offset +3")
    print("   '?'@5 → 'the'@6 with offset +1")
    print("   '?'@5 → 'dog'@7 with offset +2")
    print()
    print("But when we query with: [what is the dog doing ?]")
    print("                          0    1  2   3   4     5")
    print()
    print("We're trying to predict position 6 (after the question).")
    print()
    print("The offsets needed for position 6:")
    print("   'dog'@3 needs offset +3 to reach position 6")
    print("   '?'@5 needs offset +1 to reach position 6")
    print()
    print("But what was learned:")
    print("   'dog'@3 has association with offset +4 (not +3!)")
    print("   '?'@5 has association with offset +1 ✓ (this works!)")
    print()
    print("So PSAM finds:")
    print("   ✓ '?' → 'the' (offset +1 matches!)")
    print("   ✗ 'dog' → 'dog' (offset +3 needed, but +4 was learned)")
    print()
    print("="*70)
    print()
    print("💡 THE SOLUTION:")
    print()
    print("We need to make the positions align! Options:")
    print()
    print("1. Remove the question mark from the training sequence:")
    print("   [what is the dog doing the dog is rolling ...]")
    print("    0    1  2   3   4     5   6   7  8       ...")
    print("   Now: 'dog'@3 → 'dog'@6 with offset +3 ✓")
    print()
    print("2. Don't include the question mark in the query context:")
    print("   Query: [what is the dog doing]")
    print("           0    1  2   3   4")
    print("   Predict position 5")
    print("   'dog'@3 → position 5 needs offset +2")
    print("   But we learned offset +4 from training... still wrong!")
    print()
    print("3. Use a separator token approach:")
    print("   Train: [what is the dog doing <SEP> the dog is rolling ...]")
    print("   Query: [what is the dog doing <SEP>]")
    print("   The <SEP> token creates a clear boundary")
    print()
    print("4. Better solution: Don't concatenate Q+A at all!")
    print("   Instead, train on JUST the answers:")
    print("   [the dog is rolling on the rug]")
    print("   [the horse is galloping across the field]")
    print()
    print("   Then at inference time, use the question to prime context")
    print("   and let PSAM generate from the learned answer patterns.")
    print()
    print("="*70)
    print()
    print("🧪 Let's test approach #4:")
    print()

    if not is_library_available():
        print("❌ Native library not available")
        return 1

    # Build vocabulary
    vocab = {
        "the": 1, "dog": 2, "is": 3, "rolling": 4, "on": 5, "rug": 6, ".": 7,
        "horse": 8, "galloping": 9, "across": 10, "field": 11,
        "cat": 12, "sleeping": 13, "mat": 14,
        "what": 15, "doing": 16, "?": 17
    }
    reverse_vocab = {v: k for k, v in vocab.items()}

    # Train only on ANSWERS
    answers = [
        [1, 2, 3, 4, 5, 1, 6, 7],      # the dog is rolling on the rug .
        [1, 8, 3, 9, 10, 1, 11, 7],    # the horse is galloping across the field .
        [1, 12, 3, 13, 5, 1, 14, 7],   # the cat is sleeping on the mat .
    ]

    psam = PSAM(vocab_size=len(vocab) + 5, window=8, top_k=20)

    print("Training on answers only:")
    for answer in answers:
        psam.train_batch(answer)
        words = [reverse_vocab[t] for t in answer]
        print(f"   {' '.join(words)}")

    psam.finalize_training()
    print("\n✓ Training complete\n")

    # Now query with just key words from the question
    print("Testing: Query with key content words from question\n")

    # For "What is the dog doing?", the key word is "dog"
    test_contexts = [
        ([2], "dog"),           # Just "dog"
        ([2, 16], "dog doing"), # "dog doing"
    ]

    for context, desc in test_contexts:
        print(f"Context: {desc} = {context}")

        pred_ids, scores, _, support_counts, _ = psam.predict(context, max_predictions=8)

        print(f"  Top predictions:")
        for i, (token_id, score, support) in enumerate(zip(pred_ids, scores, support_counts)):
            word = reverse_vocab.get(token_id, f'<UNK:{token_id}>')
            print(f"    {i+1}. '{word}' (score: {score:.2f}, support: {support})")
        print()

    print("="*70)
    print("\n📊 CONCLUSION:")
    print()
    print("The offset mismatch is the root cause!")
    print()
    print("For Q&A tasks with PSAM, you need to think carefully about:")
    print("  1. How questions and answers are positioned in training")
    print("  2. What context you provide at inference time")
    print("  3. Whether the offsets will line up")
    print()
    print("The concatenated Q+A approach creates offset mismatches unless")
    print("the query context length exactly matches the training positions.")

    psam.destroy()
    return 0

if __name__ == "__main__":
    exit(main())

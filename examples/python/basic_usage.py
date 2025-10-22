#!/usr/bin/env python3
"""
Basic usage example for libpsam

Demonstrates:
- Creating a model
- Training on token sequences
- Making predictions
- Saving and loading
"""

from psam import PSAM, is_library_available

def main():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║          libpsam - Python Basic Usage Example             ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    # Check if native library is available
    if not is_library_available():
        print("❌ Native library not available")
        print("   Build libpsam.so and set LIBPSAM_PATH environment variable")
        return 1

    print("✓ Native library loaded\n")

    # Configuration
    VOCAB_SIZE = 100
    WINDOW = 8
    TOP_K = 10

    print("📦 Creating PSAM model...")
    print(f"   - Vocabulary size: {VOCAB_SIZE}")
    print(f"   - Window: {WINDOW}")
    print(f"   - Top-K: {TOP_K}\n")

    psam = PSAM(vocab_size=VOCAB_SIZE, window=WINDOW, top_k=TOP_K)

    # Training data: "the quick brown fox jumps over the lazy dog"
    tokens = [1, 2, 3, 4, 5, 6, 1, 7, 8]

    print("📚 Training on sequence...")
    print(f"   Tokens: {tokens}\n")

    psam.train_batch(tokens)
    psam.finalize_training()

    print("✓ Training complete!\n")

    # Get statistics
    stats = psam.stats()
    print("📊 Model Statistics:")
    print(f"   - Vocabulary: {stats.vocab_size} tokens")
    print(f"   - Rows: {stats.row_count}")
    print(f"   - Edges: {stats.edge_count}")
    print(f"   - Memory: {stats.memory_bytes} bytes ({stats.memory_bytes / 1024:.1f} KB)\n")

    # Make predictions
    print("🔮 Making predictions...")
    context = [1, 2, 3]  # "the quick brown"
    print(f"   Context: {context}\n")

    token_ids, scores = psam.predict(context, max_predictions=5)

    print("   Predictions:")
    for i, (token_id, score) in enumerate(zip(token_ids, scores)):
        print(f"   {i + 1}. Token {token_id} (score: {score:.3f})")
    print()

    # Sample from distribution
    print("🎲 Sampling with different temperatures...")
    for temp in [0.5, 1.0, 2.0]:
        sampled = psam.sample(context, temperature=temp)
        print(f"   T={temp}: Token {sampled}")
    print()

    # Save model
    filepath = "example_model.psam"
    print(f"💾 Saving model to '{filepath}'...")

    psam.save(filepath)
    print("✓ Model saved!\n")

    # Load model back
    print(f"📂 Loading model from '{filepath}'...")
    loaded = PSAM.load(filepath)
    print("✓ Model loaded!\n")

    # Verify loaded model works
    print("🔮 Testing loaded model...")
    loaded_ids, loaded_scores = loaded.predict(context, max_predictions=3)
    print(f"✓ Loaded model works! First prediction: Token {loaded_ids[0]}\n")

    # Cleanup
    psam.destroy()
    loaded.destroy()

    print("🎉 Example complete!\n")
    print(f"libpsam version: {PSAM.version()}")

    return 0

if __name__ == "__main__":
    exit(main())

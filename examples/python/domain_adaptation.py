#!/usr/bin/env python3
"""
Domain adaptation example using layer composition

Demonstrates hot-swappable domain layers
"""

from psam import PSAM, is_library_available

def main():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║          libpsam - Domain Adaptation Example              ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    if not is_library_available():
        print("❌ Native library not available")
        return 1

    VOCAB_SIZE = 100
    WINDOW = 8
    TOP_K = 5

    # Create base model (general domain)
    print("📦 Creating base model (general domain)...")
    base = PSAM(vocab_size=VOCAB_SIZE, window=WINDOW, top_k=TOP_K)

    # General text: "the cat sat on the mat"
    general_tokens = [1, 2, 3, 4, 1, 5]
    base.train_batch(general_tokens)
    base.finalize_training()
    print("✓ Base model trained\n")

    # Create medical domain model
    print("📦 Creating medical domain model...")
    medical = PSAM(vocab_size=VOCAB_SIZE, window=WINDOW, top_k=TOP_K)

    # Medical text: "the patient has acute pain"
    medical_tokens = [1, 10, 11, 12, 13]
    medical.train_batch(medical_tokens)
    medical.finalize_training()
    print("✓ Medical model trained\n")

    # Create legal domain model
    print("📦 Creating legal domain model...")
    legal = PSAM(vocab_size=VOCAB_SIZE, window=WINDOW, top_k=TOP_K)

    # Legal text: "the plaintiff claims damages"
    legal_tokens = [1, 20, 21, 22]
    legal.train_batch(legal_tokens)
    legal.finalize_training()
    print("✓ Legal model trained\n")

    # Test context
    context = [1]  # "the"

    # Predictions with base model only
    print("🔮 Predictions (base model only):")
    token_ids, scores = base.predict(context, max_predictions=TOP_K)
    for i, (tid, score) in enumerate(zip(token_ids, scores)):
        print(f"   {i + 1}. Token {tid} ({score:.3f})")
    print()

    # Add medical layer
    print("➕ Adding medical layer (weight: 1.5)...")
    base.add_layer("medical", medical, weight=1.5)

    print("🔮 Predictions (base + medical):")
    token_ids, scores = base.predict(context, max_predictions=TOP_K)
    for i, (tid, score) in enumerate(zip(token_ids, scores)):
        print(f"   {i + 1}. Token {tid} ({score:.3f})")
    print()

    # Update medical layer weight
    print("⚙️  Updating medical layer weight to 2.0...")
    base.update_layer_weight("medical", new_weight=2.0)

    print("🔮 Predictions (base + medical 2.0×):")
    token_ids, scores = base.predict(context, max_predictions=TOP_K)
    for i, (tid, score) in enumerate(zip(token_ids, scores)):
        print(f"   {i + 1}. Token {tid} ({score:.3f})")
    print()

    # Remove medical, add legal
    print("🔄 Switching to legal domain...")
    base.remove_layer("medical")
    base.add_layer("legal", legal, weight=1.5)

    print("🔮 Predictions (base + legal):")
    token_ids, scores = base.predict(context, max_predictions=TOP_K)
    for i, (tid, score) in enumerate(zip(token_ids, scores)):
        print(f"   {i + 1}. Token {tid} ({score:.3f})")
    print()

    print("🎉 Domain adaptation demo complete!")

    # Cleanup
    base.destroy()
    medical.destroy()
    legal.destroy()

    return 0

if __name__ == "__main__":
    exit(main())

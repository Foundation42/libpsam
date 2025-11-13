#!/usr/bin/env python3
"""
Stateful residual generation - maintains residual buffer across generation steps

This is the KEY to making residuals work for sequential generation!
"""

from psam import PSAM, ResidualConfig, is_library_available
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

# Test with smaller dataset first
QA_PAIRS = [
    ("What is the dog doing?", "The dog is rolling on the rug."),
    ("What is the horse doing?", "The horse is galloping across the field."),
    ("What is the cat doing?", "The cat is sleeping on the mat."),
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

@dataclass
class DeferredAssociation:
    """Represents a single deferred association"""
    candidate: int
    contribution: float
    remaining_offset: int

class StatefulResidualGenerator:
    """
    Maintains residual state across generation steps.

    This simulates what a C-level stateful implementation would do.
    """
    def __init__(self, psam, residual_config):
        self.psam = psam
        self.config = residual_config
        self.residual_buffer: List[DeferredAssociation] = []

    def predict_with_state(self, context: List[int], max_predictions: int = 20):
        """
        Predict next token while maintaining residual state

        Returns: (token_ids, scores, raw_strengths, support_counts, probabilities)
        """
        # Get base predictions
        pred_ids, scores, raw_strengths, support_counts, probs = self.psam.predict(
            context,
            max_predictions=max_predictions,
            residual_config=None  # We'll apply residuals manually
        )

        # Apply residuals that are ready to fire (remaining_offset == 0)
        scores_with_residuals = scores.copy()
        token_to_idx = {tid: i for i, tid in enumerate(pred_ids)}

        residuals_to_keep = []
        for assoc in self.residual_buffer:
            if assoc.remaining_offset == 0:
                # Fire now!
                if assoc.candidate in token_to_idx:
                    idx = token_to_idx[assoc.candidate]
                    scores_with_residuals[idx] += assoc.contribution
                # Don't keep this residual
            else:
                # Age it and keep it
                assoc.remaining_offset -= 1
                residuals_to_keep.append(assoc)

        self.residual_buffer = residuals_to_keep

        # Compute NEW residuals for future positions
        self._compute_future_residuals(context)

        # Re-sort by new scores
        scored_preds = [(tid, score) for tid, score in zip(pred_ids, scores_with_residuals)]
        scored_preds.sort(key=lambda x: -x[1])

        final_ids = [tid for tid, _ in scored_preds]
        final_scores = np.array([score for _, score in scored_preds], dtype=np.float32)

        return final_ids, final_scores, raw_strengths, support_counts, probs

    def _compute_future_residuals(self, context: List[int]):
        """
        Compute deferred associations for future positions

        This is a Python simulation of what the C code does internally.
        We'll use the PSAM explain API to discover associations.
        """
        if not self.config.enable:
            return

        vocab_size = self.psam.vocab_size
        window = self.psam.window

        # For each context token, look for associations at future offsets
        for ctx_idx, token in enumerate(context):
            current_offset = len(context) - ctx_idx

            # Look ahead
            for lookahead in range(1, self.config.max_lookahead + 1):
                future_offset = current_offset + lookahead

                if future_offset > window:
                    break

                # We need to discover what associations exist at this offset
                # This is tricky in Python without direct CSR access
                # For now, we'll use a heuristic: predict from a synthetic context

                # Create a synthetic context that puts this token at the right distance
                # This is approximate but demonstrates the concept
                if future_offset <= window:
                    # Simulate by creating padding + token
                    synthetic_context = [0] * (future_offset - 1) + [token]

                    try:
                        # Predict to see what this token would activate
                        future_ids, future_scores, _, _, _ = self.psam.predict(
                            synthetic_context,
                            max_predictions=10,
                            residual_config=None
                        )

                        # Add top associations as residuals
                        defer_decay = pow(self.config.residual_decay, lookahead)

                        for fid, fscore in zip(future_ids[:5], future_scores[:5]):
                            if fscore > 0:  # Only positive contributions
                                contribution = fscore * defer_decay * self.config.residual_blend * 0.1

                                # Add to residual buffer
                                self.residual_buffer.append(DeferredAssociation(
                                    candidate=fid,
                                    contribution=contribution,
                                    remaining_offset=lookahead - 1
                                ))
                    except:
                        pass  # Skip if prediction fails

def generate_with_stateful_residuals(psam, question_tokens, vocab, residual_config, max_length=20):
    """Generate answer using stateful residual buffer"""
    reverse_vocab = {v: k for k, v in vocab.items()}
    generated = question_tokens.copy()

    # Create stateful generator
    generator = StatefulResidualGenerator(psam, residual_config)

    print(f"  Starting generation with stateful residuals...")
    print(f"  Initial context: {decode_tokens(question_tokens, vocab)}")

    for step in range(max_length):
        pred_ids, scores, _, _, _ = generator.predict_with_state(generated, max_predictions=10)

        if len(pred_ids) == 0:
            break

        # Sample with low temperature
        temperature = 0.2
        logits = scores / temperature
        logits = logits - np.max(logits)
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores)
        next_token = int(np.random.choice(pred_ids, p=probs))

        word = reverse_vocab.get(next_token, f'<{next_token}>')

        # Show top 3 predictions for this step
        if step < 5:  # Show first few steps
            print(f"  Step {step+1}: Top 3: ", end="")
            for i in range(min(3, len(pred_ids))):
                w = reverse_vocab.get(pred_ids[i], f'<{pred_ids[i]}>')
                print(f"{w}({scores[i]:.1f}) ", end="")
            print(f" → Generated: '{word}'")

        generated.append(next_token)
        if word == '.':
            break

    return generated

def main():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     Stateful Residual Generation Test                     ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    if not is_library_available():
        print("❌ Native library not available")
        return 1

    vocab = build_vocab(QA_PAIRS)
    reverse_vocab = {v: k for k, v in vocab.items()}

    psam = PSAM(vocab_size=len(vocab) + 10, window=8, top_k=20)

    # Train
    print("🎓 Training...")
    for question, answer in QA_PAIRS:
        q_tokens = encode_text(question, vocab)
        a_tokens = encode_text(answer, vocab)
        sequence = q_tokens + a_tokens
        psam.train_batch(sequence)

    psam.finalize_training()
    print("✓ Training complete\n")

    # Test
    test_q = "What is the dog doing?"
    print("="*70)
    print(f"TEST: {test_q}")
    print("="*70)

    q_tokens = encode_text(test_q, vocab)

    residual_config = ResidualConfig(
        max_lookahead=4,
        residual_decay=0.9,
        residual_blend=0.8,
        enable=True
    )

    print("\nGenerating with STATEFUL residuals:\n")
    np.random.seed(42)
    generated = generate_with_stateful_residuals(psam, q_tokens, vocab, residual_config)

    result = decode_tokens(generated, vocab)
    answer = decode_tokens(generated[len(q_tokens):], vocab)

    print(f"\nFull output: {result}")
    print(f"Answer only: {answer}")

    if 'dog' in answer and 'rolling' in answer:
        print("\n✅ SUCCESS! Got the right subject and action!")
    elif 'dog' in answer:
        print("\n✓ Partial success - got the subject")
    else:
        print("\n❌ Still getting wrong subject")

    psam.destroy()
    return 0

if __name__ == "__main__":
    exit(main())

# Residual/Deferred Activation for PSAM

## Problem Statement

When training PSAM on concatenated Q&A sequences, offset mismatches prevent proper predictions:

**Training:**
```
[what is the dog doing ? the dog is rolling ...]
 0    1  2   3   4     5  6   7   8  9
```
Learns: `dog@3 → dog@7` (offset +4)

**Inference:**
```
[what is the dog doing ?]
 0    1  2   3   4     5
```
Predicting position 6 needs: `dog@3 → position 6` (offset +3)

**The mismatch:** PSAM learned offset +4, but needs offset +3 to fire!

## Core Insight

The association `dog@3 → dog@7` (offset +4) **exists and is strong**, but it's "waiting" for the right moment to fire. Instead of discarding it at position 6, we should **defer** it and apply it when we reach position 7.

## Proposed Solution: Residual Activation Buffer

### High-Level Concept

Maintain a **residual buffer** that tracks associations that would fire at future positions:

```
Position 6: Predicting next token
  Current associations (offset matches):
    ? → the (offset +1) ✓ fires now

  Deferred associations (future offsets):
    dog → dog (offset +4) → Store in residual[+3]
    doing → dog (offset +3) → Store in residual[+2]

Position 7: Predicting next token
  Current associations:
    the → cat/horse/dog

  Residual from previous position:
    dog → dog (was deferred, now fires!) ✓
    doing → dog (still deferred to residual[+1])
```

### Implementation Strategy

#### 1. Data Structure

```c
typedef struct {
    uint32_t candidate;      // Token being boosted
    float contribution;      // Score contribution
    int remaining_offset;    // How many positions until it fires
} DeferredAssociation;

typedef struct {
    DeferredAssociation* associations;
    size_t count;
    size_t capacity;
} ResidualBuffer;
```

#### 2. Algorithm (during prediction)

For each prediction at position `j`:

**Step 1: Compute current-position scores (existing logic)**
```c
for each context_token at position i {
    offset = j - i;
    if (offset >= 1 && offset <= window) {
        // Look up associations with this offset
        associations = get_associations(context_token, offset);

        for each association {
            score[association.target] += IDF × decay × PPMI;
        }
    }
}
```

**Step 2: Compute deferred activations (NEW)**
```c
for each context_token at position i {
    for (future_offset = 1; future_offset <= MAX_LOOKAHEAD; future_offset++) {
        actual_offset = (j + future_offset) - i;

        if (actual_offset >= 1 && actual_offset <= window) {
            associations = get_associations(context_token, actual_offset);

            for each association {
                deferred_score = IDF × decay × PPMI × RESIDUAL_DECAY^future_offset;

                add_to_residual_buffer(
                    association.target,
                    deferred_score,
                    future_offset
                );
            }
        }
    }
}
```

**Step 3: Apply residuals from previous positions**
```c
for each residual in buffer where remaining_offset == 0 {
    score[residual.candidate] += residual.contribution;
}
```

**Step 4: Age residuals**
```c
for each residual in buffer {
    residual.remaining_offset -= 1;

    if (residual.remaining_offset < 0) {
        // Remove expired residual
        remove_from_buffer(residual);
    }
}
```

#### 3. Key Parameters

- **`MAX_LOOKAHEAD`**: How many positions ahead to compute (default: 3-5)
- **`RESIDUAL_DECAY`**: Decay factor for deferred activations (default: 0.7-0.9)
- **`RESIDUAL_BLEND`**: Weight for residual vs current scores (default: 0.3-0.5)

### Expected Behavior

With residuals, the Q&A prediction becomes:

**Position 6:** Predict after "doing ?"
- Current: `? → the` (strong)
- Deferred: `dog → dog` stored for position +1
- **Prediction: "the"** ✓

**Position 7:** Predict after "doing ? the"
- Current: `the → cat/horse/dog` (all similar)
- **Residual fires: `dog → dog`** (from question context!)
- **Prediction: "dog"** with residual boost ✓

**Position 8:** Predict after "doing ? the dog"
- Current: `dog → is` (strong)
- Residuals: Various deferred associations
- **Prediction: "is"** ✓

### Implementation Variants

#### Variant A: Stateful Generation (Simple)
- Maintain residual buffer across generation steps
- Easy to implement
- Only works during sequential generation

#### Variant B: Multi-Position Lookahead (Complex)
- Predict N tokens simultaneously with full residual graph
- Allows beam search with residuals
- More complex but more powerful

#### Variant C: Residual Layers (Modular)
- Make residuals a composable layer
- Can enable/disable per model
- Fits existing layer composition architecture

## Implementation Roadmap

### Phase 1: Core Infrastructure
- [ ] Add `get_associations_by_offset()` to C API
- [ ] Implement `ResidualBuffer` data structure
- [ ] Add `psam_predict_with_residuals()` function

### Phase 2: Python Bindings
- [ ] Expose residual prediction to Python
- [ ] Add `enable_residuals` parameter to predict()
- [ ] Create test suite

### Phase 3: Tuning & Optimization
- [ ] Benchmark performance impact
- [ ] Tune hyperparameters (decay, lookahead)
- [ ] Optimize memory usage
- [ ] Add SIMD for residual computation

### Phase 4: Advanced Features
- [ ] Residual beam search
- [ ] Per-layer residual control
- [ ] Adaptive lookahead based on context

## Example Use Cases

### 1. Q&A Systems (this issue)
- Question context influences answer generation
- Deferred subject/object associations

### 2. Long-Range Dependencies
- Sentence-initial topic affects sentence-final words
- Narrative coherence across long contexts

### 3. Code Generation
- Function name influences variable names throughout
- Import statements affect later code

### 4. Multi-Turn Dialogue
- Earlier conversation turns influence later responses
- Topic continuity across turns

## Performance Considerations

### Computational Cost
- **Current**: O(context_length × top_k)
- **With residuals**: O(context_length × top_k × lookahead)
- Expected overhead: 2-5× with lookahead=3

### Memory Cost
- **Residual buffer**: O(lookahead × candidates) per position
- Typical: ~5KB per prediction with lookahead=3, top_k=20

### Mitigation Strategies
- Adaptive lookahead (increase only when needed)
- Sparse residuals (only track high-confidence associations)
- Batch residual computation (SIMD/vectorized)

## Alternative Approaches Considered

### 1. Offset Fuzzing
**Idea:** Allow associations to fire with nearby offsets (±1)
**Pros:** Simple, no state
**Cons:** Loses position-specific precision, may introduce noise

### 2. Variable-Length Context
**Idea:** Train with variable question lengths
**Pros:** Makes offsets more robust
**Cons:** Doesn't solve the fundamental issue, requires more data

### 3. Explicit Position Embeddings
**Idea:** Add position tokens to vocabulary
**Pros:** Makes positions explicit
**Cons:** Increases vocabulary, breaks interpretability

### 4. Residual Activation (Chosen)
**Idea:** Defer associations until offsets match
**Pros:** Maintains position-specific precision, interpretable, flexible
**Cons:** Requires state during generation, some overhead

## Open Questions

1. **Residual accumulation:** Should residuals from multiple positions accumulate or replace?

2. **Decay function:** Linear, exponential, or learned decay?

3. **Threshold:** Should very weak residuals be pruned?

4. **Negative residuals:** Can we subtract/suppress unwanted continuations?

5. **Cross-layer residuals:** Should residuals propagate across composed layers?

## Success Metrics

To validate this approach:

1. **Q&A accuracy:** Test on the animal dataset
   - Target: >90% correct subject prediction

2. **Long-range dependencies:** Benchmark on tasks requiring window > 8
   - Compare with/without residuals

3. **Performance overhead:** Measure latency increase
   - Target: <3× slowdown with lookahead=3

4. **Generalization:** Test on unseen Q&A patterns
   - Ensure residuals don't overfit

## References

- **Neural cache models**: Cache activations for future use (Grave et al., 2017)
- **Persistent memory networks**: Maintain state across predictions (Sukhbaatar et al., 2015)
- **Attention residuals**: Carry forward attention weights (Transformers)
- **Recursive neural networks**: Maintain hidden state across sequence

## Conclusion

Residual/deferred activation is a natural extension of PSAM's position-specific association model. By tracking "future" associations and applying them when offsets match, we can solve the Q&A offset mismatch problem while maintaining PSAM's interpretability and efficiency.

The mechanism is conceptually simple: "remember what you wanted to say and say it when the time is right."

---

**Author:** Christian Beaumont
**Contributors:** Claude Code (Anthropic)
**Date:** 2025-01-05
**Status:** Design Proposal

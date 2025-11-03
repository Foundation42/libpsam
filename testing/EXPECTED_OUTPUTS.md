# Expected Test Outputs

This document shows what successful test results should look like.

## Test 1: Pure Layer Dominance (Weight A=1.0, B=0.0)

### Input
```bash
psam predict --model test_01_composed.psamc --prompt "apple" --top-k 10
```

### Expected Output
```
# Top 10 predictions for prompt: "apple"

ant         0.3245
arrow       0.2891
apple       0.1456
anchor      0.0923
atlas       0.0612
axe         0.0487
arch        0.0234
angel       0.0152

# Notice: 100% of predictions are from Vocabulary A
# No B-vocabulary tokens (ball, bat, bear, etc.) appear
```

### Verification Output
```
============================================================
PREDICTION ANALYSIS
============================================================

Top Predictions:
  ant          0.3245  [A]
  arrow        0.2891  [A]
  apple        0.1456  [A]
  anchor       0.0923  [A]
  atlas        0.0612  [A]
  axe          0.0487  [A]
  arch         0.0234  [A]
  angel        0.0152  [A]

Probability Mass by Vocabulary:
  A       : 1.0000 (100.0%) ██████████████████████████████████████████████████
  B       : 0.0000 (  0.0%)

Expected vs Actual:
  ✓ A: Expected 100.0%, Got 100.0% (Δ  +0.0%)
  ✓ B: Expected   0.0%, Got   0.0% (Δ  +0.0%)
============================================================

RESULT: ✓ PASS
```

---

## Test 2: Equal Blend (Weight A=0.5, B=0.5)

### Input
```bash
psam predict --model test_02_composed.psamc --prompt "apple" --top-k 10
```

### Expected Output
```
# Top 10 predictions for prompt: "apple"

ant         0.1623  # Vocab A
ball        0.1445  # Vocab B
arrow       0.1445  # Vocab A
bat         0.1223  # Vocab B
bear        0.0978  # Vocab B
apple       0.0728  # Vocab A
anchor      0.0462  # Vocab A
boat        0.0412  # Vocab B
atlas       0.0306  # Vocab A
bell        0.0298  # Vocab B

# Notice: Mix of A and B vocabulary tokens
# Roughly 50/50 distribution
```

### Verification Output
```
============================================================
PREDICTION ANALYSIS
============================================================

Top Predictions:
  ant          0.1623  [A]
  ball         0.1445  [B]
  arrow        0.1445  [A]
  bat          0.1223  [B]
  bear         0.0978  [B]
  apple        0.0728  [A]
  anchor       0.0462  [A]
  boat         0.0412  [B]
  atlas        0.0306  [A]
  bell         0.0298  [B]

Probability Mass by Vocabulary:
  A       : 0.4564 ( 48.7%) ████████████████████████
  B       : 0.4806 ( 51.3%) █████████████████████████

Expected vs Actual:
  ✓ A: Expected  50.0%, Got  48.7% (Δ  -1.3%)
  ✓ B: Expected  50.0%, Got  51.3% (Δ  +1.3%)
============================================================

RESULT: ✓ PASS (within tolerance)
```

---

## Test 3: Weighted Blend (Weight A=0.7, B=0.3)

### Expected Output
```
============================================================
PREDICTION ANALYSIS
============================================================

Probability Mass by Vocabulary:
  A       : 0.6823 ( 69.2%) ██████████████████████████████████
  B       : 0.3042 ( 30.8%) ███████████████

Expected vs Actual:
  ✓ A: Expected  70.0%, Got  69.2% (Δ  -0.8%)
  ✓ B: Expected  30.0%, Got  30.8% (Δ  +0.8%)
============================================================

RESULT: ✓ PASS
```

---

## Test 4: Three-Layer Blend (Weights 0.5, 0.3, 0.2)

### Expected Output
```
# Mix of A, B, and C vocabulary tokens

Probability Mass by Vocabulary:
  A       : 0.5134 ( 51.3%) █████████████████████████
  B       : 0.2912 ( 29.1%) ██████████████
  C       : 0.1954 ( 19.6%) █████████

Expected vs Actual:
  ✓ A: Expected  50.0%, Got  51.3% (Δ  +1.3%)
  ✓ B: Expected  30.0%, Got  29.1% (Δ  -0.9%)
  ✓ C: Expected  20.0%, Got  19.6% (Δ  -0.4%)

RESULT: ✓ PASS
```

---

## Weight Sweep: Smooth Transition

### Expected Visualization
```
Weight Sweep Visualization
======================================================================

Vocabulary Distribution vs Weight
(X-axis: Weight of Layer A, Y-axis: Probability Mass %)

100.0 |*                                                          
 90.0 |*                                                          
 80.0 | *                                                         
 70.0 | *                                                         
 60.0 |  *                        +                              
 50.0 |   *                      ++                              
 40.0 |    *                   ++                                
 30.0 |     *                ++                                  
 20.0 |      **            ++                                    
 10.0 |        **        ++                                      
  0.0 |          *******+                                        
      +-----------------------------------------------------------
       0.00  0.20  0.40  0.60  0.80  1.00

Legend:
  * = Vocab A
  + = Vocab B

Detailed Results:
----------------------------------------------------------------------
Weight A | Weight B | Vocab A% | Vocab B% | Expected A% | Δ
----------------------------------------------------------------------
  1.00  |   0.00  |  100.0% |    0.0% |    100.0% |  +0.0% ✓
  0.80  |   0.20  |   79.8% |   20.2% |     80.0% |  -0.2% ✓
  0.60  |   0.40  |   60.1% |   39.9% |     60.0% |  +0.1% ✓
  0.50  |   0.50  |   48.7% |   51.3% |     50.0% |  -1.3% ✓
  0.40  |   0.60  |   39.2% |   60.8% |     40.0% |  -0.8% ✓
  0.20  |   0.80  |   19.9% |   80.1% |     20.0% |  -0.1% ✓
  0.00  |   1.00  |    0.0% |  100.0% |      0.0% |  +0.0% ✓
----------------------------------------------------------------------

RESULT: ✓ ALL PASS - Smooth linear transition observed
```

---

## Test 5: Markov Transitions

### Layer A Pattern
```
Training sequence shows:
- apple → ant (90% of the time)
- ant → arrow (90% of the time)
- arrow → apple (90% of the time)
```

### Layer B Pattern
```
Training sequence shows:
- ball → bat (90% of the time)
- bat → bear (90% of the time)
- bear → ball (90% of the time)
```

### Pure Layer A (weight 1.0, 0.0)
```bash
Prompt: "apple"

Top predictions:
  ant     0.8234  # Strong transition (0.9 learned)
  arrow   0.0912
  anchor  0.0234

✓ Shows pure A pattern preserved
```

### Blended (weight 0.5, 0.5)
```bash
Prompt: "apple"

Top predictions:
  ant     0.4117  # A-pattern (weakened)
  bat     0.3723  # B-pattern appearing!
  arrow   0.0456  # A-pattern
  bear    0.0389  # B-pattern appearing!

✓ Shows both patterns mixing
```

**Key Insight**: When layers are blended, you should see BOTH "ant" (from A's pattern) and "bat" (from B's pattern) appearing, even though they're from different vocabularies. This proves the composition is actually mixing the learned transition probabilities, not just the token distributions.

---

## Common Failure Modes

### ✗ FAIL: Composition Not Working
```
# If composition is broken, you might see:

Weight A=0.5, B=0.5, but predictions show:

Probability Mass:
  A: 100.0%  # Should be ~50%
  B:   0.0%  # Should be ~50%

→ Indicates layer B is not being used at all
```

### ✗ FAIL: Non-Linear Blending
```
# Weight sweep shows:

Weight A | Vocab A%
---------|----------
  1.00   |  100%
  0.80   |  100%  ← Should be ~80%
  0.60   |   98%  ← Should be ~60%
  0.50   |   52%
  0.40   |    2%  ← Sudden drop
  0.20   |    0%
  0.00   |    0%

→ Indicates numerical instability or incorrect blending math
```

### ✗ FAIL: Wrong Vocabulary
```
# Pure A layer (weight 1.0, 0.0) predicting:

ball    0.2345  # B-vocab shouldn't appear!
bat     0.1234  # B-vocab shouldn't appear!
ant     0.0891  # A-vocab (correct but low probability)

→ Indicates model mixup or incorrect layer loading
```

---

## Debugging Checklist

When tests fail:

1. **Check model files exist**
   ```bash
   ls -lh composition_test_models/*.psam
   ```

2. **Verify vocabularies match**
   ```bash
   psam vocab layer_a.psam
   psam vocab layer_b.psam
   ```

3. **Test individual layers first**
   ```bash
   psam predict --model layer_a.psam --prompt "apple"
   # Should show 100% A-vocab
   ```

4. **Check composition metadata**
   ```bash
   psam info composition.psamc
   # Should list all layers with correct weights
   ```

5. **Try simple weight combinations**
   - Start with (1.0, 0.0) - should match pure layer
   - Then try (0.0, 1.0) - should match other pure layer
   - Then try (0.5, 0.5) - should show mixing

---

## What Success Looks Like

✓ Pure layers show 100% of their vocabulary  
✓ Blended layers show proportional mixing  
✓ Weight changes produce smooth transitions  
✓ Zero weights completely eliminate influence  
✓ Learned patterns (like Markov transitions) are preserved in blends  
✓ Three+ layer compositions respect all weights  

If all these work with simple vocabularies like "apple" and "ball", then composition is working correctly and will work with complex real-world data!
# PSAM Composition Testing Framework

A comprehensive testing framework for validating PSAM's composition capabilities using ultra-simple synthetic vocabularies for transparent, verifiable behavior.

## Philosophy

Instead of testing with complex real-world data where behavior is ambiguous, this framework uses **deliberately simple vocabularies** where compositional effects are immediately visible:

- **Vocab A**: apple, ant, arrow, anchor, atlas, axe, angel, arch
- **Vocab B**: ball, bat, bear, boat, bell, bird, bone, bread
- **Vocab C**: cat, car, cave, coin, crown, cloud, cup, cliff
- **Vocab D**: dog, drum, duck, door, desk, dirt, dove, dune

With these disjoint vocabularies, you can **see at a glance** if composition is working:
- Pure A-weighted model predicting "ball"? → Bug!
- 50/50 blend showing 80% A-tokens? → Bug!
- Smooth weight transition? → Success!

## Quick Start

### 1. Generate Test Data
```bash
python synthetic_composition_test.py
```

This creates:
- `synthetic_test_suite/` - 8 test cases with training data
- `synthetic_test_suite/vocabs/` - Vocabulary files
- `synthetic_test_suite/README.md` - Detailed test documentation

### 2. Run Full Test Suite
```bash
./run_composition_tests.sh
```

This will:
- Train models for each layer
- Create compositions with various weight configurations
- Test predictions
- Verify results against expected behavior
- Run a weight sweep experiment

### 3. Visualize Results
```bash
python visualize_sweep.py --results-dir composition_test_results
```

Creates ASCII plots showing how vocabulary distribution changes with layer weights.

## Automated Test Harnesses

### Python (pytest)
- Requires `pytest` and a built `libpsam` shared library (the Python binding now auto-detects `build/core/Release/libpsam.so*`).
- Run from the repo root:
  ```bash
  pytest testing/python
  ```
- Exercises training, layered composition, and sampler normalization through the Python ctypes bindings.

### Bun / TypeScript
- Requires Bun ≥1.3 and the native JS bindings (`pnpm install`/`npm install` already done for development).
- Execute:
  ```bash
  bun test testing/bun/composition.test.ts
  ```
- Validates the FFI bindings, ensuring layered composites behave correctly when consumed from the JavaScript API.

## Test Cases

### Test 1: Pure Layer Dominance ✓
**Purpose**: Verify single layer with weight=1.0 completely dominates

**Setup**:
- Layer A trained on: "apple ant arrow apple ant arrow..."
- Layer B trained on: "ball bear bat ball bear bat..."
- Composition: Weight A=1.0, Weight B=0.0

**Expected**: ONLY A-vocabulary tokens in predictions

**Pass Criteria**: 100% of probability mass on A-tokens

---

### Test 2: Equal Blend ✓
**Purpose**: Verify 50/50 blend shows equal representation

**Setup**: Same models, weights (0.5, 0.5)

**Expected**: Roughly equal A and B tokens

**Pass Criteria**: 40-60% probability mass on each vocabulary

---

### Test 3: Weighted Blend (70/30) ✓
**Purpose**: Verify proportional blending

**Setup**: Weights (0.7, 0.3)

**Expected**: ~70% A-tokens, ~30% B-tokens

**Pass Criteria**: Within ±15% of target distribution

---

### Test 4: Three-Layer Blend ✓
**Purpose**: Test multi-layer composition

**Setup**: Three layers with weights (0.5, 0.3, 0.2)

**Expected**: Probability mass proportional to weights

**Pass Criteria**: Each vocab within ±15% of target

---

### Test 5: Markov Transitions ✓
**Purpose**: Test blending of learned patterns

**Setup**:
- Layer A: Strong apple→ant transition (0.9)
- Layer B: Strong ball→bat transition (0.9)

**Expected**: Blended model shows both patterns

**Pass Criteria**: Mixed predictions reflect both transition matrices

---

### Test 6: Zero Weight ✓
**Purpose**: Verify weight=0 eliminates layer influence

**Setup**: Weights (1.0, 0.0)

**Expected**: Identical to Test 1

**Pass Criteria**: Zero probability on B-tokens

---

### Test 7: Domain Alternation 🔄
**Purpose**: Test sequences switching between domains

**Setup**: Training data alternates A/B chunks

**Expected**: Model learns domain boundaries

**Evaluation**: Manual inspection of context-dependent behavior

---

### Test 8: Uniform Baseline 📊
**Purpose**: Baseline with no learned patterns

**Setup**: Completely random token sequences

**Expected**: Roughly uniform predictions

**Evaluation**: Entropy analysis

---

## File Structure

```
.
├── synthetic_composition_test.py    # Generate test data
├── verify_composition.py            # Analyze prediction results
├── visualize_sweep.py              # Visualize weight sweeps
├── run_composition_tests.sh        # Automated test runner
│
├── synthetic_test_suite/
│   ├── test_01_pure_dominance/
│   │   ├── layer_a.txt             # Training data
│   │   ├── layer_b.txt
│   │   └── metadata.json           # Test expectations
│   ├── test_02_equal_blend/
│   ├── ... (8 tests total)
│   └── vocabs/
│       ├── vocab_A.tsv
│       ├── vocab_B.tsv
│       ├── vocab_A_B.tsv           # Unified vocabularies
│       └── ...
│
├── composition_test_models/         # (Created during testing)
│   ├── test_01_pure_dominance_layer_a.psam
│   ├── test_01_pure_dominance_composed.psamc
│   └── ...
│
└── composition_test_results/        # (Created during testing)
    ├── test_01_pure_dominance_predictions.txt
    ├── sweep_1.0_0.0_predictions.txt
    └── ...
```

## Manual Testing

### Train a Single Model
```bash
psam train \
    --input synthetic_test_suite/test_01_pure_dominance/layer_a.txt \
    --vocab synthetic_test_suite/vocabs/vocab_A_B.tsv \
    --output my_model.psam \
    --n-gram 3
```

### Create a Composition
```bash
psam compose \
    --layer models/layer_a.psam 0.7 \
    --layer models/layer_b.psam 0.3 \
    --output my_composition.psamc
```

### Test Predictions
```bash
psam predict \
    --model my_composition.psamc \
    --prompt "apple" \
    --top-k 20 \
    | python verify_composition.py --stdin --test test_03_weighted_blend
```

## Weight Sweep Experiment

Test smooth interpolation between pure A and pure B:

```bash
# Train base models once
psam train --input test_01/layer_a.txt --vocab vocab_A_B.tsv --output a.psam
psam train --input test_01/layer_b.txt --vocab vocab_A_B.tsv --output b.psam

# Test different weight combinations
for w in 1.0 0.8 0.6 0.5 0.4 0.2 0.0; do
    wb=$(echo "1.0 - $w" | bc -l)
    
    psam compose --layer a.psam $w --layer b.psam $wb --output sweep_${w}.psamc
    psam predict --model sweep_${w}.psamc --prompt "apple" --top-k 20
done
```

Expected: Smooth, linear transition of probability mass from A to B vocabulary.

## Verification Metrics

### 1. Vocabulary Distribution
Measure percentage of probability mass on each vocabulary:
```
Vocab A: 70.3% ████████████████████████
Vocab B: 29.7% ██████████
```

### 2. Expected vs Actual
```
✓ A: Expected 70.0%, Got 70.3% (Δ +0.3%)
✓ B: Expected 30.0%, Got 29.7% (Δ -0.3%)
```

### 3. Linearity Check
Plot weight vs distribution - should be linear for proper blending.

### 4. Transition Fidelity
For Markov tests, verify learned transitions are preserved in blend.

## Advanced Testing

### Test Aligned Compositions
```bash
# Train models with different vocabularies
psam train --input a.txt --vocab vocab_A.tsv --output a.psam
psam train --input b.txt --vocab vocab_B.tsv --output b.psam

# Create aligned composition
psam compose --from-vocabs \
    --layer a.psam 0.5 \
    --layer b.psam 0.5 \
    --output aligned.psamc

# Test (vocab auto-discovered from .psamc)
psam predict --model aligned.psamc --prompt "apple"
```

### Test Dynamic Weight Adjustment
If your implementation supports runtime weight changes:
```javascript
const composite = loadComposite('my.psamc');
composite.setLayerWeight('medical', 1.8);  // Boost medical layer
composite.predict(context);
```

### Test FIM (Fill-in-Middle)
```bash
psam compose \
    --layer base.psam 1.0 \
    --layer creative.psam 0.5 \
    --with-fim fim_model.psam \
    --output fim_composition.psamc
```

## Debugging Tips

### Problem: Composition predicts only one vocabulary
**Check**: Are both models actually loaded? Are weights non-zero?
```bash
psam info composition.psamc  # Should show all layers
```

### Problem: Distribution doesn't match weights
**Check**: Is vocabulary alignment working? Are models trained on same vocab?
```bash
# Verify vocabularies match
diff <(psam vocab a.psam) <(psam vocab b.psam)
```

### Problem: Sudden jumps in weight sweep
**Check**: Numerical precision issues? Try smaller weight increments.
**Check**: Are models overfitted? Try more training data.

### Problem: Zero predictions for some prompts
**Check**: Is prompt tokenizing correctly?
```bash
psam tokenize --vocab vocab.tsv --text "apple ant"
```

## CI/CD Integration

Add to your test pipeline:
```yaml
- name: Test PSAM Composition
  run: |
    python synthetic_composition_test.py
    ./run_composition_tests.sh
    python verify_composition.py --test test_01_pure_dominance \
      --input results/test_01_predictions.txt
```

## Performance Benchmarking

Time composition operations:
```bash
time psam compose --layer a.psam 0.5 --layer b.psam 0.5 --output test.psamc
time psam predict --model test.psamc --prompt "apple" --top-k 100
```

Expected: Composition overhead should be minimal (<5% compared to single model).

## Extending the Framework

### Add New Vocabulary Sets
Edit `synthetic_composition_test.py`:
```python
VOCAB_SETS = {
    'E': ['eagle', 'earth', 'egg', 'elbow', ...],
    'F': ['fire', 'fish', 'flag', 'fork', ...],
}
```

### Add New Test Cases
```python
test_new = CompositionTestCase(
    name='test_09_my_new_test',
    description='Test some new behavior'
)
# Add layers and prompts
tests.append(test_new)
```

### Add New Metrics
Edit `verify_composition.py`:
```python
def compute_entropy(predictions):
    # Your metric here
    pass
```

## Expected Behavior Summary

| Test | Weight A | Weight B | Expected A% | Expected B% | Tolerance |
|------|----------|----------|-------------|-------------|-----------|
| 1    | 1.0      | 0.0      | 100%        | 0%          | ±5%       |
| 2    | 0.5      | 0.5      | 50%         | 50%         | ±15%      |
| 3    | 0.7      | 0.3      | 70%         | 30%         | ±15%      |
| 4    | 0.5/0.3/0.2 | -     | 50/30/20%   | -           | ±15%      |
| 6    | 1.0      | 0.0      | 100%        | 0%          | ±5%       |

## Success Criteria

Composition is working correctly if:

✓ Pure layers (weight 1.0) → 100% from that vocabulary  
✓ Equal blend (0.5/0.5) → ~50/50 split (±15%)  
✓ Weighted blend → proportional to weights (±15%)  
✓ Zero weight → complete elimination  
✓ Weight sweep → smooth linear transition  
✓ Three+ layers → all weights respected  

## FAQ

**Q: Why such simple vocabularies?**  
A: Complex data obscures bugs. With 8-word vocabularies, you can **see** composition working (or not) at a glance.

**Q: Why ±15% tolerance?**  
A: Accounts for statistical variation in small vocabularies. Tighten for production.

**Q: What if I don't have the PSAM CLI?**  
A: Use the Python/JS bindings directly with the generated .txt files.

**Q: Can I use this for other architectures?**  
A: Yes! The synthetic data approach works for any compositional system.

## License

This testing framework is part of the PSAM project. Use freely for testing and development.

## Contributing

Improvements welcome:
- Additional test cases
- Better visualization
- More sophisticated metrics
- Edge case coverage

## Contact

For questions about PSAM composition or this test framework, see the main PSAM documentation.

---

**Remember**: If you can't see composition working with "apple" and "ball", it won't work with Shakespeare and medical texts!

# Quick Start Guide - PSAM Composition Testing

Get testing in 5 minutes!

## Step 1: Generate Test Data (30 seconds)

```bash
python synthetic_composition_test.py
```

**Output**: Creates `synthetic_test_suite/` with 8 test cases and vocabularies.

## Step 2: Train Your First Models (2 minutes)

```bash
cd synthetic_test_suite

# Train Layer A (apple, ant, arrow...)
psam train \
    --input test_01_pure_dominance/layer_a.txt \
    --vocab vocabs/vocab_A_B.tsv \
    --output ../models/layer_a.psam \
    --n-gram 3

# Train Layer B (ball, bat, bear...)
psam train \
    --input test_01_pure_dominance/layer_b.txt \
    --vocab vocabs/vocab_A_B.tsv \
    --output ../models/layer_b.psam \
    --n-gram 3

cd ..
```

## Step 3: Test Pure Layer (1 minute)

```bash
# Create composition with pure A (weight 1.0, 0.0)
psam compose \
    --layer models/layer_a.psam 1.0 \
    --layer models/layer_b.psam 0.0 \
    --output compositions/pure_a.psamc

# Test prediction
psam predict \
    --model compositions/pure_a.psamc \
    --prompt "apple" \
    --top-k 10
```

**Expected**: Should predict ONLY A-vocabulary (ant, arrow, anchor...)

**Verify**:
```bash
psam predict --model compositions/pure_a.psamc --prompt "apple" --top-k 10 \
    | python verify_composition.py --stdin --test test_01_pure_dominance
```

## Step 4: Test Equal Blend (1 minute)

```bash
# Create 50/50 blend
psam compose \
    --layer models/layer_a.psam 0.5 \
    --layer models/layer_b.psam 0.5 \
    --output compositions/blend_50_50.psamc

# Test prediction
psam predict \
    --model compositions/blend_50_50.psamc \
    --prompt "apple" \
    --top-k 10 \
    | python verify_composition.py --stdin --test test_02_equal_blend
```

**Expected**: Should predict BOTH A and B vocabulary (~50/50 split)

## Step 5: Weight Sweep (2 minutes)

```bash
# Test multiple weight combinations
for w in 1.0 0.8 0.6 0.5 0.4 0.2 0.0; do
    wb=$(echo "1.0 - $w" | bc -l)
    
    psam compose \
        --layer models/layer_a.psam $w \
        --layer models/layer_b.psam $wb \
        --output compositions/sweep_${w}.psamc
    
    echo "Weight A=$w, B=$wb:"
    psam predict --model compositions/sweep_${w}.psamc --prompt "apple" --top-k 5
    echo
done
```

**Expected**: Smooth transition from pure A → pure B vocabulary

## What You Just Tested

✅ **Pure dominance**: Single layer with weight 1.0 works  
✅ **Weighted blending**: Two layers mix proportionally  
✅ **Weight sweep**: Smooth interpolation between layers  

## If Something Failed

### No B-vocabulary appearing in 50/50 blend?
```bash
# Check layer B works alone
psam compose --layer models/layer_b.psam 1.0 --output test_b.psamc
psam predict --model test_b.psamc --prompt "ball" --top-k 5
# Should show ONLY B-vocabulary (bat, bear, boat...)
```

### Wrong proportions in blend?
```bash
# Verify vocabulary files
head vocabs/vocab_A_B.tsv
# Should show: 0 apple, 1 ant, 2 arrow, ... 8 ball, 9 bat, etc.

# Check model sizes
psam info models/layer_a.psam
psam info models/layer_b.psam
```

### Crashes or errors?
```bash
# Check PSAM version
psam --version

# Validate training data
wc -w test_01_pure_dominance/layer_a.txt  # Should be 1000 tokens
head -n 50 test_01_pure_dominance/layer_a.txt  # Should show "apple ant arrow..." pattern
```

## Next Steps

### Run Full Test Suite
```bash
./run_composition_tests.sh
```

### Try Three Layers
```bash
# Generate more test data
cd synthetic_test_suite

psam train \
    --input test_04_three_layer/layer_a.txt \
    --vocab vocabs/vocab_A_B_C.tsv \
    --output ../models/three_a.psam

psam train \
    --input test_04_three_layer/layer_b.txt \
    --vocab vocabs/vocab_A_B_C.tsv \
    --output ../models/three_b.psam

psam train \
    --input test_04_three_layer/layer_c.txt \
    --vocab vocabs/vocab_A_B_C.tsv \
    --output ../models/three_c.psam

cd ..

# Compose with three layers
psam compose \
    --layer models/three_a.psam 0.5 \
    --layer models/three_b.psam 0.3 \
    --layer models/three_c.psam 0.2 \
    --output compositions/three_layer.psamc

# Test
psam predict --model compositions/three_layer.psamc --prompt "apple" --top-k 10 \
    | python verify_composition.py --stdin --test test_04_three_layer
```

**Expected**: 50% A-vocab, 30% B-vocab, 20% C-vocab

### Visualize Results
```bash
python visualize_sweep.py --results-dir composition_test_results
```

## Time Investment

- Initial setup: **5 minutes**
- Each additional test: **1-2 minutes**
- Full test suite: **10-15 minutes**

## Validation Checklist

Once tests pass, you have verified:

✅ Layer loading works  
✅ Weight mixing works  
✅ Composition save/load works  
✅ Prediction blending works  
✅ Zero-weight exclusion works  
✅ Multi-layer composition works  

**You're ready to test with real data!**

## Real-World Next Steps

```bash
# Train on your actual data
psam train --input shakespeare_tragedies.txt --vocab unified.tsv --output tragedy.psam
psam train --input shakespeare_comedies.txt --vocab unified.tsv --output comedy.psam

# Create genre-blended model
psam compose \
    --layer tragedy.psam 0.6 \
    --layer comedy.psam 0.4 \
    --output shakespeare_mixed.psamc

# Test with actual Shakespeare
psam predict --model shakespeare_mixed.psamc --prompt "to be or not to be"
```

---

**Remember**: If it works with "apple" and "ball", it'll work with real data. These simple tests prove the composition machinery itself is correct!
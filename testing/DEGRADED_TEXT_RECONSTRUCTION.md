# Degraded Text Reconstruction with PSAM Composition

A novel application of PSAM's composition capabilities: using layered models to "heal" corrupted or degraded text.

## The Idea

What if you could train one model on clean text and another on corrupted text, then blend them to help reconstruct degraded input? This is exactly what PSAM composition enables!

## How It Works

### 1. Text Corruption

The `TextCorruptor` class applies realistic degradations:

```python
corruptor = TextCorruptor(corruption_rate=0.15)
corrupted, stats = corruptor.corrupt_text("to be or not to be")
# Result: "t0 be 0r n0t to be" (with OCR-like errors)
```

**Corruption Types:**
- **Substitution**: OCR errors (`o→0`, `l→1`, `i→!`, `s→$`)
- **Transposition**: Typos (`th→ht`, `er→re`, `an→na`)
- **Deletion**: Missing characters (`hello→helo`)
- **Insertion**: Extra characters (`hello→heallo`)

### 2. Training Strategy

```
Clean Model:      Trained on pristine Shakespeare text
                  Learns: "to be or" → "not"

Corruption Model: Trained on degraded Shakespeare
                  Learns: "t0 be 0r" → "n0t"

Healing Model:    Composition (80% clean + 20% corruption)
                  Bridges both distributions
```

### 3. Composition for Healing

```python
# Create healing composite
healing = clean_model.create_layered_composite()
healing.set_base_weight(0.8)  # Favor clean patterns
healing.add_layer("corruption_aware", corrupt_model, 0.2)

# Test on degraded input
corrupted_context = "t0 be 0r"  # Corrupted version
predictions = healing.predict(corrupted_context)
# Can predict both clean and corrupted forms!
```

## Test Results

### Example Output

```
=== Corruption Statistics ===
Total tokens: 3000
Corrupted: 300 (10.0%)
By type: {
  'substitution': 71,
  'transposition': 70,
  'deletion': 65,
  'insertion': 95
}

=== Clean Model Predictions (context: 'to be or') ===
1. use             0.5118
2. ere             0.1168
3. thinking        0.0893
4. fire            0.0677
5. that            0.0627

=== Corruption Model Predictions (same context) ===
1. use             0.4556
2. fre             0.1136  ← Corrupted version of "ere"!
3. that            0.1087
4. ere             0.1032  ← Clean version still present
5. thinking        0.0789

=== Healing Model (80% clean + 20% corruption) ===
1. use             0.4460
2. ere             0.1311  ← Clean form ranks higher
3. thinking        0.1050
4. that            0.0875
5. air             0.0588
```

### Corruption Pattern Learning

The corruption model actually learns corruption patterns:

```
Context: 'hello'

Clean model predictions:
  ['world', 'lose', 'to', 'are', 'hello']

Corrupt model predictions:
  ['woarld', 'owrld', 'wolrd', 'w0r1d', 'w0rid']
  ↑ Various corruptions of "world"!
```

## Real-World Applications

This technique could be used for:

### 1. **OCR Error Correction**
- Train clean model on modern text
- Train corruption model on OCR output
- Use composition to suggest corrections

### 2. **Typo-Tolerant Search**
- Blend clean and typo-aware models
- Accept both `definately` and `definitely`
- Rank clean forms higher

### 3. **Historical Document Restoration**
- Train on clean modern texts
- Train on degraded historical scans
- Reconstruct damaged sections

### 4. **Noisy Social Media Text**
- Clean model: Formal language
- Corruption model: Internet slang, misspellings
- Bridge formal and informal usage

## Running the Tests

```bash
# Run degraded text reconstruction tests
pytest testing/python/test_degraded_reconstruction.py -v -s

# Test corruption utilities only
pytest testing/python/test_degraded_reconstruction.py::test_text_corruption_utilities

# Full Shakespeare reconstruction test
pytest testing/python/test_degraded_reconstruction.py::test_degraded_shakespeare_reconstruction
```

## Test Coverage

✅ **test_text_corruption_utilities**
- Validates corruption mechanisms
- Checks corruption statistics tracking
- Ensures reproducibility (seeded RNG)

✅ **test_degraded_shakespeare_reconstruction**
- Uses real Shakespeare text (Hamlet, ~3000 words)
- 10% corruption rate with mixed error types
- Tests clean, corrupted, and blended models
- Validates predictions on both clean and degraded contexts

✅ **test_corruption_pattern_learning**
- Proves corruption model learns actual corruption patterns
- Shows clear divergence from clean model
- Demonstrates pattern-based composition

## Key Insights

### 1. Composition Captures Degradation
The corruption model doesn't just "average" corrupted text—it actually learns the **patterns of corruption**:
- `world` → `woarld`, `owrld`, `w0r1d`
- `ere` → `fre`
- Systematic character substitutions

### 2. Blending Provides Flexibility
By adjusting composition weights, you can:
- **High clean weight (80%+)**: Prefer clean reconstructions
- **Balanced (50/50)**: Accept both clean and corrupted forms
- **High corruption weight (70%+)**: Understand highly degraded input

### 3. Context Matters
The same composition behaves differently based on input:
- Clean context → Clean predictions dominate
- Corrupted context → Both forms appear
- Mixed context → Adaptive behavior

## Extending the Framework

### Add New Corruption Types

```python
class AdvancedCorruptor(TextCorruptor):
    def corrupt_token(self, token):
        # Add word-level corruptions
        WORD_SUBSTITUTIONS = {
            'you': ['u', 'ya', 'yuu'],
            'are': ['r', 'ar'],
            'the': ['teh', 'th'],
        }
        # ... custom logic
```

### Try Different Text Sources

```python
# Medical texts with common abbreviations
clean_medical = load_medical_text()
abbreviated = apply_medical_abbreviations(clean_medical)

# Legal documents with OCR errors
clean_legal = load_legal_text()
scanned_legal = apply_ocr_corruption(clean_legal)
```

### Multi-Layer Corruption Models

```python
composite = clean_model.create_layered_composite()
composite.set_base_weight(0.6)
composite.add_layer("ocr_errors", ocr_model, 0.2)
composite.add_layer("typos", typo_model, 0.1)
composite.add_layer("abbreviations", abbrev_model, 0.1)
```

## Performance Notes

- **Training time**: ~0.1s for 3000 words (both models)
- **Prediction time**: ~1ms per prediction
- **Memory**: Minimal overhead for composition
- **Vocabulary**: Unified vocab handles both clean and corrupted tokens

## Why This Matters

This experiment demonstrates that **PSAM composition isn't just about blending topics or domains**—it can learn and bridge **different representations of the same content**:

- Clean vs. corrupted
- Formal vs. informal
- Modern vs. archaic
- Full words vs. abbreviations

The same composition mechanism that blends Shakespeare tragedies and comedies can also blend clean text with its degraded versions!

## Future Directions

1. **Iterative Healing**: Use predictions to refine corrupted text, then re-predict
2. **Confidence Scoring**: Weight composition based on degradation confidence
3. **Domain-Specific Corruption**: Train on actual OCR, SMS, or Tweet data
4. **Interactive Correction**: Human-in-the-loop with composition suggestions

## Citation

If you use this degraded text reconstruction approach in your research:

```bibtex
@software{psam_degradation_2025,
  title={Degraded Text Reconstruction using PSAM Composition},
  author={Foundation42},
  year={2025},
  url={https://github.com/Foundation42/libpsam}
}
```

---

**Remember**: If PSAM can blend Shakespeare and corrupted Shakespeare, it can blend any content with its degraded forms. This opens up exciting possibilities for robust text processing in noisy real-world conditions!

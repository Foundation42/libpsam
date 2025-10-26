# PSAM Sampler Guide

Complete guide to sampling and temperature control in PSAM.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Logit Transform Modes](#logit-transform-modes)
- [Temperature Ranges](#temperature-ranges)
- [Migration Guide](#migration-guide)
- [Best Practices](#best-practices)

---

## Overview

PSAM v1.1+ includes a configurable sampler that makes temperature and top-p sampling work intuitively in the 0.1-2.0 range. This is achieved through **z-score normalization** of prediction scores before applying temperature scaling.

### The Problem (Pre-1.1)

PSAM models return raw accumulated scores that can have very large absolute values (e.g., 300-400). When applying softmax directly:

```
exp(331) / (exp(331) + exp(321) + ...) ≈ 99.9%
```

The top prediction dominates completely, making temperature values of 0.1-2.0 ineffective. You needed temperatures of 10-100 to see variation.

### The Solution (v1.1+)

**Z-score normalization** transforms scores to have mean=0 and std=1 before temperature scaling:

```c
z_score = (score - mean) / std
logit = z_score / temperature
prob = softmax(logit)
```

This makes temperature behave like standard language models: 0.5 = more deterministic, 1.0 = balanced, 2.0 = more random.

---

## Quick Start

### C API

```c
#include <psam.h>

psam_sampler_t sampler = {
    .transform = PSAM_LOGIT_ZSCORE,  // default, enables intuitive temps
    .temperature = 0.8f,             // 0.1-2.0 range works well
    .top_k = 0,                      // 0 = use model default
    .top_p = 0.9f,                   // nucleus sampling
    .seed = 42                       // for reproducibility
};

psam_prediction_t preds[32];
int n = psam_predict_with_sampler(model, context, ctx_len, &sampler, preds, 32);

// Predictions now have both raw scores and calibrated probabilities
for (int i = 0; i < n; i++) {
    printf("Token %u: score=%.2f prob=%.4f\n",
           preds[i].token, preds[i].score, preds[i].calibrated_prob);
}
```

### CLI

```bash
# Default (zscore, temp=1.0, top_p=0.95)
psam predict --model hamlet.psam --prompt "To be" --vocab hamlet.tsv

# Low temperature (more deterministic)
psam predict --model hamlet.psam --prompt "To be" --vocab hamlet.tsv \
  --temperature 0.3 --logit-transform zscore

# High temperature (more random)
psam generate --model hamlet.psam --prompt "To be or not to" --vocab hamlet.tsv \
  --count 20 --temperature 1.5 --logit-transform zscore

# Legacy mode (for old behavior with high temps)
psam predict --model hamlet.psam --prompt "To be" --vocab hamlet.tsv \
  --temperature 100.0 --logit-transform legacy
```

---

## Logit Transform Modes

### ZSCORE (Default, Recommended)

**Per-step z-score normalization**: `(score - mean) / std`

- **Temperature range**: 0.1 - 2.0 works intuitively
- **Adapts** to context length, model size, and composition fan-in
- **Consistent behavior** across different models and domains

**When to use**: Almost always. This is the default for good reason.

**Example results** (Hamlet model, context "To be"):

```
Temperature 0.3:  Top token = 76.2% prob (peaked, deterministic)
Temperature 1.0:  Top token = 44.2% prob (balanced)
Temperature 2.0:  Top token = 25.3% prob (flat, diverse)
```

### RAW

**No normalization**, just numerical stability (max logit subtraction).

- **Temperature range**: Depends on model (usually 10-100 like legacy)
- **Use case**: When you need direct access to raw score distributions
- **Warning**: Behavior varies significantly across models

### LEGACY

**Preserves pre-1.1 behavior** exactly.

- **Temperature range**: 10-100 (same as old PSAM)
- **Use case**: Reproducing old experiments with identical outputs
- **Migration path**: For backward compatibility during transition

### CALIBRATED

**Reserved for future use** (composite-level calibration).

Currently behaves like ZSCORE. In future versions, will use per-composite calibration data when available.

---

## Temperature Ranges

### ZSCORE Mode (Recommended)

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| 0.1 - 0.5 | Very deterministic, peaked distribution | Extractive tasks, factual completion |
| 0.6 - 1.0 | Balanced, natural variation | General text generation |
| 1.0 - 1.5 | More diverse, exploratory | Creative writing, brainstorming |
| 1.5 - 2.0 | Very diverse, surprising | Maximum creativity, outlier generation |
| > 2.0 | Approaching uniform | Rarely useful |

### LEGACY Mode

| Temperature | Effect |
|-------------|--------|
| 1-10 | Extremely peaked (nearly deterministic) |
| 10-50 | Deterministic to balanced |
| 50-100 | Balanced to diverse |
| 100+ | Very diverse |

---

## Migration Guide

### From Pre-1.1 PSAM

**Scenario 1: Using old API without explicit temperature**

```c
// OLD (pre-1.1)
psam_predict(model, context, len, preds, 32);
// No temperature control, effectively temp=1.0 with LEGACY mode

// NEW (recommended)
psam_sampler_t sampler = {.transform = PSAM_LOGIT_ZSCORE, .temperature = 1.0f};
psam_predict_with_sampler(model, context, len, &sampler, preds, 32);
// Z-score normalization, temp=1.0 in intuitive range
```

**Scenario 2: Using high temperatures (50-100)**

```c
// OLD
// (used sample_prediction with temp=100 via custom code)

// NEW (option 1: LEGACY mode for exact reproduction)
psam_sampler_t sampler = {.transform = PSAM_LOGIT_LEGACY, .temperature = 100.0f};
psam_predict_with_sampler(model, context, len, &sampler, preds, 32);

// NEW (option 2: ZSCORE mode with intuitive temp, recommended)
psam_sampler_t sampler = {.transform = PSAM_LOGIT_ZSCORE, .temperature = 1.5f};
psam_predict_with_sampler(model, context, len, &sampler, preds, 32);
// Produces similar diversity but with z-score normalization
```

### Backward Compatibility

The old `psam_predict()` API **still works** and uses LEGACY behavior internally to preserve exact outputs. To opt-in to the new zscore behavior, use `psam_predict_with_sampler()`.

---

## Best Practices

### 1. Start with Defaults

```c
// NULL sampler uses defaults: zscore, temp=1.0, top_p=0.95
psam_predict_with_sampler(model, context, len, NULL, preds, 32);
```

### 2. Tune Temperature First

Temperature has the biggest impact. Start with 1.0 and adjust:
- Too deterministic? Increase to 1.2-1.5
- Too random? Decrease to 0.6-0.8

### 3. Use Top-P for Safety

Top-p (nucleus sampling) prevents extreme outliers:

```c
sampler.top_p = 0.9f;  // Keeps cumulative probability mass at 90%
```

### 4. Use Seeds for Reproducibility

```c
sampler.seed = 42;  // Same seed = same output (given same context)
```

### 5. Monitor Calibrated Probabilities

The `calibrated_prob` field gives you post-softmax probabilities:

```c
if (preds[0].calibrated_prob > 0.95f) {
    // Model is very confident, might want to use this prediction
}
```

### 6. Composites: Use Defaults from .psamc

```c
// If sampler=NULL, uses defaults loaded from .psamc file
psam_composite_predict_with_sampler(composite, ctx, len, NULL, preds, 32);
```

This allows you to tune sampler settings once during composition and have them persist.

---

## Advanced: Per-Layer Bias in Composites

When using composites, you can adjust per-layer bias to shift rank distributions:

```c
// Boost "code" layer by adding +5.0 bias
psam_composite_update_layer_bias(composite, "code", 5.0f);

// Layer score computation: weight * score + bias
// Bias shifts all predictions from that layer uniformly
```

This is useful for:
- **Domain adaptation**: Boost domain-specific layers
- **Calibration**: Correct for systematic under/over-prediction
- **Mixing control**: Fine-tune layer contributions beyond just weight

**Combined with z-score normalization**, bias gives you linear "faders" (weight) and trim (bias) per layer.

---

## Examples

### Example 1: Deterministic Completion

```bash
psam generate --model hamlet.psam --prompt "To be or not to be" \
  --vocab hamlet.tsv --count 20 --temperature 0.3 --seed 42
```

Output: `that is the question: Whether 'tis nobler in the mind to suffer`

### Example 2: Creative Variation

```bash
psam generate --model hamlet.psam --prompt "To be or not to be" \
  --vocab hamlet.tsv --count 20 --temperature 1.8 --seed 42
```

Output varies significantly between runs, more surprising word choices.

### Example 3: Composite with Custom Sampler

```c
psam_composite_t* comp = psam_composite_load_file("shakespeare.psamc", false);

psam_sampler_t sampler = {
    .transform = PSAM_LOGIT_ZSCORE,
    .temperature = 0.9f,
    .top_p = 0.92f,
    .seed = time(NULL)
};

psam_composite_predict_with_sampler(comp, context, len, &sampler, preds, 32);
```

---

## Troubleshooting

### "Temperature doesn't seem to affect output"

**Check your transform mode**. If you're using `PSAM_LOGIT_RAW` or `PSAM_LOGIT_LEGACY`, you need much higher temperatures (10-100). Switch to `PSAM_LOGIT_ZSCORE`.

### "Output is too deterministic even at temp=2.0"

The model's predictions might be extremely confident. Try:
1. Check `calibrated_prob` values - are they >99%?
2. Lower `top_p` to allow more candidates: `top_p = 0.8f`
3. Increase `top_k`: `top_k = 100`
4. Consider if the context naturally has only one likely continuation

### "I want exact pre-1.1 behavior"

Use `PSAM_LOGIT_LEGACY` transform mode with the same temperature you used before (10-100 range).

---

## Technical Details

### Z-Score Computation

For each prediction step:

```python
scores = [pred.score for pred in predictions]
mean = sum(scores) / len(scores)
std = sqrt(sum((s - mean)**2 for s in scores) / len(scores))
std = max(std, 1e-6)  # avoid division by zero

for i in range(len(predictions)):
    z_scores[i] = (scores[i] - mean) / std
    logits[i] = z_scores[i] / temperature

probs = softmax(logits)
```

### Numerical Stability

Both RAW and ZSCORE modes apply max logit subtraction in softmax for numerical stability:

```python
def softmax(logits):
    max_logit = max(logits)
    exp_logits = [exp(l - max_logit) for l in logits]
    sum_exp = sum(exp_logits)
    return [e / sum_exp for e in exp_logits]
```

This prevents overflow without changing the output distribution.

---

## See Also

- [API Reference](API.md) - Complete API documentation
- [Composition Guide](Composition.md) - Multi-model composition
- [CLI Reference](CLI.md) - Command-line usage

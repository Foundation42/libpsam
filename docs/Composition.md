## **Core Composition API**

### **1. Layered Model Builder**
```c
// C API
psam_composite_t* psam_create_layered(psam_model_t* base);
psam_error_t psam_composite_add_layer(psam_composite_t* composite, const char* name, psam_model_t* layer, float weight);
psam_error_t psam_composite_remove_layer(psam_composite_t* composite, const char* name);
psam_error_t psam_composite_set_base_weight(psam_composite_t* composite, float weight);
int psam_composite_predict(psam_composite_t* composite, const uint32_t* ctx, size_t len, psam_prediction_t* out, size_t max_preds);
```

```javascript
// JS API - Fluent builder pattern
const composite = PSAM.layered()
  .base(generalModel)
  .layer('medical', medicalModel, 1.5)
  .layer('empathy', empathyModel, 1.2)
  .layer('formal', formalModel, 0.8)
  .build();
```

### **2. Sequenced Model Builder**
```c
// C API
psam_sequence_t* psam_create_sequence();
void psam_add_sequence_stage(psam_sequence_t* seq, psam_model_t* model, int token_length, const char* description);
```

```javascript  
// JS API - Temporal sequencing
const storyWriter = PSAM.sequenced()
  .stage('hook', openingModel, 50)      // First 50 tokens
  .stage('build', narrativeModel, 200)  // Next 200 tokens
  .stage('climax', dramaticModel, 100)  // Then 100 tokens
  .stage('resolve', endingModel)        // Until completion
  .build();
```

### **3. Routed Model Builder**
```c
// C API - with custom routing function
psam_routed_t* psam_create_routed(psam_router_func_t router);
```

```javascript
// JS API - Conditional routing
const smartRouter = PSAM.routed()
  .condition(context => context.hasCode(), codeModel, 2.0)
  .condition(context => context.isMedical(), medicalModel, 1.8)
  .condition(context => context.isCasual(), casualModel, 1.2)
  .default(generalModel, 1.0)
  .build();
```

### **4. Ensemble Model Builder**
```javascript
// JS API - Voting ensemble
const expertPanel = PSAM.ensemble()
  .expert('grammar', grammarModel, 1.5)
  .expert('style', styleModel, 1.2) 
  .expert('creativity', creativeModel, 0.9)
  .strategy('weighted_average')  // or 'consensus', 'rank_fusion'
  .build();
```

## **Unified Composite Interface**

All composite models would share a common interface:

```c
// Unified prediction across all topologies
int psam_composite_predict(psam_composite_t* comp, 
                          uint32_t* context, int context_len,
                          psam_prediction_t* predictions, int max_preds);
```

```javascript
// Consistent JS API regardless of topology
const predictions = compositeModel.predict(context, maxPredictions);
// predictions now include scores, rawStrengths, supportCounts, probabilities arrays
const saved = compositeModel.save('my-composition.psamc');  // Composite format
```

## **Composite Model File Format**

New file extension `.psamc` for composite models:

```
.psamc file structure:
[Header]
magic: "PSAMCOMP"
version: 1
topology_type: layered|sequenced|routed|ensemble

[Components]
- base_model: ref or embedded
- layers: [{name, model_ref, weight, activation_conditions?}]
- routing_logic? (for routed types)
- sequence_stages? (for sequenced types)
- topology metadata is encoded in the `PSAMC_SECTION_LAYER` table so weights/IDs survive save/load

[Model References]
- Either embedded .psam models
- Or URLs/paths to external models

## **Aligned Composite Workflow**

Week 3 introduced aligned composites: a layered blend that shares a unified vocabulary even when
each source model was trained on a different local vocab. The runtime now exposes:

- `psam_build_vocab_alignment_from_files()` to create a unified lexicon plus bidirectional maps.
- `psam_create_composite_aligned()` / `psam_composite_aligned_predict_with_sampler()` for
  inference through the unified space.
- `psam_composite_save_v1()` / `psam_composite_load_aligned()` to persist the alignment alongside
  the composite manifest, including SHA-256 checks for the map binaries and the unified vocab file.

CLI highlights:

- `psam compose --from-vocabs ...` emits `.psamc` artifacts with an `alignment` block that stores
  relative paths for `*.l2u.u32` / `*.u2l.pairs` remap files and the unified `*.tsv`.
- `psam predict --model aligned.psamc --prompt "..."` no longer needs a `--vocab` flag; for aligned
  composites the CLI resolves the saved unified vocabulary automatically. (Raw `.psam` models still
  require `--vocab`, and you can always fall back to `--ctx-ids`.)
- `psam generate` uses the same auto-discovery, so long-form sampling tests can be driven end-to-end
  from the saved composite.

This closes the loop we sketched in the sprint notes: researchers can build a federated composite
from Shakespeare plays, programming-language corpora, or multilingual domains, save it once, and hand
the `.psamc` to a teammate who can immediately run `psam predict`/`psam generate` against it without
remembering the supporting vocab files.
```

## **Advanced Composition Features**

### **Dynamic Weight Adjustment**
```javascript
// Adjust layer weights in real-time based on performance
compositeModel.setLayerWeight('medical', 1.8);
compositeModel.autoTuneWeights(validationData);  // Learn optimal weights
```

### **Layer Activation Conditions**
```javascript
// Conditional layer activation
compositeModel.setLayerActivation('code', {
  condition: (context) => context.hasPythonSyntax(),
  warmup: 10,    // Gradually increase weight over 10 tokens
  cooldown: 5     // Gradually decrease over 5 tokens
});
```

### **Cross-Layer Attention** (Advanced)
```javascript
// Layers can influence each other
compositeModel.enableCrossLayerAttention({
  from: 'creative',
  to: 'technical',
  strength: 0.3   // Creative layer can slightly influence technical
});
```

## **Builder API Examples**

### **Complete Workflow**
```javascript
// Build a sophisticated writing assistant
const myWriter = PSAM.layered()
  .base(await PSAM.load('models/general-medium.psam'))
  .layer('creative', await PSAM.load('models/creative-fiction.psam'), 1.6)
  .layer('technical', await PSAM.load('models/technical-writing.psam'), 1.2)
  .layer('dialog', await PSAM.load('models/conversational.psam'), 0.9)
  .withFIM(await PSAM.load('models/fim-creative.psam'))  // FIM capability
  .build();

// Save the entire composition
await myWriter.save('my-writer-composition.psamc');

// Load and use seamlessly
const loadedWriter = await PSAM.load('my-writer-composition.psamc');
```

### **Music Composition Pipeline**
```javascript
const musicProducer = PSAM.sequenced()
  .stage('intro', await PSAM.load('models/midi-intro-patterns.psam'), 32)
  .stage('verse', await PSAM.load('models/midi-verse-melodies.psam'), 64) 
  .stage('chorus', await PSAM.load('models/midi-chorus-hooks.psam'), 32)
  .stage('bridge', await PSAM.load('models/midi-bridge-transitions.psam'), 16)
  .stage('outro', await PSAM.load('models/midi-outro-endings.psam'))
  .build();
```

## **Implementation Strategy**

### **Phase 1: Core Composition API**
- `LayeredBuilder` - basic weighted layer mixing
- Composite model save/load format
- Unified prediction interface
- Regression target: pull a few Shakespeare plays from `corpora/text/` to sanity-check that layered blends actually shift lexical focus (e.g., tragedies vs comedies).
- A reproducible harness now lives in `scripts/shakespeare_harness.py`; it trains tragedy/comedy overlays that share the `tiny_shakespeare` vocabulary, emits `.psamc` artifacts, and prints sample predictions for a prompt so we can spot regression drift quickly.
- CLI `psam compose` already emits balanced `.psamc` manifests (first `--layer` is base, subsequent entries are overlays with weight 1.0 by default); bindings can call `psam_composite_save_file` and `psam_composite_load_file` directly.

### **Phase 2: Advanced Topologies** 
- `SequencedBuilder` - temporal staging
- `RoutedBuilder` - conditional routing
- Performance optimizations for composites

### **Phase 3: Dynamic Features**
- Real-time weight adjustment
- Layer activation conditions
- Cross-layer interactions

## **The Beauty of This Approach**

1. **Progressive Disclosure** - Start simple, add complexity as needed
2. **Consistent Mental Model** - All topologies use the same prediction interface  
3. **Serializable Compositions** - Save/load complex arrangements
4. **Performance Maintained** - Composites can be optimized under the hood
5. **Extensible** - New topology types can be added later

This would make libpsam the "Docker Compose" of AI models - with declarative, reproducible, and easily shareable model compositions.

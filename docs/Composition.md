## **Core Composition API**

### **1. Layered Model Builder**
```c
// C API
psam_composite_t* psam_create_layered(psam_model_t* base);
void psam_add_layer(psam_composite_t* composite, const char* name, psam_model_t* layer, float weight);
void psam_remove_layer(psam_composite_t* composite, const char* name);
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

[Model References]
- Either embedded .psam models
- Or URLs/paths to external models
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

import { useState, useEffect } from 'react';
import { Play, Zap, Info, RefreshCw, Settings } from 'lucide-react';

// Import WASM bindings (loaded via script tag in index.html)
interface PSAMInstance {
  trainBatch(tokens: number[]): void;
  finalizeTraining(): void;
  predict(context: number[], maxPredictions?: number): { ids: number[]; scores: Float32Array };
  sample(context: number[], temperature?: number): number;
  stats(): {
    vocabSize: number;
    rowCount: number;
    edgeCount: number;
    totalTokens: number;
    memoryBytes: number;
  };
  destroy(): void;
}

const testScenarios = [
  { name: "Simple Pattern", text: "the cat sat on the mat. the dog sat on the rug." },
  { name: "Repetition", text: "a b c d. a b c e. a b c f. a b c g." },
  { name: "Sequences", text: "one two three four. five six seven eight. nine ten eleven twelve." },
  { name: "Story", text: "once upon a time in a small village, there lived a curious young girl named luna. luna loved to explore the forest near her home. one sunny morning, luna decided to venture deeper into the woods than ever before. she discovered a hidden clearing where magical butterflies danced in the golden sunlight. the butterflies led her to an ancient oak tree with a door carved into its trunk. luna opened the door and found a library filled with books that whispered secrets of the forest. she spent hours reading about the creatures and plants that called the forest home. as the sun began to set, the butterflies guided luna back to the village. from that day on, luna visited the magical library every week, learning more about the wonders of nature. with each visit, the library revealed new secrets. luna learned the language of birds and how to read the patterns in tree bark. the ancient books taught her about healing herbs and the stories written in the stars. one autumn evening, the butterflies brought luna a special gift, a silver key that unlocked a hidden chamber deep within the oak tree. inside the chamber, luna found a crystal that glowed with soft blue light. the crystal showed her visions of the forest's past and glimpses of its future. luna realized she had become the forest's keeper, entrusted with protecting its magic for generations to come." },
];

const PSAMWasmDemo = () => {
  const [psam, setPsam] = useState<PSAMInstance | null>(null);
  const [vocab, setVocab] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Training
  const [text, setText] = useState("the cat sat on the mat. the dog sat on the rug. the bird sat on the branch. the frog sat on the log.");
  const [trained, setTrained] = useState(false);

  // Inference
  const [inferenceInput, setInferenceInput] = useState("luna loved to explore the");
  const [predictions, setPredictions] = useState<{ token: number; word: string; score: number; probability: number }[]>([]);

  // Live prediction updates
  useEffect(() => {
    if (!psam || !trained) {
      setPredictions([]);
      return;
    }

    try {
      const contextWords = inferenceInput.toLowerCase().match(/\w+|[.,!?;]/g) || [];
      const contextTokens = contextWords.map(w => vocab.indexOf(w)).filter(t => t >= 0);

      if (contextTokens.length === 0) {
        setPredictions([]);
        return;
      }

      const result = psam.predict(contextTokens.slice(-contextWindow), topK);

      // Calculate probabilities with temperature
      const logits = Array.from(result.scores).map(s => s / temperature);
      const maxLogit = Math.max(...logits);
      const expScores = logits.map(l => Math.exp(l - maxLogit));
      const sumExp = expScores.reduce((a, b) => a + b, 0);
      const probs = expScores.map(e => e / sumExp);

      const preds = result.ids.map((id, i) => ({
        token: id,
        word: vocab[id] || `<${id}>`,
        score: result.scores[i],
        probability: probs[i]
      }));

      setPredictions(preds);
    } catch (err) {
      // Silent fail for live updates
      setPredictions([]);
    }
  }, [inferenceInput, psam, trained, vocab, contextWindow, topK, temperature]);

  // Auto-generation
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationHistory, setGenerationHistory] = useState<{
    input: string;
    predicted: string;
    score: number;
    probability: number;
    confidence: number;
    alternatives: { word: string; probability: number }[];
  }[]>([]);

  // Parameters
  const [contextWindow, setContextWindow] = useState(8);
  const [topK, setTopK] = useState(32);
  const [minEvidence, setMinEvidence] = useState(1);
  const [alpha, setAlpha] = useState(0.1); // distance decay
  const [enableIdf, setEnableIdf] = useState(true);
  const [enablePpmi, setEnablePpmi] = useState(true);
  const [edgeDropout, setEdgeDropout] = useState(0.0);
  const [temperature, setTemperature] = useState(1.0);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStep, setTrainingStep] = useState(0);
  const [tokens, setTokens] = useState<number[]>([]);

  // Stats
  const [stats, setStats] = useState<any>(null);

  // Load WASM module on mount
  useEffect(() => {
    const loadWasm = async () => {
      try {
        setLoading(true);
        setError(null);

        // Wait for createPSAMModule to be available
        let retries = 0;
        while (!(window as any).createPSAMModule && retries < 50) {
          await new Promise(resolve => setTimeout(resolve, 100));
          retries++;
        }

        if (!(window as any).createPSAMModule) {
          throw new Error('WASM module not loaded. Make sure psam.js is included in index.html');
        }

        // Initialize the WASM module
        const Module = await (window as any).createPSAMModule({
          locateFile: (path: string) => {
            if (path.endsWith('.wasm')) {
              return './wasm/' + path;
            }
            return path;
          }
        });

        // Create wrapper for PSAM instance with full config
        const create = (config: {
          vocabSize: number;
          window: number;
          topK: number;
          alpha: number;
          minEvidence: number;
          enableIdf: boolean;
          enablePpmi: boolean;
          edgeDropout: number;
        }): PSAMInstance => {
          // Allocate config struct (32 bytes)
          const configPtr = Module._malloc(32);
          const view32 = new Uint32Array(Module.HEAPU32.buffer, configPtr, 8);
          const viewF32 = new Float32Array(Module.HEAPF32.buffer, configPtr, 8);
          const view8 = new Uint8Array(Module.HEAPU8.buffer, configPtr, 32);

          // Fill config struct (matching C struct layout with padding)
          view32[0] = config.vocabSize;        // offset 0: vocab_size (uint32_t)
          view32[1] = config.window;           // offset 4: window (uint32_t)
          view32[2] = config.topK;             // offset 8: top_k (uint32_t)
          viewF32[3] = config.alpha;           // offset 12: alpha (float)
          viewF32[4] = config.minEvidence;     // offset 16: min_evidence (float)
          view8[20] = config.enableIdf ? 1 : 0;   // offset 20: enable_idf (bool)
          view8[21] = config.enablePpmi ? 1 : 0;  // offset 21: enable_ppmi (bool)
          // offset 22-23: padding
          viewF32[6] = config.edgeDropout;     // offset 24: edge_dropout (float)

          const psam_create_with_config = Module.cwrap('psam_create_with_config', 'number', ['number']);
          const handle = psam_create_with_config(configPtr);

          Module._free(configPtr);

          if (!handle) {
            throw new Error('Failed to create PSAM model');
          }

          return {
            trainBatch: (tokens: number[]) => {
              const tokensArray = new Uint32Array(tokens);
              const tokensPtr = Module._malloc(tokensArray.length * 4);
              Module.HEAPU32.set(tokensArray, tokensPtr / 4);

              const psam_train_batch = Module.cwrap('psam_train_batch', 'number', ['number', 'number', 'number']);
              psam_train_batch(handle, tokensPtr, tokensArray.length);

              Module._free(tokensPtr);
            },

            finalizeTraining: () => {
              const psam_finalize = Module.cwrap('psam_finalize_training', 'number', ['number']);
              psam_finalize(handle);
            },

            predict: (context: number[], maxPredictions = 10) => {
              const contextArray = new Uint32Array(context);
              const contextPtr = Module._malloc(contextArray.length * 4);
              Module.HEAPU32.set(contextArray, contextPtr / 4);

              const predsPtr = Module._malloc(maxPredictions * 12);

              const psam_predict = Module.cwrap('psam_predict', 'number', ['number', 'number', 'number', 'number', 'number']);
              const numPreds = psam_predict(handle, contextPtr, contextArray.length, predsPtr, maxPredictions);

              const ids: number[] = [];
              const scores: number[] = [];

              if (numPreds > 0) {
                for (let i = 0; i < numPreds; i++) {
                  const offset = predsPtr / 4 + i * 3;
                  ids.push(Module.HEAPU32[offset]);
                  scores.push(Module.HEAPF32[offset + 1]);
                }
              }

              Module._free(contextPtr);
              Module._free(predsPtr);

              return { ids, scores: new Float32Array(scores) };
            },

            sample: (context: number[], temperature = 1.0) => {
              const result = this.predict(context, 10);
              if (result.ids.length === 0) return 0;

              // Apply temperature and softmax
              const logits = Array.from(result.scores).map(s => s / temperature);
              const maxLogit = Math.max(...logits);
              const expScores = logits.map(l => Math.exp(l - maxLogit));
              const sumExp = expScores.reduce((a, b) => a + b, 0);
              const probs = expScores.map(e => e / sumExp);

              // Sample from distribution
              const rand = Math.random();
              let cumsum = 0;
              for (let i = 0; i < probs.length; i++) {
                cumsum += probs[i];
                if (rand < cumsum) {
                  return result.ids[i];
                }
              }

              return result.ids[0];
            },

            stats: () => {
              const statsPtr = Module._malloc(32);
              const psam_get_stats = Module.cwrap('psam_get_stats', 'number', ['number', 'number']);
              psam_get_stats(handle, statsPtr);

              const view32 = new Uint32Array(Module.HEAPU32.buffer, statsPtr, 8);

              const stats = {
                vocabSize: view32[0],
                rowCount: view32[1],
                edgeCount: Number(BigInt(view32[2]) | (BigInt(view32[3]) << 32n)),
                totalTokens: Number(BigInt(view32[4]) | (BigInt(view32[5]) << 32n)),
                memoryBytes: Number(BigInt(view32[6]) | (BigInt(view32[7]) << 32n)),
              };

              Module._free(statsPtr);
              return stats;
            },

            destroy: () => {
              const psam_destroy = Module.cwrap('psam_destroy', null, ['number']);
              psam_destroy(handle);
            }
          };
        };

        // Store the create function and Module for later use
        (window as any).__psamCreate = create;
        (window as any).__psamModule = Module;
        setLoading(false);
      } catch (err) {
        console.error('Failed to load WASM:', err);
        setError(err instanceof Error ? err.message : 'Failed to load WASM');
        setLoading(false);
      }
    };

    loadWasm();

    return () => {
      if (psam) {
        psam.destroy();
      }
    };
  }, []);

  // Effect for auto-training
  useEffect(() => {
    if (isTraining && trainingStep < tokens.length - 1) {
      const timer = setTimeout(() => {
        setTrainingStep(s => s + 1);
      }, 50);
      return () => clearTimeout(timer);
    } else if (isTraining && trainingStep >= tokens.length - 1) {
      setIsTraining(false);
    }
  }, [isTraining, trainingStep, tokens.length]);

  const tokenize = (text: string): { tokens: number[]; vocab: string[] } => {
    // Match JS version tokenization: words and punctuation
    const words = text.toLowerCase().match(/\w+|[.,!?;]/g) || [];
    const uniqueWords = [...new Set(words)];
    const tokens = words.map(w => uniqueWords.indexOf(w));
    return { tokens, vocab: uniqueWords };
  };

  const handleTrain = () => {
    const createFn = (window as any).__psamCreate;
    if (!createFn) return;

    try {
      // Destroy old model if exists
      if (psam) {
        psam.destroy();
        setPsam(null);
      }

      // Tokenize text
      const { tokens: newTokens, vocab: newVocab } = tokenize(text);
      setVocab(newVocab);
      setTokens(newTokens);
      setTrainingStep(0);

      // Create new model with current parameters
      const instance = createFn({
        vocabSize: newVocab.length,
        window: contextWindow,
        topK: topK,
        alpha: alpha,
        minEvidence: minEvidence,
        enableIdf: enableIdf,
        enablePpmi: enablePpmi,
        edgeDropout: edgeDropout,
      });

      // Train
      instance.trainBatch(newTokens);
      instance.finalizeTraining();

      const modelStats = instance.stats();
      setStats(modelStats);
      setPsam(instance);
      setTrained(true);
      setPredictions([]);
      setGenerationHistory([]);
      setTrainingStep(newTokens.length - 1);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Training failed');
    }
  };


  const handleGenerate = () => {
    if (!psam || !trained || isGenerating) return;

    setIsGenerating(true);
    const contextWords = inferenceInput.toLowerCase().match(/\w+|[.,!?;]/g) || [];
    const history: typeof generationHistory = [];

    let currentTokens = contextWords.map(w => vocab.indexOf(w)).filter(t => t >= 0);
    let currentInput = inferenceInput;

    for (let i = 0; i < 10; i++) {
      const result = psam.predict(currentTokens.slice(-contextWindow), topK);

      if (result.ids.length === 0) break;

      // Calculate probabilities with temperature
      const logits = Array.from(result.scores).map(s => s / temperature);
      const maxLogit = Math.max(...logits);
      const expScores = logits.map(l => Math.exp(l - maxLogit));
      const sumExp = expScores.reduce((a, b) => a + b, 0);
      const probs = expScores.map(e => e / sumExp);

      // Sample from distribution
      const rand = Math.random();
      let cumsum = 0;
      let selectedIdx = 0;

      for (let j = 0; j < probs.length; j++) {
        cumsum += probs[j];
        if (rand < cumsum) {
          selectedIdx = j;
          break;
        }
      }

      const selectedToken = result.ids[selectedIdx];
      const word = vocab[selectedToken] || `<${selectedToken}>`;

      const confidence = result.ids.length >= 2 ? result.scores[0] / result.scores[1] : Infinity;

      history.push({
        input: currentInput,
        predicted: word,
        score: result.scores[selectedIdx],
        probability: probs[selectedIdx],
        confidence,
        alternatives: result.ids.slice(1, 3).map((id, i) => ({
          word: vocab[id] || `<${id}>`,
          probability: probs[i + 1]
        }))
      });

      currentInput = currentInput + ' ' + word;
      currentTokens.push(selectedToken);
    }

    setGenerationHistory(history);
    setInferenceInput(currentInput);
    setIsGenerating(false);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading WASM module...</p>
        </div>
      </div>
    );
  }

  if (error && !psam) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 max-w-md">
          <h2 className="text-red-800 font-bold mb-2">Error Loading WASM</h2>
          <p className="text-red-600">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-8 mb-6">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-3xl font-bold text-gray-800">PSAM WASM Demo</h1>
            <div className="flex items-center gap-2 bg-green-100 px-3 py-1 rounded-full">
              <Zap className="w-4 h-4 text-green-600" />
              <span className="text-sm font-medium text-green-800">WebAssembly Powered</span>
            </div>
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
            <div className="flex items-start gap-2">
              <Info className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
              <div className="text-sm text-blue-800">
                <p className="font-medium mb-1">Real C Library via WebAssembly</p>
                <p>This demo uses the actual libpsam C library compiled to WASM. It's <strong>20-200× faster</strong> than pure JavaScript!</p>
              </div>
            </div>
          </div>

          {/* Quick Test Scenarios */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">Quick Tests</label>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {testScenarios.map((scenario, idx) => (
                <button
                  key={idx}
                  onClick={() => setText(scenario.text)}
                  className="px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors text-gray-700"
                >
                  {scenario.name}
                </button>
              ))}
            </div>
          </div>

          {/* Training Text */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Training Text
            </label>
            <textarea
              value={text}
              onChange={(e) => {
                setText(e.target.value);
                setTrained(false);
                setTrainingStep(0);
              }}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent font-mono text-sm"
              rows={4}
              placeholder="Enter text to train on..."
            />
          </div>

          {/* Parameters */}
          <div className="mb-4">
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-800 mb-2"
            >
              <Settings className="w-4 h-4" />
              {showAdvanced ? 'Hide' : 'Show'} Parameters
            </button>

            {showAdvanced && (
              <div className="p-4 bg-gray-50 rounded-lg space-y-3">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">Context Window</label>
                    <input
                      type="number"
                      value={contextWindow}
                      onChange={(e) => setContextWindow(Math.max(1, parseInt(e.target.value) || 1))}
                      className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                      min="1"
                      max="20"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">Top-K</label>
                    <input
                      type="number"
                      value={topK}
                      onChange={(e) => setTopK(Math.max(1, parseInt(e.target.value) || 1))}
                      className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                      min="1"
                      max="128"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">Min Evidence</label>
                    <input
                      type="number"
                      value={minEvidence}
                      onChange={(e) => setMinEvidence(Math.max(1, parseInt(e.target.value) || 1))}
                      className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                      min="1"
                      max="5"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">Distance Decay (α)</label>
                    <input
                      type="number"
                      value={alpha}
                      onChange={(e) => setAlpha(parseFloat(e.target.value))}
                      className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                      step="0.05"
                      min="0"
                      max="1"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">Edge Dropout</label>
                    <input
                      type="number"
                      value={edgeDropout}
                      onChange={(e) => setEdgeDropout(parseFloat(e.target.value))}
                      className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                      step="0.05"
                      min="0"
                      max="0.5"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">Temperature</label>
                    <input
                      type="number"
                      value={temperature}
                      onChange={(e) => setTemperature(Math.max(0.1, parseFloat(e.target.value)))}
                      className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                      step="0.1"
                      min="0.1"
                      max="2"
                    />
                  </div>

                  <div className="flex flex-col gap-2">
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={enableIdf}
                        onChange={(e) => setEnableIdf(e.target.checked)}
                        className="rounded"
                      />
                      <label className="text-xs font-medium text-gray-700">Enable IDF</label>
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={enablePpmi}
                        onChange={(e) => setEnablePpmi(e.target.checked)}
                        className="rounded"
                      />
                      <label className="text-xs font-medium text-gray-700">Enable PPMI</label>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Train Button */}
          <button
            onClick={handleTrain}
            disabled={loading}
            className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-300 text-white font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2 mb-4"
          >
            <Play className="w-5 h-5" />
            Train Model
          </button>

          {/* Training Progress Visualization */}
          {tokens.length > 0 && (
            <div className="mb-6">
              <div className="text-sm text-gray-600 mb-2">
                Training Progress: {trainingStep} / {tokens.length - 1}
              </div>
              <div className="font-mono text-sm p-3 bg-gray-50 rounded border border-gray-200 overflow-x-auto">
                {tokens.map((_, idx) => {
                  const word = vocab[tokens[idx]] || '';
                  return (
                    <span
                      key={idx}
                      className={`mr-1 px-1 rounded ${
                        idx === trainingStep
                          ? 'bg-indigo-600 text-white'
                          : idx < trainingStep
                          ? 'bg-green-200'
                          : 'bg-gray-200'
                      }`}
                    >
                      {word}
                    </span>
                  );
                })}
              </div>
            </div>
          )}

          {/* Stats */}
          {stats && (
            <div className="bg-gray-50 rounded-lg p-4 mb-6">
              <h3 className="font-medium text-gray-800 mb-3">📊 Model Statistics</h3>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
                <div>
                  <div className="text-gray-500">Vocabulary</div>
                  <div className="font-bold text-gray-800">{stats.vocabSize}</div>
                </div>
                <div>
                  <div className="text-gray-500">Rows</div>
                  <div className="font-bold text-gray-800">{stats.rowCount}</div>
                </div>
                <div>
                  <div className="text-gray-500">Edges</div>
                  <div className="font-bold text-gray-800">{stats.edgeCount}</div>
                </div>
                <div>
                  <div className="text-gray-500">Tokens</div>
                  <div className="font-bold text-gray-800">{stats.totalTokens}</div>
                </div>
                <div>
                  <div className="text-gray-500">Memory</div>
                  <div className="font-bold text-gray-800">{(stats.memoryBytes / 1024).toFixed(1)} KB</div>
                </div>
              </div>
            </div>
          )}

          {/* Inference Section */}
          {trained && (
            <div className="border-t pt-6">
              <h3 className="text-xl font-semibold mb-4 text-gray-800">🔮 Inference</h3>

              {/* Context Input */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Context (predictions update as you type)
                </label>
                <input
                  type="text"
                  value={inferenceInput}
                  onChange={(e) => setInferenceInput(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent font-mono text-sm"
                  placeholder="Enter context..."
                />
              </div>

              {/* Action Buttons */}
              <div className="flex gap-3 mb-6">
                <button
                  onClick={() => {
                    if (!psam || !trained || predictions.length === 0) return;
                    const topPrediction = predictions[0];
                    setInferenceInput(inferenceInput + ' ' + topPrediction.word);
                  }}
                  disabled={!trained || predictions.length === 0}
                  className="bg-green-600 hover:bg-green-700 disabled:bg-gray-300 text-white font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                  <Zap className="w-5 h-5" />
                  Generate Next
                </button>
                <button
                  onClick={handleGenerate}
                  disabled={!trained || isGenerating}
                  className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-300 text-white font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                  <RefreshCw className={`w-5 h-5 ${isGenerating ? 'animate-spin' : ''}`} />
                  Auto-Generate (10 tokens)
                </button>
              </div>

              {/* Predictions */}
              {predictions.length > 0 && (
                <div className="bg-gray-50 rounded-lg p-4 mb-4">
                  <h4 className="font-medium text-gray-800 mb-3">Top Predictions</h4>
                  <div className="space-y-2">
                    {predictions.slice(0, 5).map((pred, i) => (
                      <div key={i} className="flex items-center gap-3">
                        <div className="w-32 text-right font-mono text-xs">
                          <div>{pred.score.toFixed(3)}</div>
                          <div className="text-gray-500">({(pred.probability * 100).toFixed(1)}%)</div>
                        </div>
                        <div className="flex-1 bg-gray-200 rounded-full h-8 overflow-hidden">
                          <div
                            className="bg-indigo-600 h-full flex items-center px-3 text-white text-sm font-semibold"
                            style={{ width: `${pred.probability * 100}%` }}
                          >
                            "{pred.word}"
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Generation History */}
              {generationHistory.length > 0 && (
                <div className="bg-white rounded-lg border border-gray-200 p-4">
                  <h4 className="font-medium text-gray-800 mb-3">Generation History</h4>
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {generationHistory.map((item, idx) => (
                      <div key={idx} className="p-2 bg-gray-50 rounded border border-gray-200 text-sm">
                        <div className="font-mono mb-1">
                          <span className="text-gray-600">{item.input}</span>
                          <span className="text-green-600 font-bold"> {item.predicted}</span>
                        </div>
                        <div className="text-xs text-gray-500">
                          Score: {item.score.toFixed(3)} |
                          Prob: {(item.probability * 100).toFixed(1)}% |
                          Conf: {item.confidence.toFixed(2)}x
                          {item.alternatives.length > 0 && (
                            <span className="ml-2">
                              | Alt: {item.alternatives.map((a) => `"${a.word}" (${(a.probability * 100).toFixed(0)}%)`).join(', ')}
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PSAMWasmDemo;

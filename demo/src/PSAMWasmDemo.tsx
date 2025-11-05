import { useState, useEffect } from 'react';
import { Play, Zap, Info, RefreshCw, Settings } from 'lucide-react';

declare global {
  interface Window {
    createPSAMModule?: any;
    PSAM_WASM_PATH?: string;
  }
}

function normalizePath(path: string): string {
  return path.endsWith('/') ? path.slice(0, -1) : path;
}

function resolveWasmDir(): string {
  if (typeof window !== 'undefined' && window.PSAM_WASM_PATH) {
    return normalizePath(window.PSAM_WASM_PATH);
  }
  const base = (import.meta as any).env?.BASE_URL ?? '/';
  return normalizePath(base) + '/wasm';
}

async function ensureWasmModuleLoaded(): Promise<void> {
  if (typeof window === 'undefined' || window.createPSAMModule) {
    return;
  }

  const src = `${resolveWasmDir()}/psam.js`;

  await new Promise<void>((resolve, reject) => {
    const existing = document.querySelector(`script[data-psam-wasm="${src}"]`) as HTMLScriptElement | null;
    if (existing) {
      if (window.createPSAMModule) {
        resolve();
        return;
      }
      existing.addEventListener('load', () => resolve(), { once: true });
      existing.addEventListener('error', () => reject(new Error(`Failed to load ${src}`)), { once: true });
      return;
    }

    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.dataset.psamWasm = src;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(script);
  });
}

// Import WASM bindings (loaded via script tag in index.html)
interface PSAMInstance {
  trainBatch(tokens: number[]): void;
  finalizeTraining(): void;
  predict(
    context: number[],
    maxPredictions?: number,
    temperature?: number
  ): {
    ids: number[];
    scores: Float32Array;
    rawStrengths: Float32Array;
    supportCounts: Uint16Array;
    probabilities: Float32Array;
  };
  explain?(context: number[], candidateToken: number, maxTerms?: number): {
    candidate: number;
    total: number;
    bias: number;
    termCount: number;
    terms: {
      source: number;
      offset: number;
      weight: number;
      idf: number;
      decay: number;
      contribution: number;
    }[];
  };
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
  { name: "Simple Pattern", text: "the cat sat on the mat. the dog sat on the rug.", suggestedInput: "the cat" },
  { name: "Repetition", text: "a b c d. a b c e. a b c f. a b c g.", suggestedInput: "a b c" },
  { name: "Sequences", text: "one two three four. five six seven eight. nine ten eleven twelve.", suggestedInput: "one two three" },
  { name: "Story", text: "once upon a time in a small village, there lived a curious young girl named luna. luna loved to explore the forest near her home. one sunny morning, luna decided to venture deeper into the woods than ever before. she discovered a hidden clearing where magical butterflies danced in the golden sunlight. the butterflies led her to an ancient oak tree with a door carved into its trunk. luna opened the door and found a library filled with books that whispered secrets of the forest. she spent hours reading about the creatures and plants that called the forest home. as the sun began to set, the butterflies guided luna back to the village. from that day on, luna visited the magical library every week, learning more about the wonders of nature. with each visit, the library revealed new secrets. luna learned the language of birds and how to read the patterns in tree bark. the ancient books taught her about healing herbs and the stories written in the stars. one autumn evening, the butterflies brought luna a special gift, a silver key that unlocked a hidden chamber deep within the oak tree. inside the chamber, luna found a crystal that glowed with soft blue light. the crystal showed her visions of the forest's past and glimpses of its future. luna realized she had become the forest's keeper, entrusted with protecting its magic for generations to come.", suggestedInput: "luna loved to explore the" },
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
  const [inferenceInput, setInferenceInput] = useState("the cat");
  const [predictions, setPredictions] = useState<{
    token: number;
    word: string;
    score: number;
    rawStrength: number;
    supportCount: number;
    probability: number;
  }[]>([]);
  const [stochasticSamples, setStochasticSamples] = useState<{
    word: string;
    count: number;
    probability: number;
  }[]>([]);
  const [explanation, setExplanation] = useState<{
    token: string;
    total: number;
    bias: number;
    termCount: number;
    terms: {
      sourceToken: number;
      sourceWord: string;
      offset: number;
      weight: number;
      idf: number;
      decay: number;
      contribution: number;
    }[];
  } | null>(null);

  // Auto-generation
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationHistory, setGenerationHistory] = useState<{
    input: string;
    predicted: string;
    score: number;
    rawStrength: number;
    supportCount: number;
    probability: number;
    confidence: number;
    alternatives: { word: string; probability: number; rawStrength: number; supportCount: number }[];
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
  const [samplingMode, setSamplingMode] = useState<'greedy' | 'stochastic'>('greedy');
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStep, setTrainingStep] = useState(0);
  const [tokens, setTokens] = useState<number[]>([]);

  // Stats
  const [stats, setStats] = useState<any>(null);

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

      const result = psam.predict(contextTokens.slice(-contextWindow), topK, temperature);

      // Use calibrated probabilities from sampler
      const preds = result.ids.map((id, i) => ({
        token: id,
        word: vocab[id] || `<${id}>`,
        score: result.scores[i] ?? 0,
        rawStrength: result.rawStrengths[i] ?? 0,
        supportCount: result.supportCounts[i] ?? 0,
        probability: result.probabilities[i] ?? 0,
      }));

      setPredictions(preds);

      // Generate stochastic samples
      if (result.ids.length > 0) {
        const samples = generateStochasticSamples(
          result.ids,
          Array.from(result.probabilities),
          100
        );
        setStochasticSamples(samples);
      }

      // Generate explanation for top prediction
      if (preds.length > 0 && psam.explain) {
        try {
          const topToken = preds[0].token;
          const explainResult = psam.explain(contextTokens.slice(-contextWindow), topToken, 10);

          setExplanation({
            token: preds[0].word,
            total: explainResult.total,
            bias: explainResult.bias,
            termCount: explainResult.termCount,
            terms: explainResult.terms.map(term => ({
              sourceToken: term.source,
              sourceWord: vocab[term.source] || `<${term.source}>`,
              offset: term.offset,
              weight: term.weight,
              idf: term.idf,
              decay: term.decay,
              contribution: term.contribution,
            }))
          });
        } catch (err) {
          setExplanation(null);
        }
      } else {
        setExplanation(null);
      }
    } catch (err) {
      // Silent fail for live updates
      setPredictions([]);
      setExplanation(null);
    }
  }, [inferenceInput, psam, trained, vocab, contextWindow, topK, temperature]);

  // Load WASM module on mount
  useEffect(() => {
    const loadWasm = async () => {
      try {
        setLoading(true);
        setError(null);

        await ensureWasmModuleLoaded();

        if (!(window as any).createPSAMModule) {
          throw new Error('PSAM WASM module script failed to load');
        }

        const Module = await (window as any).createPSAMModule({
          locateFile: (path: string) => {
            const baseDir = resolveWasmDir();
            return path.endsWith('.wasm') ? `${baseDir}/${path}` : path;
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

            predict: (context: number[], maxPredictions = 10, temp = 1.0) => {
              const contextArray = new Uint32Array(context);
              const contextPtr = Module._malloc(contextArray.length * 4);
              Module.HEAPU32.set(contextArray, contextPtr / 4);
              const PREDICTION_SIZE = 20;
              const predsPtr = Module._malloc(maxPredictions * PREDICTION_SIZE);

              // Create sampler config with z-score normalization
              const samplerPtr = Module._malloc(24);
              const samplerView = new DataView(Module.HEAPU8.buffer, samplerPtr, 24);
              samplerView.setUint32(0, 1, true); // PSAM_LOGIT_ZSCORE
              samplerView.setFloat32(4, temp, true); // Use passed temperature parameter
              samplerView.setInt32(8, 0, true); // top_k (0 = use model default)
              samplerView.setFloat32(12, 0.95, true); // top_p
              samplerView.setBigUint64(16, BigInt(Math.floor(Math.random() * 0xFFFFFFFF)), true);

              const psam_predict_with_sampler = Module.cwrap('psam_predict_with_sampler', 'number',
                ['number', 'number', 'number', 'number', 'number', 'number']);
              const numPreds = psam_predict_with_sampler(handle, contextPtr, contextArray.length, samplerPtr, predsPtr, maxPredictions);

              const ids: number[] = [];
              const scores: number[] = [];
              const rawStrengths: number[] = [];
              const supportCounts: number[] = [];
              const probs: number[] = [];

              if (numPreds > 0) {
                for (let i = 0; i < numPreds; i++) {
                  const base = predsPtr + i * PREDICTION_SIZE;
                  const idx32 = base >>> 2;
                  ids.push(Module.HEAPU32[idx32]);
                  scores.push(Module.HEAPF32[idx32 + 1]);
                  rawStrengths.push(Module.HEAPF32[idx32 + 2]);
                  const idx16 = base >>> 1;
                  supportCounts.push(Module.HEAPU16[idx16 + 6]);
                  probs.push(Module.HEAPF32[idx32 + 4]);
                }
              }

              Module._free(contextPtr);
              Module._free(predsPtr);
              Module._free(samplerPtr);

              return {
                ids,
                scores: new Float32Array(scores),
                rawStrengths: new Float32Array(rawStrengths),
                supportCounts: new Uint16Array(supportCounts),
                probabilities: new Float32Array(probs)
              };
            },

            explain: (context: number[], candidateToken: number, maxTerms = 32) => {
              const contextArray = new Uint32Array(context);
              const contextPtr = Module._malloc(contextArray.length * 4);
              Module.HEAPU32.set(contextArray, contextPtr / 4);

              const termsPtr = maxTerms > 0 ? Module._malloc(maxTerms * 24) : 0; // 24 bytes per term
              const resultPtr = Module._malloc(16);

              const psam_explain = Module.cwrap('psam_explain', 'number',
                ['number', 'number', 'number', 'number', 'number', 'number', 'number']);
              const err = psam_explain(handle, contextPtr, contextArray.length, candidateToken, termsPtr, maxTerms, resultPtr);

              if (err < 0) {
                Module._free(contextPtr);
                if (termsPtr) Module._free(termsPtr);
                Module._free(resultPtr);
                throw new Error(`psam_explain failed with code ${err}`);
              }

              const base = resultPtr >>> 2;
              const candidate = Module.HEAPU32[base];
              const total = Module.HEAPF32[base + 1];
              const bias = Module.HEAPF32[base + 2];
              const termCount = Module.HEAP32[base + 3];

              const terms: any[] = [];

              if (termsPtr && termCount > 0) {
                const written = Math.min(termCount, maxTerms);
                const view = new DataView(Module.HEAPU8.buffer, termsPtr, written * 24);

                for (let i = 0; i < written; i++) {
                  const offset = i * 24;
                  terms.push({
                    source: view.getUint32(offset, true),
                    offset: view.getInt16(offset + 4, true),
                    weight: view.getFloat32(offset + 8, true),
                    idf: view.getFloat32(offset + 12, true),
                    decay: view.getFloat32(offset + 16, true),
                    contribution: view.getFloat32(offset + 20, true),
                  });
                }
              }

              Module._free(contextPtr);
              if (termsPtr) Module._free(termsPtr);
              Module._free(resultPtr);

              return {
                candidate,
                total,
                bias,
                termCount,
                terms,
              };
            },

            sample: (context: number[], temp = 1.0) => {
              // Pass temperature to predict
              const result = this.predict(context, 10, temp);
              if (result.ids.length === 0) return 0;

              // Use calibrated probabilities from sampler
              const probs = Array.from(result.probabilities);

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

  const sampleToken = (ids: number[], probs: number[]): number => {
    if (samplingMode === 'greedy') {
      return ids[0];
    } else {
      // Stochastic sampling
      const totalProb = probs.reduce((sum, value) => sum + value, 0);
      if (totalProb === 0) return ids[0];

      const target = Math.random() * totalProb;
      let cumsum = 0;

      for (let j = 0; j < probs.length; j++) {
        cumsum += probs[j];
        if (target <= cumsum) {
          return ids[j];
        }
      }

      return ids[ids.length - 1]; // fallback
    }
  };

  const generateStochasticSamples = (ids: number[], probs: number[], numSamples: number = 100) => {
    const samples: { [key: string]: number } = {};
    const wordMap: { [key: string]: number } = {};

    // Collect samples
    for (let i = 0; i < numSamples; i++) {
      const totalProb = probs.reduce((sum, value) => sum + value, 0);
      if (totalProb === 0) break;

      const target = Math.random() * totalProb;
      let cumsum = 0;
      let selectedIdx = 0;

      for (let j = 0; j < probs.length; j++) {
        cumsum += probs[j];
        if (target <= cumsum) {
          selectedIdx = j;
          break;
        }
      }

      const word = vocab[ids[selectedIdx]] || `<${ids[selectedIdx]}>`;
      samples[word] = (samples[word] || 0) + 1;
      wordMap[word] = probs[selectedIdx];
    }

    // Convert to array and sort by count
    const result = Object.entries(samples)
      .map(([word, count]) => ({
        word,
        count,
        probability: wordMap[word],
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);

    return result;
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
      const result = psam.predict(currentTokens.slice(-contextWindow), topK, temperature);

      if (result.ids.length === 0) break;

      // Use calibrated probabilities from sampler
      const probs = Array.from(result.probabilities);

      // Select token based on current sampling mode (matching "Generate Next" behavior)
      let selectedToken: number;
      let word: string;

      if (samplingMode === 'greedy') {
        // Greedy: always pick top prediction
        selectedToken = result.ids[0];
        word = vocab[selectedToken] || `<${selectedToken}>`;
      } else {
        // Stochastic: generate samples and pick the winner
        const samples = generateStochasticSamples(result.ids, probs, 100);
        if (samples.length > 0) {
          word = samples[0].word;
          // Find the token ID for this word
          const tokenId = vocab.indexOf(word);
          selectedToken = tokenId >= 0 ? tokenId : result.ids[0];
        } else {
          // Fallback
          selectedToken = result.ids[0];
          word = vocab[selectedToken] || `<${selectedToken}>`;
        }
      }

      const selectedIdx = result.ids.indexOf(selectedToken);

      const confidence = result.ids.length >= 2 ? result.scores[0] / result.scores[1] : Infinity;

      history.push({
        input: currentInput,
        predicted: word,
        score: result.scores[selectedIdx],
        rawStrength: result.rawStrengths[selectedIdx] ?? 0,
        supportCount: result.supportCounts[selectedIdx] ?? 0,
        probability: probs[selectedIdx],
        confidence,
        alternatives: result.ids.slice(1, 3).map((id, i) => ({
          word: vocab[id] || `<${id}>`,
          probability: probs[i + 1],
          rawStrength: result.rawStrengths[i + 1] ?? 0,
          supportCount: result.supportCounts[i + 1] ?? 0,
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
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4 md:p-8">
      <div className="bg-white rounded-lg shadow-lg p-4 md:p-8 mb-6">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-6">
            <h1 className="text-2xl sm:text-3xl font-bold text-gray-800">PSAM WASM Demo</h1>
            <div className="flex items-center gap-2 bg-green-100 px-3 py-1 rounded-full w-fit">
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
                  onClick={() => {
                    setText(scenario.text);
                    setInferenceInput(scenario.suggestedInput);
                    setTrained(false);
                    setPredictions([]);
                    setStochasticSamples([]);
                  }}
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
                <textarea
                  value={inferenceInput}
                  onChange={(e) => setInferenceInput(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent font-mono text-sm resize-y"
                  placeholder="Enter context..."
                  rows={2}
                />
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col sm:flex-row gap-3 mb-6">
                <button
                  onClick={() => {
                    if (!psam || !trained || predictions.length === 0) return;

                    let selectedWord: string;

                    if (samplingMode === 'greedy') {
                      // Greedy: always pick top prediction
                      selectedWord = predictions[0].word;
                    } else {
                      // Stochastic: use the top one from cached samples (already drawn)
                      if (stochasticSamples.length > 0) {
                        // The samples are already sorted by count (most frequent first)
                        // So just pick the winner from the 100 draws we already did
                        selectedWord = stochasticSamples[0].word;
                      } else {
                        // Fallback: use live sampling (shouldn't normally happen)
                        const ids = predictions.map(p => p.token);
                        const probs = predictions.map(p => p.probability);
                        const selectedToken = sampleToken(ids, probs);
                        const selectedPrediction = predictions.find(p => p.token === selectedToken) || predictions[0];
                        selectedWord = selectedPrediction.word;
                      }
                    }

                    setInferenceInput(inferenceInput + ' ' + selectedWord);
                  }}
                  disabled={!trained || predictions.length === 0}
                  className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-300 text-white font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                  <Zap className="w-5 h-5" />
                  <span className="hidden sm:inline">Generate Next</span>
                  <span className="sm:hidden">Next</span>
                </button>
                <button
                  onClick={handleGenerate}
                  disabled={!trained || isGenerating}
                  className="flex-1 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-300 text-white font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                  <RefreshCw className={`w-5 h-5 ${isGenerating ? 'animate-spin' : ''}`} />
                  <span className="hidden sm:inline">Auto-Generate (10)</span>
                  <span className="sm:hidden">Auto (10)</span>
                </button>
              </div>

              {/* Dual Predictions - Greedy vs Stochastic */}
              {predictions.length > 0 && (
                <div className="grid grid-cols-1 gap-4 mb-4">
                  {/* Greedy Panel */}
                  <div
                    onClick={() => setSamplingMode('greedy')}
                    className={`rounded-lg p-4 cursor-pointer transition-all ${
                      samplingMode === 'greedy'
                        ? 'bg-blue-50 border-2 border-blue-500 shadow-lg'
                        : 'bg-gray-50 border-2 border-gray-200 hover:border-blue-300'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-medium text-gray-800 flex items-center gap-2">
                        🎯 Greedy
                      </h4>
                      {samplingMode === 'greedy' && (
                        <span className="px-2 py-1 bg-blue-500 text-white text-xs rounded font-medium">
                          ACTIVE
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-gray-600 mb-3">
                      Always picks highest probability
                    </div>
                    <div className="space-y-2">
                      {predictions.slice(0, 5).map((pred, i) => (
                        <div key={i} className="flex items-center gap-2">
                          <div className="w-24 text-right font-mono text-[10px] leading-tight">
                            <div className="font-bold">{(pred.probability * 100).toFixed(1)}%</div>
                            <div className="text-gray-500">sup:{pred.supportCount}</div>
                          </div>
                          <div className="flex-1 bg-gray-200 rounded-full h-6 overflow-hidden">
                            <div
                              className={`h-full flex items-center px-2 text-white text-xs font-semibold ${
                                i === 0 ? 'bg-blue-600' : 'bg-blue-400'
                              }`}
                              style={{ width: `${pred.probability * 100}%` }}
                            >
                              {pred.word}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Stochastic Panel */}
                  <div
                    onClick={() => setSamplingMode('stochastic')}
                    className={`rounded-lg p-4 cursor-pointer transition-all ${
                      samplingMode === 'stochastic'
                        ? 'bg-purple-50 border-2 border-purple-500 shadow-lg'
                        : 'bg-gray-50 border-2 border-gray-200 hover:border-purple-300'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-medium text-gray-800 flex items-center gap-2">
                        🎲 Stochastic
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            if (predictions.length > 0) {
                              const samples = generateStochasticSamples(
                                predictions.map(p => p.token),
                                predictions.map(p => p.probability),
                                100
                              );
                              setStochasticSamples(samples);
                            }
                          }}
                          className="ml-2 p-1 hover:bg-purple-200 rounded text-xs"
                          title="Resample"
                        >
                          🔄
                        </button>
                      </h4>
                      {samplingMode === 'stochastic' && (
                        <span className="px-2 py-1 bg-purple-500 text-white text-xs rounded font-medium">
                          ACTIVE
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-gray-600 mb-3">
                      Results from 100 draws (top one wins)
                    </div>

                    {/* Temperature Control */}
                    <div className="mb-3 p-3 bg-white rounded border border-purple-200" onClick={(e) => e.stopPropagation()}>
                      <label className="block text-xs font-medium text-gray-700 mb-1">
                        Temperature
                      </label>
                      <input
                        type="number"
                        value={temperature}
                        onChange={(e) => setTemperature(Math.max(0.1, parseFloat(e.target.value)))}
                        className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                        step="0.1"
                        min="0.1"
                        max="10"
                      />
                      <div className="text-xs text-gray-500 mt-1">
                        Lower = focused, Higher = diverse
                      </div>
                    </div>

                    <div className="space-y-2">
                      {stochasticSamples.length > 0 ? (
                        stochasticSamples.map((sample, i) => (
                          <div key={i} className="flex items-center gap-2">
                            <div className="w-24 text-right font-mono text-[10px] leading-tight">
                              <div className="font-bold">{sample.count}/100</div>
                              <div className="text-gray-500">p:{(sample.probability * 100).toFixed(1)}%</div>
                            </div>
                            <div className="flex-1 bg-gray-200 rounded-full h-6 overflow-hidden">
                              <div
                                className="bg-purple-600 h-full flex items-center px-2 text-white text-xs font-semibold"
                                style={{ width: `${(sample.count / 100) * 100}%` }}
                              >
                                {sample.word}
                              </div>
                            </div>
                          </div>
                        ))
                      ) : (
                        predictions.slice(0, 5).map((pred, i) => (
                          <div key={i} className="flex items-center gap-2">
                            <div className="w-24 text-right font-mono text-[10px] leading-tight">
                              <div className="font-bold">{(pred.probability * 100).toFixed(1)}%</div>
                              <div className="text-gray-500">sup:{pred.supportCount}</div>
                            </div>
                            <div className="flex-1 bg-gray-200 rounded-full h-6 overflow-hidden">
                              <div
                                className="bg-purple-600 h-full flex items-center px-2 text-white text-xs font-semibold"
                                style={{ width: `${pred.probability * 100}%` }}
                              >
                                {pred.word}
                              </div>
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Explanation */}
              {explanation && (
                <div className="bg-amber-50 rounded-lg p-4 mb-4 border border-amber-200">
                  <h4 className="font-medium text-gray-800 mb-3 flex items-center gap-2">
                    <span className="text-amber-600">🔍</span>
                    Why predict "{explanation.token}"?
                  </h4>
                  <div className="text-xs text-gray-600 mb-3">
                    Bias {explanation.bias.toFixed(4)} + top {explanation.terms.length} / {explanation.termCount} contributions = total {explanation.total.toFixed(4)}
                  </div>
                  <div className="space-y-1.5 max-h-64 overflow-y-auto">
                    {explanation.terms.slice(0, 8).map((term, i) => (
                      <div key={i} className="font-mono text-xs bg-white p-2 rounded border border-amber-200">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-indigo-600 font-semibold">"{term.sourceWord}"</span>
                          <span className="text-gray-400">(offset {term.offset > 0 ? '+' : ''}{term.offset})</span>
                          <span className="text-green-600 ml-auto font-bold">
                            {term.contribution.toFixed(4)}
                          </span>
                        </div>
                        <div className="text-gray-500 text-[10px]">
                          weight:{term.weight.toFixed(3)} ×
                          idf:{term.idf.toFixed(3)} ×
                          decay:{term.decay.toFixed(3)}
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="mt-2 text-xs text-gray-500 italic">
                    Total score includes model bias plus all contributing terms.
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
                        <div className="text-xs text-gray-500 space-x-1">
                          <span>Score: {item.score.toFixed(3)}</span>
                          <span>Raw: {item.rawStrength.toFixed(3)}</span>
                          <span>Support: {item.supportCount}</span>
                          <span>Prob: {(item.probability * 100).toFixed(1)}%</span>
                          <span>Conf: {item.confidence.toFixed(2)}x</span>
                          {item.alternatives.length > 0 && (
                            <span className="ml-1">
                              | Alt: {item.alternatives
                                .map(
                                  (a) =>
                                    `"${a.word}" ${(a.probability * 100).toFixed(0)}% · raw ${a.rawStrength.toFixed(2)} · support ${a.supportCount}`
                                )
                                .join(', ')}
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
  );
};

export default PSAMWasmDemo;

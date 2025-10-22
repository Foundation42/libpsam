import { useState, useEffect } from 'react';
import { Play, Zap, Info } from 'lucide-react';

// Import WASM bindings (loaded via script tag in index.html)
declare global {
  interface Window {
    PSAM_WASM_PATH: string;
  }
}

// Type for WASM module (from psam-bindings.js)
interface PSAMWasm {
  create(vocabSize: number, window: number, topK: number): Promise<PSAMInstance>;
}

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

const PSAMWasmDemo = () => {
  const [psam, setPsam] = useState<PSAMInstance | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [text, setText] = useState("the quick brown fox jumps over the lazy dog");
  const [predictions, setPredictions] = useState<{ token: number; score: number }[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [trained, setTrained] = useState(false);

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
              // Use relative path to work with GitHub Pages base path
              return './wasm/' + path;
            }
            return path;
          }
        });

        // Create wrapper for PSAM instance
        const create = (vocabSize: number, window: number, topK: number): PSAMInstance => {
          const psam_create = Module.cwrap('psam_create', 'number', ['number', 'number', 'number']);
          const handle = psam_create(vocabSize, window, topK);

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
              const result = psam.predict(context);
              if (result.ids.length === 0) return 0;

              // Simple sampling - just return top prediction
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

        // Create PSAM instance
        const instance = create(100, 8, 32);
        setPsam(instance);
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

  const handleTrain = () => {
    if (!psam) return;

    try {
      // Simple tokenization: split by space and convert to numbers
      const words = text.toLowerCase().split(/\s+/);
      const vocab = [...new Set(words)];
      const tokens = words.map(w => vocab.indexOf(w));

      // Train
      psam.trainBatch(tokens);
      psam.finalizeTraining();

      // Get stats
      const modelStats = psam.stats();
      setStats(modelStats);
      setTrained(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Training failed');
    }
  };

  const handlePredict = () => {
    if (!psam || !trained) return;

    try {
      const words = text.toLowerCase().split(/\s+/);
      const vocab = [...new Set(words)];

      // Use last 3 words as context
      const contextWords = words.slice(-3);
      const contextTokens = contextWords.map(w => vocab.indexOf(w));

      // Predict
      const result = psam.predict(contextTokens, 10);

      // Convert to display format
      const preds = result.ids.map((id, i) => ({
        token: id,
        word: vocab[id] || `<${id}>`,
        score: result.scores[i]
      }));

      setPredictions(preds as any);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
    }
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

  if (error) {
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
              <p className="text-sm text-blue-800">
                This demo uses the actual libpsam WASM module compiled from C. It's <strong>20-200× faster</strong> than pure JavaScript!
              </p>
            </div>
          </div>

          {/* Training Text */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Training Text
            </label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              rows={4}
              placeholder="Enter text to train on..."
            />
          </div>

          {/* Train Button */}
          <button
            onClick={handleTrain}
            disabled={!psam}
            className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-300 text-white font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2 mb-6"
          >
            <Play className="w-5 h-5" />
            Train Model
          </button>

          {/* Stats */}
          {stats && (
            <div className="bg-gray-50 rounded-lg p-4 mb-6">
              <h3 className="font-medium text-gray-800 mb-2">Model Statistics</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
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
                  <div className="text-gray-500">Memory</div>
                  <div className="font-bold text-gray-800">{(stats.memoryBytes / 1024).toFixed(1)} KB</div>
                </div>
              </div>
            </div>
          )}

          {/* Predict Button */}
          <button
            onClick={handlePredict}
            disabled={!trained}
            className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-300 text-white font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2 mb-6"
          >
            <Zap className="w-5 h-5" />
            Predict Next Token
          </button>

          {/* Predictions */}
          {predictions.length > 0 && (
            <div className="bg-gray-50 rounded-lg p-4">
              <h3 className="font-medium text-gray-800 mb-3">Predictions</h3>
              <div className="space-y-2">
                {predictions.slice(0, 5).map((pred, i) => (
                  <div key={i} className="flex items-center gap-3">
                    <div className="bg-indigo-100 text-indigo-800 px-2 py-1 rounded text-sm font-mono">
                      {pred.word}
                    </div>
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-indigo-600 h-2 rounded-full transition-all"
                        style={{ width: `${Math.min(100, (pred.score + 10) * 5)}%` }}
                      />
                    </div>
                    <div className="text-sm text-gray-600 w-16 text-right">
                      {pred.score.toFixed(2)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PSAMWasmDemo;

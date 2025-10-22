/**
 * WASM bindings for libpsam
 *
 * Provides browser-compatible PSAM using WebAssembly
 */

import type { TokenId, InferenceResult, ModelStats, TrainablePSAM, PersistenceOptions } from './types.js';

// Type for the Emscripten module
interface EmscriptenModule {
  _malloc(size: number): number;
  _free(ptr: number): void;
  HEAPU32: Uint32Array;
  HEAPF32: Float32Array;
  cwrap: (name: string, returnType: string | null, argTypes: string[]) => Function;
  ccall: (name: string, returnType: string | null, argTypes: string[], args: any[]) => any;
  UTF8ToString(ptr: number): string;
}

// Singleton WASM module instance
let wasmModuleInstance: EmscriptenModule | null = null;
let wasmModulePromise: Promise<EmscriptenModule> | null = null;

/**
 * Load the WASM module (call once)
 */
export async function loadPSAMModule(): Promise<EmscriptenModule> {
  if (wasmModuleInstance) {
    return wasmModuleInstance;
  }

  if (wasmModulePromise) {
    return wasmModulePromise;
  }

  wasmModulePromise = new Promise((resolve, reject) => {
    // @ts-ignore - createPSAMModule is provided by psam.js
    if (typeof createPSAMModule === 'undefined') {
      reject(new Error('PSAM WASM module not loaded. Include psam.js before this script.'));
      return;
    }

    // @ts-ignore
    createPSAMModule({
      locateFile: (path: string) => {
        // Allow customization via global variable
        // @ts-ignore
        if (typeof PSAM_WASM_PATH !== 'undefined') {
          // @ts-ignore
          return PSAM_WASM_PATH + '/' + path;
        }
        return path;
      }
    }).then((module: EmscriptenModule) => {
      wasmModuleInstance = module;
      resolve(module);
    }).catch(reject);
  });

  return wasmModulePromise;
}

/**
 * WASM PSAM implementation (browser-compatible)
 */
export class PSAMWASM implements TrainablePSAM {
  private module: EmscriptenModule;
  private handle: number;
  private _vocabSize: number;
  private _window: number;
  private _topK: number;

  private constructor(module: EmscriptenModule, handle: number, vocabSize: number, window: number, topK: number) {
    this.module = module;
    this.handle = handle;
    this._vocabSize = vocabSize;
    this._window = window;
    this._topK = topK;
  }

  /**
   * Create a new PSAM model
   */
  static async create(vocabSize: number, window: number, topK: number): Promise<PSAMWASM> {
    const module = await loadPSAMModule();

    const psam_create = module.cwrap('psam_create', 'number', ['number', 'number', 'number']);
    const handle = psam_create(vocabSize, window, topK);

    if (!handle) {
      throw new Error('Failed to create PSAM model');
    }

    return new PSAMWASM(module, handle, vocabSize, window, topK);
  }

  trainToken(token: TokenId): void {
    const psam_train_token = this.module.cwrap('psam_train_token', 'number', ['number', 'number']);
    psam_train_token(this.handle, token);
  }

  train(tokens: TokenId[]): void {
    for (const token of tokens) {
      this.trainToken(token);
    }
  }

  trainBatch(tokens: TokenId[] | Uint32Array): void {
    const tokensArray = tokens instanceof Uint32Array ? tokens : new Uint32Array(tokens);
    const numTokens = tokensArray.length;

    // Allocate memory for tokens
    const tokensPtr = this.module._malloc(numTokens * 4);
    this.module.HEAPU32.set(tokensArray, tokensPtr / 4);

    // Call psam_train_batch
    const psam_train_batch = this.module.cwrap('psam_train_batch', 'number', ['number', 'number', 'number']);
    psam_train_batch(this.handle, tokensPtr, numTokens);

    // Free memory
    this.module._free(tokensPtr);
  }

  finalizeTraining(): void {
    const psam_finalize_training = this.module.cwrap('psam_finalize_training', 'number', ['number']);
    const result = psam_finalize_training(this.handle);

    if (result !== 0) {
      throw new Error(`Failed to finalize training: error code ${result}`);
    }
  }

  predict(context: TokenId[], maxPredictions?: number): InferenceResult {
    const contextArray = new Uint32Array(context);
    const maxPreds = maxPredictions || this._topK;

    // Allocate memory for context
    const contextPtr = this.module._malloc(contextArray.length * 4);
    this.module.HEAPU32.set(contextArray, contextPtr / 4);

    // Allocate memory for predictions (12 bytes per prediction: token + score + calibrated_prob)
    const predsPtr = this.module._malloc(maxPreds * 12);

    // Call psam_predict
    const psam_predict = this.module.cwrap('psam_predict', 'number', ['number', 'number', 'number', 'number', 'number']);
    const numPreds = psam_predict(this.handle, contextPtr, contextArray.length, predsPtr, maxPreds);

    // Read predictions
    const ids: number[] = [];
    const scores: number[] = [];

    if (numPreds > 0) {
      for (let i = 0; i < numPreds; i++) {
        const offset = predsPtr / 4 + i * 3; // 3 floats per prediction (token is uint32, score/prob are float)
        ids.push(this.module.HEAPU32[offset]);
        scores.push(this.module.HEAPF32[offset + 1]);
      }
    }

    // Free memory
    this.module._free(contextPtr);
    this.module._free(predsPtr);

    return { ids, scores };
  }

  sample(context: TokenId[], temperature: number = 1.0): TokenId {
    const result = this.predict(context);

    if (result.ids.length === 0) {
      throw new Error('No predictions available');
    }

    // Apply temperature and softmax
    const logits = result.scores.map(s => s / temperature);
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
  }

  addLayer(layerId: string, overlay: PSAMWASM, weight: number): void {
    // Not implemented in browser version (would require string marshalling)
    throw new Error('Layer composition not yet implemented in WASM');
  }

  removeLayer(layerId: string): void {
    throw new Error('Layer composition not yet implemented in WASM');
  }

  updateLayerWeight(layerId: string, newWeight: number): void {
    throw new Error('Layer composition not yet implemented in WASM');
  }

  save(path: string, options?: PersistenceOptions): void {
    // Not implemented in browser (would need filesystem API or download)
    throw new Error('Save not implemented in WASM (use serialization APIs instead)');
  }

  static load(path: string): Promise<PSAMWASM> {
    throw new Error('Load not implemented in WASM (use serialization APIs instead)');
  }

  stats(): ModelStats {
    // Allocate memory for stats struct
    const statsPtr = this.module._malloc(32); // Approximate size

    const psam_get_stats = this.module.cwrap('psam_get_stats', 'number', ['number', 'number']);
    psam_get_stats(this.handle, statsPtr);

    // Read stats (this is approximate, actual struct may differ)
    const statsView = new Uint32Array(this.module.HEAPU32.buffer, statsPtr, 8);

    const stats: ModelStats = {
      vocabSize: statsView[0],
      totalTokens: Number(BigInt(statsView[1]) | (BigInt(statsView[2]) << 32n)),
      rowCount: statsView[3],
      edgeCount: Number(BigInt(statsView[4]) | (BigInt(statsView[5]) << 32n)),
      memoryBytes: Number(BigInt(statsView[6]) | (BigInt(statsView[7]) << 32n)),
    };

    this.module._free(statsPtr);
    return stats;
  }

  destroy(): void {
    if (this.handle) {
      const psam_destroy = this.module.cwrap('psam_destroy', null, ['number']);
      psam_destroy(this.handle);
      this.handle = 0;
    }
  }
}

export function isWASMAvailable(): boolean {
  return typeof WebAssembly !== 'undefined';
}

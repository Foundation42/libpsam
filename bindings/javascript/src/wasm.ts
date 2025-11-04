/**
 * WASM bindings for libpsam
 *
 * Provides browser-compatible PSAM using WebAssembly
 */

import type { TokenId, InferenceResult, ModelStats, TrainablePSAM, PersistenceOptions, ExplainTerm, ExplainResult, SamplerConfig } from './types.js';
import { LogitTransform } from './types.js';

// Type for the Emscripten module
interface EmscriptenModule {
  _malloc(size: number): number;
  _free(ptr: number): void;
  HEAPU32: Uint32Array;
  HEAP32: Int32Array;
  HEAPF32: Float32Array;
  HEAPU16: Uint16Array;
  HEAPU8: Uint8Array;
  cwrap: (name: string, returnType: string | null, argTypes: string[]) => Function;
  ccall: (name: string, returnType: string | null, argTypes: string[], args: any[]) => any;
  UTF8ToString(ptr: number): string;
}

const EXPLAIN_TERM_SIZE = 24;
const EXPLAIN_RESULT_SIZE = 16;
const SAMPLER_SIZE = 24;

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
 * Helper to serialize sampler config to memory buffer
 */
function serializeSampler(sampler: SamplerConfig): ArrayBuffer {
  const buffer = new ArrayBuffer(SAMPLER_SIZE);
  const view = new DataView(buffer);

  // psam_sampler_t layout (24 bytes):
  // uint32_t transform (offset 0)
  // float temperature (offset 4)
  // int32_t top_k (offset 8)
  // float top_p (offset 12)
  // uint64_t seed (offset 16)

  view.setUint32(0, sampler.transform ?? LogitTransform.ZSCORE, true);
  view.setFloat32(4, sampler.temperature ?? 1.0, true);
  view.setInt32(8, sampler.topK ?? 0, true);
  view.setFloat32(12, sampler.topP ?? 0.95, true);

  const seed = sampler.seed ?? Math.floor(Math.random() * 0xFFFFFFFF);
  view.setBigUint64(16, BigInt(seed), true);

  return buffer;
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

  predict(context: TokenId[], maxPredictions?: number, sampler?: SamplerConfig): InferenceResult {
    const contextArray = new Uint32Array(context);
    const maxPreds = maxPredictions || this._topK;

    // Allocate memory for context
    const contextPtr = this.module._malloc(contextArray.length * 4);
    this.module.HEAPU32.set(contextArray, contextPtr / 4);

    // Allocate memory for predictions (12 bytes per prediction: token + score + calibrated_prob)
    const PREDICTION_SIZE = 20;
    const predsPtr = this.module._malloc(maxPreds * PREDICTION_SIZE);

    let numPreds: number;

    if (sampler) {
      // Use sampler API
      const samplerBuffer = serializeSampler(sampler);
      const samplerPtr = this.module._malloc(SAMPLER_SIZE);
      this.module.HEAPU8.set(new Uint8Array(samplerBuffer), samplerPtr);

      const psam_predict_with_sampler = this.module.cwrap('psam_predict_with_sampler', 'number',
        ['number', 'number', 'number', 'number', 'number', 'number']);
      numPreds = psam_predict_with_sampler(this.handle, contextPtr, contextArray.length, samplerPtr, predsPtr, maxPreds);

      this.module._free(samplerPtr);
    } else {
      // Use legacy API
      const psam_predict = this.module.cwrap('psam_predict', 'number', ['number', 'number', 'number', 'number', 'number']);
      numPreds = psam_predict(this.handle, contextPtr, contextArray.length, predsPtr, maxPreds);
    }

    // Read predictions
    const ids: number[] = [];
    const scoresArray: number[] = [];
    const rawStrengthArray: number[] = [];
    const supportCountArray: number[] = [];
    const probabilitiesArray: number[] = [];

    if (numPreds > 0) {
      for (let i = 0; i < numPreds; i++) {
        const base = predsPtr + i * PREDICTION_SIZE;
        const idx32 = base >>> 2;
        ids.push(this.module.HEAPU32[idx32]);
        scoresArray.push(this.module.HEAPF32[idx32 + 1]);
        rawStrengthArray.push(this.module.HEAPF32[idx32 + 2]);
        const idx16 = base >>> 1;
        supportCountArray.push(this.module.HEAPU16[idx16 + 6]);
        probabilitiesArray.push(this.module.HEAPF32[idx32 + 4]);
      }
    }

    // Free memory
    this.module._free(contextPtr);
    this.module._free(predsPtr);

    return {
      ids,
      scores: new Float32Array(scoresArray),
      rawStrengths: new Float32Array(rawStrengthArray),
      supportCounts: new Uint16Array(supportCountArray),
      probabilities: new Float32Array(probabilitiesArray)
    };
  }

  explain(context: TokenId[], candidateToken: TokenId, maxTerms?: number): ExplainResult {
    const contextArray = new Uint32Array(context);
    const maxT = maxTerms ?? 32;

    const contextPtr = this.module._malloc(contextArray.length * 4);
    this.module.HEAPU32.set(contextArray, contextPtr / 4);

    const termsPtr = maxT > 0 ? this.module._malloc(maxT * EXPLAIN_TERM_SIZE) : 0;
    const resultPtr = this.module._malloc(EXPLAIN_RESULT_SIZE);

    const psam_explain = this.module.cwrap('psam_explain', 'number',
      ['number', 'number', 'number', 'number', 'number', 'number', 'number']);

    const err = psam_explain(
      this.handle,
      contextPtr,
      contextArray.length,
      candidateToken,
      termsPtr,
      maxT,
      resultPtr,
    );

    if (err < 0) {
      this.module._free(contextPtr);
      if (termsPtr) this.module._free(termsPtr);
      this.module._free(resultPtr);
      throw new Error(`psam_explain failed with code ${err}`);
    }

    const baseIndex = resultPtr >>> 2;
    const candidate = this.module.HEAPU32[baseIndex];
    const total = this.module.HEAPF32[baseIndex + 1];
    const bias = this.module.HEAPF32[baseIndex + 2];
    const termCount = this.module.HEAP32[baseIndex + 3];

    const terms: ExplainTerm[] = [];

    if (termsPtr && termCount > 0) {
      const written = Math.min(termCount, maxT);
      const view = new DataView(this.module.HEAPU8.buffer, termsPtr, written * EXPLAIN_TERM_SIZE);

      for (let i = 0; i < written; i++) {
        const base = i * EXPLAIN_TERM_SIZE;
        terms.push({
          source: view.getUint32(base, true),
          offset: view.getInt16(base + 4, true),
          weight: view.getFloat32(base + 8, true),
          idf: view.getFloat32(base + 12, true),
          decay: view.getFloat32(base + 16, true),
          contribution: view.getFloat32(base + 20, true),
        });
      }
    }

    this.module._free(contextPtr);
    if (termsPtr) this.module._free(termsPtr);
    this.module._free(resultPtr);

    return {
      candidate,
      total,
      bias,
      termCount,
      terms,
    };
  }

  sample(context: TokenId[], temperature: number = 1.0, topP: number = 0.95): TokenId {
    // Use sampler API for consistent behavior
    const sampler: SamplerConfig = {
      transform: LogitTransform.ZSCORE,
      temperature,
      topK: this._topK,
      topP,
      seed: Math.floor(Math.random() * 0xFFFFFFFF)
    };

    const result = this.predict(context, undefined, sampler);

    if (result.ids.length === 0) {
      throw new Error('No predictions available');
    }

    // Sample from calibrated probabilities
    const probs = result.probabilities || result.scores;
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

  save(path: string, options?: PersistenceOptions): void {
    // Not implemented in browser (would need filesystem API or download)
    throw new Error('Save not implemented in WASM (use serialization APIs instead)');
  }

  static load(path: string): Promise<PSAMWASM> {
    throw new Error('Load not implemented in WASM (use serialization APIs instead)');
  }

  stats(): ModelStats {
    // psam_stats_t struct layout (from psam.h):
    // uint32_t vocab_size     (4 bytes, offset 0)
    // uint32_t row_count      (4 bytes, offset 4)
    // uint64_t edge_count     (8 bytes, offset 8)
    // uint64_t total_tokens   (8 bytes, offset 16)
    // size_t memory_bytes     (4/8 bytes, offset 24)
    // Total: 28 bytes (32-bit) or 32 bytes (64-bit)

    const statsPtr = this.module._malloc(32);

    const psam_get_stats = this.module.cwrap('psam_get_stats', 'number', ['number', 'number']);
    psam_get_stats(this.handle, statsPtr);

    // Read as 32-bit words
    const view32 = new Uint32Array(this.module.HEAPU32.buffer, statsPtr, 8);

    const stats: ModelStats = {
      vocabSize: view32[0],
      rowCount: view32[1],
      edgeCount: Number(BigInt(view32[2]) | (BigInt(view32[3]) << 32n)),
      totalTokens: Number(BigInt(view32[4]) | (BigInt(view32[5]) << 32n)),
      memoryBytes: Number(BigInt(view32[6]) | (BigInt(view32[7]) << 32n)),
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
  // Use indirect access to avoid TypeScript error
  return typeof (globalThis as any).WebAssembly !== 'undefined';
}

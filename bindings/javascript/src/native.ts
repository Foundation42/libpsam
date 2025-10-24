/**
 * Native bindings for libpsam using FFI
 *
 * Supports both Bun (bun:ffi) and Node.js (via dynamic detection)
 */

import type { TokenId, InferenceResult, ModelStats, TrainablePSAM, PersistenceOptions, ExplainTerm } from './types.js';

// Platform-specific FFI loading
let FFI: any = null;
let isBun = false;

try {
  // Try Bun first
  // @ts-ignore - optional bun-specific module
  FFI = await import('bun:ffi');
  isBun = true;
} catch {
  try {
    // Fall back to Node.js ffi-napi (if available)
    // @ts-ignore - optional Node.js module
    FFI = await import('ffi-napi');
  } catch {
    console.warn('No FFI implementation available. Native bindings require Bun or ffi-napi.');
  }
}

/* ============================ Error Handling ============================ */

enum PSAMError {
  OK = 0,
  NULL_PARAM = -1,
  INVALID_CONFIG = -2,
  OUT_OF_MEMORY = -3,
  IO = -4,
  INVALID_MODEL = -5,
  NOT_TRAINED = -6,
  LAYER_NOT_FOUND = -7,
}

function checkError(code: number, operation: string, lib: any): void {
  if (code < 0) {
    const errMsg = lib?.symbols?.psam_error_string?.(code) || lib?.psam_error_string?.(code);
    throw new Error(`libpsam ${operation} failed: ${errMsg || `error code ${code}`}`);
  }
}

/* ============================ Library Loading ============================ */

const PREDICTION_SIZE = 12; // sizeof(psam_prediction_t)
const STATS_SIZE = 32; // sizeof(psam_stats_t)
const EXPLAIN_TERM_SIZE = 28; // sizeof(psam_explain_term_t)

let cachedLib: any = null;

function loadLibrary(libraryPath?: string): any {
  if (cachedLib) return cachedLib;

  const path = libraryPath ||
    process.env.LIBPSAM_PATH ||
    './libpsam.so'; // Default path

  if (isBun && FFI) {
    // Bun FFI
    const { dlopen, FFIType } = FFI;

    cachedLib = dlopen(path, {
      psam_create: { args: [FFIType.u32, FFIType.u32, FFIType.u32], returns: FFIType.ptr },
      psam_destroy: { args: [FFIType.ptr], returns: FFIType.void },
      psam_train_token: { args: [FFIType.ptr, FFIType.u32], returns: FFIType.i32 },
      psam_train_batch: { args: [FFIType.ptr, FFIType.ptr, FFIType.u64], returns: FFIType.i32 },
      psam_finalize_training: { args: [FFIType.ptr], returns: FFIType.i32 },
      psam_predict: {
        args: [FFIType.ptr, FFIType.ptr, FFIType.u64, FFIType.ptr, FFIType.u64],
        returns: FFIType.i32,
      },
      psam_explain: {
        args: [FFIType.ptr, FFIType.ptr, FFIType.u64, FFIType.u32, FFIType.ptr, FFIType.u64],
        returns: FFIType.i32,
      },
      psam_add_layer: { args: [FFIType.ptr, FFIType.cstring, FFIType.ptr, FFIType.f32], returns: FFIType.i32 },
      psam_remove_layer: { args: [FFIType.ptr, FFIType.cstring], returns: FFIType.i32 },
      psam_update_layer_weight: { args: [FFIType.ptr, FFIType.cstring, FFIType.f32], returns: FFIType.i32 },
      psam_save: { args: [FFIType.ptr, FFIType.cstring], returns: FFIType.i32 },
      psam_load: { args: [FFIType.cstring], returns: FFIType.ptr },
      psam_get_stats: { args: [FFIType.ptr, FFIType.ptr], returns: FFIType.i32 },
      psam_error_string: { args: [FFIType.i32], returns: FFIType.cstring },
      psam_version: { args: [], returns: FFIType.cstring },
    });
  } else if (FFI) {
    // Node.js ffi-napi
    const ffi = FFI;
    cachedLib = ffi.Library(path, {
      psam_create: ['pointer', ['uint32', 'uint32', 'uint32']],
      psam_destroy: ['void', ['pointer']],
      psam_train_token: ['int32', ['pointer', 'uint32']],
      psam_train_batch: ['int32', ['pointer', 'pointer', 'uint64']],
      psam_finalize_training: ['int32', ['pointer']],
      psam_predict: ['int32', ['pointer', 'pointer', 'uint64', 'pointer', 'uint64']],
      psam_explain: ['int32', ['pointer', 'pointer', 'uint64', 'uint32', 'pointer', 'uint64']],
      psam_add_layer: ['int32', ['pointer', 'string', 'pointer', 'float']],
      psam_remove_layer: ['int32', ['pointer', 'string']],
      psam_update_layer_weight: ['int32', ['pointer', 'string', 'float']],
      psam_save: ['int32', ['pointer', 'string']],
      psam_load: ['pointer', ['string']],
      psam_get_stats: ['int32', ['pointer', 'pointer']],
      psam_error_string: ['string', ['int32']],
      psam_version: ['string', []],
    });
  } else {
    throw new Error('No FFI implementation available');
  }

  return cachedLib;
}

export function isNativeAvailable(): boolean {
  try {
    loadLibrary();
    return true;
  } catch {
    return false;
  }
}

/* ============================ PSAMNative Class ============================ */

/**
 * Native PSAM implementation using libpsam via FFI
 *
 * Provides 20-200× performance improvement over pure JavaScript
 */
export class PSAMNative implements TrainablePSAM {
  private handle: any;
  private lib: any;
  private _vocabSize: number;
  private _window: number;
  private _topK: number;

  constructor(vocabSize: number, window: number, topK: number, libraryPath?: string) {
    this.lib = loadLibrary(libraryPath);

    const symbols = this.lib.symbols || this.lib;
    this.handle = symbols.psam_create(vocabSize, window, topK);

    if (!this.handle) {
      throw new Error('Failed to create PSAM model');
    }

    this._vocabSize = vocabSize;
    this._window = window;
    this._topK = topK;
  }

  trainToken(token: TokenId): void {
    const symbols = this.lib.symbols || this.lib;
    const result = symbols.psam_train_token(this.handle, token);
    checkError(result, 'trainToken', this.lib);
  }

  train(tokens: TokenId[]): void {
    for (const token of tokens) {
      this.trainToken(token);
    }
    this.finalizeTraining();
  }

  trainBatch(tokens: TokenId[] | Uint32Array): void {
    const symbols = this.lib.symbols || this.lib;
    const tokensArray = tokens instanceof Uint32Array ? tokens : new Uint32Array(tokens);

    const result = symbols.psam_train_batch(
      this.handle,
      tokensArray,
      BigInt(tokensArray.length)
    );
    checkError(result, 'trainBatch', this.lib);
  }

  finalizeTraining(): void {
    const symbols = this.lib.symbols || this.lib;
    const result = symbols.psam_finalize_training(this.handle);
    checkError(result, 'finalizeTraining', this.lib);
  }

  predict(context: TokenId[], maxPredictions?: number): InferenceResult {
    const symbols = this.lib.symbols || this.lib;
    const limit = maxPredictions ?? this._topK;

    const outBuffer = new Uint8Array(limit * PREDICTION_SIZE);
    const contextArray = new Uint32Array(context);

    const numPreds = symbols.psam_predict(
      this.handle,
      contextArray,
      BigInt(contextArray.length),
      outBuffer,
      BigInt(limit)
    );

    if (numPreds < 0) {
      checkError(numPreds, 'predict', this.lib);
    }

    const ids: TokenId[] = [];
    const scores = new Float32Array(numPreds);
    const view = new DataView(outBuffer.buffer);

    for (let i = 0; i < numPreds; i++) {
      const offset = i * PREDICTION_SIZE;
      ids.push(view.getUint32(offset, true));
      scores[i] = view.getFloat32(offset + 4, true);
    }

    return { ids, scores };
  }

  explain(context: TokenId[], candidateToken: TokenId, maxTerms?: number): ExplainTerm[] {
    const symbols = this.lib.symbols || this.lib;
    const limit = maxTerms ?? 32;

    const outBuffer = new Uint8Array(limit * EXPLAIN_TERM_SIZE);
    const contextArray = new Uint32Array(context);

    const numTerms = symbols.psam_explain(
      this.handle,
      contextArray,
      BigInt(contextArray.length),
      candidateToken,
      outBuffer,
      BigInt(limit)
    );

    if (numTerms < 0) {
      checkError(numTerms, 'explain', this.lib);
    }

    const terms: ExplainTerm[] = [];
    const view = new DataView(outBuffer.buffer);

    for (let i = 0; i < numTerms; i++) {
      const offset = i * EXPLAIN_TERM_SIZE;
      terms.push({
        sourceToken: view.getUint32(offset, true),
        sourcePosition: view.getInt32(offset + 4, true),
        relativeOffset: view.getInt32(offset + 8, true),
        baseWeight: view.getFloat32(offset + 12, true),
        idfFactor: view.getFloat32(offset + 16, true),
        distanceDecay: view.getFloat32(offset + 20, true),
        contribution: view.getFloat32(offset + 24, true),
      });
    }

    return terms;
  }

  sample(context: TokenId[], temperature: number = 1.0): TokenId {
    const result = this.predict(context, this._topK);

    if (result.ids.length === 0) {
      throw new Error('No predictions available');
    }

    // Apply temperature and sample
    const logits = Array.from(result.scores).map(s => s / temperature);
    const maxLogit = Math.max(...logits);
    const expScores = logits.map(l => Math.exp(l - maxLogit));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    const probs = expScores.map(e => e / sumExp);

    const rand = Math.random();
    let cumProb = 0;

    for (let i = 0; i < probs.length; i++) {
      cumProb += probs[i];
      if (rand <= cumProb) {
        return result.ids[i];
      }
    }

    return result.ids[0];
  }

  addLayer(layerId: string, overlay: PSAMNative, weight: number): void {
    const symbols = this.lib.symbols || this.lib;
    const result = symbols.psam_add_layer(this.handle, layerId, overlay.handle, weight);
    checkError(result, 'addLayer', this.lib);
  }

  removeLayer(layerId: string): void {
    const symbols = this.lib.symbols || this.lib;
    const result = symbols.psam_remove_layer(this.handle, layerId);
    checkError(result, 'removeLayer', this.lib);
  }

  updateLayerWeight(layerId: string, newWeight: number): void {
    const symbols = this.lib.symbols || this.lib;
    const result = symbols.psam_update_layer_weight(this.handle, layerId, newWeight);
    checkError(result, 'updateLayerWeight', this.lib);
  }

  save(path: string, _options?: PersistenceOptions): void {
    const symbols = this.lib.symbols || this.lib;
    const result = symbols.psam_save(this.handle, path);
    checkError(result, `save to ${path}`, this.lib);
  }

  static load(path: string, libraryPath?: string): PSAMNative {
    const lib = loadLibrary(libraryPath);
    const symbols = lib.symbols || lib;

    const handle = symbols.psam_load(path);
    if (!handle) {
      throw new Error(`Failed to load model from ${path}`);
    }

    const instance = Object.create(PSAMNative.prototype);
    instance.handle = handle;
    instance.lib = lib;

    const stats = instance.stats();
    instance._vocabSize = stats.vocabSize;
    instance._window = 8; // Default, not stored in stats
    instance._topK = 32; // Default, not stored in stats

    return instance;
  }

  stats(): ModelStats {
    const symbols = this.lib.symbols || this.lib;
    const statsBuffer = new Uint8Array(STATS_SIZE);

    const result = symbols.psam_get_stats(this.handle, statsBuffer);
    checkError(result, 'getStats', this.lib);

    const view = new DataView(statsBuffer.buffer);
    return {
      vocabSize: view.getUint32(0, true),
      rowCount: view.getUint32(4, true),
      edgeCount: Number(view.getBigUint64(8, true)),
      totalTokens: Number(view.getBigUint64(16, true)),
      memoryBytes: Number(view.getBigUint64(24, true)),
    };
  }

  destroy(): void {
    if (!this.handle) return;

    const symbols = this.lib.symbols || this.lib;
    symbols.psam_destroy(this.handle);
    this.handle = null;
  }

  static version(): string {
    try {
      const lib = loadLibrary();
      const symbols = lib.symbols || lib;
      return symbols.psam_version() || 'unknown';
    } catch {
      return 'unavailable';
    }
  }
}

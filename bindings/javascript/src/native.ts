/**
 * Native bindings for libpsam using FFI
 *
 * Supports both Bun (bun:ffi) and Node.js (via dynamic detection)
 */

import type {
  TokenId,
  InferenceResult,
  ModelStats,
  TrainablePSAM,
  PersistenceOptions,
  ExplainTerm,
  ExplainResult,
  LayeredComposite,
  CompositeLayerInfo,
  PSAM,
  SaveCompositeOptions,
  CompositeLayerDescriptor,
} from './types.js';

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

export function loadComposite(
  path: string,
  verifyIntegrity: boolean = true,
  libraryPath?: string
): LayeredCompositeNative {
  return LayeredCompositeNative.loadFromFile(path, verifyIntegrity, libraryPath);
}

export function saveComposite(options: SaveCompositeOptions, libraryPath?: string): void {
  if (!options || !options.outPath || !options.baseModelPath) {
    throw new Error('saveComposite requires outPath and baseModelPath');
  }

  if (isBun) {
    throw new Error('saveComposite is currently only supported on Node.js builds');
  }

  const ref = ensureRefModule();
  if (!ref) {
    throw new Error('ref-napi is required to use saveComposite (install ffi-napi and ref-napi)');
  }

  const lib = loadLibrary(libraryPath);
  const symbols = lib.symbols || lib;
  if (!symbols.psam_composite_save_file) {
    throw new Error('libpsam does not export psam_composite_save_file');
  }

  const layers = options.layers ?? [];
  const layerCount = layers.length;
  let layerBuffer: Buffer | null = null;

  const allocatedStrings: Buffer[] = [];

  if (layerCount > 0) {
    layerBuffer = Buffer.alloc(layerCount * SIZE_OF_LAYER_FILE_STRUCT);

    for (let i = 0; i < layerCount; i++) {
      const desc: CompositeLayerDescriptor = layers[i];
      if (!desc || !desc.path) {
        throw new Error(`overlay #${i} is missing a path`);
      }

      const idPtr = desc.id ? ref.allocCString(desc.id) : ref.NULL;
      const pathPtr = ref.allocCString(desc.path);
      if (idPtr && idPtr !== ref.NULL) {
        allocatedStrings.push(idPtr);
      }
      allocatedStrings.push(pathPtr);

      const weight = typeof desc.weight === 'number' ? desc.weight : 1.0;
      const offset = i * SIZE_OF_LAYER_FILE_STRUCT;
      ref.writePointer(layerBuffer, offset, idPtr);
      if (is64Bit) {
        layerBuffer.writeFloatLE(weight, offset + 8);
        layerBuffer.writeUInt32LE(0, offset + 12);
        ref.writePointer(layerBuffer, offset + 16, pathPtr);
      } else {
        layerBuffer.writeFloatLE(weight, offset + 4);
        ref.writePointer(layerBuffer, offset + 8, pathPtr);
      }
    }
  }

  const result = symbols.psam_composite_save_file(
    options.outPath,
    options.createdBy ?? null,
    null,
    options.baseWeight ?? 1.0,
    options.baseModelPath,
    BigInt(layerCount),
    layerBuffer ?? ref.NULL,
  );
  checkError(result, 'saveComposite', lib);
}

let refModule: any = null;
function ensureRefModule(): any {
  if (isBun) {
    return null;
  }
  if (refModule) {
    return refModule;
  }
  try {
    const nodeRequire = Function('return typeof require !== "undefined" ? require : null;')();
    if (nodeRequire) {
      refModule = nodeRequire('ref-napi');
    }
  } catch {
    refModule = null;
  }
  return refModule;
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
const EXPLAIN_TERM_SIZE = 24; // sizeof(psam_explain_term_t)
const EXPLAIN_RESULT_SIZE = 16; // sizeof(psam_explain_result_t)
const PSAM_LAYER_ID_MAX = 64;
const COMPOSITE_LAYER_INFO_SIZE = 68;
const is64Bit = process.arch === 'x64' || process.arch === 'arm64';
const SIZE_OF_LAYER_FILE_STRUCT = is64Bit ? 24 : 12;
const textDecoder = new TextDecoder();

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
        args: [FFIType.ptr, FFIType.ptr, FFIType.u64, FFIType.u32, FFIType.ptr, FFIType.i32, FFIType.ptr],
        returns: FFIType.i32,
      },
      psam_create_layered: { args: [FFIType.ptr], returns: FFIType.ptr },
      psam_composite_destroy: { args: [FFIType.ptr], returns: FFIType.void },
      psam_composite_set_base_weight: { args: [FFIType.ptr, FFIType.f32], returns: FFIType.i32 },
      psam_composite_add_layer: { args: [FFIType.ptr, FFIType.cstring, FFIType.ptr, FFIType.f32], returns: FFIType.i32 },
      psam_composite_remove_layer: { args: [FFIType.ptr, FFIType.cstring], returns: FFIType.i32 },
      psam_composite_update_layer_weight: { args: [FFIType.ptr, FFIType.cstring, FFIType.f32], returns: FFIType.i32 },
      psam_composite_list_layers: { args: [FFIType.ptr, FFIType.ptr, FFIType.u64], returns: FFIType.i32 },
      psam_composite_predict: {
        args: [FFIType.ptr, FFIType.ptr, FFIType.u64, FFIType.ptr, FFIType.u64],
        returns: FFIType.i32,
      },
      psam_composite_load_file: { args: [FFIType.cstring, FFIType.i32], returns: FFIType.ptr },
      psam_composite_save_file: {
        args: [
          FFIType.cstring,
          FFIType.cstring,
          FFIType.ptr,
          FFIType.f32,
          FFIType.cstring,
          FFIType.u64,
          FFIType.ptr,
        ],
        returns: FFIType.i32,
      },
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
      psam_explain: ['int32', ['pointer', 'pointer', 'uint64', 'uint32', 'pointer', 'int32', 'pointer']],
      psam_create_layered: ['pointer', ['pointer']],
      psam_composite_destroy: ['void', ['pointer']],
      psam_composite_set_base_weight: ['int32', ['pointer', 'float']],
      psam_composite_add_layer: ['int32', ['pointer', 'string', 'pointer', 'float']],
      psam_composite_remove_layer: ['int32', ['pointer', 'string']],
      psam_composite_update_layer_weight: ['int32', ['pointer', 'string', 'float']],
      psam_composite_list_layers: ['int32', ['pointer', 'pointer', 'uint64']],
      psam_composite_predict: ['int32', ['pointer', 'pointer', 'uint64', 'pointer', 'uint64']],
      psam_composite_load_file: ['pointer', ['string', 'int']],
      psam_composite_save_file: [
        'int32',
        ['string', 'string', 'pointer', 'float', 'string', 'uint64', 'pointer'],
      ],
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

  explain(context: TokenId[], candidateToken: TokenId, maxTerms?: number): ExplainResult {
    const symbols = this.lib.symbols || this.lib;
    const limit = maxTerms ?? 32;

    const contextArray = new Uint32Array(context);
    const resultBuffer = new Uint8Array(EXPLAIN_RESULT_SIZE);

    const termBuffer = limit > 0 ? new Uint8Array(limit * EXPLAIN_TERM_SIZE) : null;

    const err = symbols.psam_explain(
      this.handle,
      contextArray,
      BigInt(contextArray.length),
      candidateToken,
      termBuffer ?? 0,
      limit,
      resultBuffer
    );

    if (err < 0) {
      checkError(err, 'explain', this.lib);
    }

    const resultView = new DataView(resultBuffer.buffer);
    const candidate = resultView.getUint32(0, true);
    const total = resultView.getFloat32(4, true);
    const bias = resultView.getFloat32(8, true);
    const termCount = resultView.getInt32(12, true);

    const terms: ExplainTerm[] = [];

    if (termBuffer && termCount > 0) {
      const view = new DataView(termBuffer.buffer);
      const written = Math.min(termCount, limit);

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

    return {
      candidate,
      total,
      bias,
      termCount,
      terms,
    };
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

  createLayeredComposite(): LayeredCompositeNative {
    const symbols = this.lib.symbols || this.lib;
    const handle = symbols.psam_create_layered(this.handle);
    if (!handle) {
      throw new Error('Failed to create layered composite (is the model finalized?)');
    }
    return new LayeredCompositeNative(this.lib, handle, this);
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

  get topK(): number {
    return this._topK;
  }

  /** @internal */
  getNativeHandle(): any {
    return this.handle;
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

export class LayeredCompositeNative implements LayeredComposite {
  private handle: any;
  private lib: any;
  private baseTopK: number;

  constructor(lib: any, handle: any, base?: PSAMNative, defaultTopK: number = 32) {
    this.lib = lib;
    this.handle = handle;
    this.baseTopK = base ? base.topK : defaultTopK;
  }

  private get symbols() {
    return this.lib.symbols || this.lib;
  }

  destroy(): void {
    if (this.handle) {
      this.symbols.psam_composite_destroy(this.handle);
      this.handle = null;
    }
  }

  setBaseWeight(weight: number): void {
    const result = this.symbols.psam_composite_set_base_weight(this.handle, weight);
    checkError(result, 'composite_set_base_weight', this.lib);
  }

  addLayer(layerId: string, overlay: PSAM, weight: number): void {
    if (!(overlay instanceof PSAMNative)) {
      throw new Error('Layered composites currently require PSAMNative overlays');
    }
    const nativeHandle = (overlay as PSAMNative).getNativeHandle();
    const result = this.symbols.psam_composite_add_layer(this.handle, layerId, nativeHandle, weight);
    checkError(result, 'composite_add_layer', this.lib);
  }

  removeLayer(layerId: string): void {
    const result = this.symbols.psam_composite_remove_layer(this.handle, layerId);
    checkError(result, 'composite_remove_layer', this.lib);
  }

  updateLayerWeight(layerId: string, newWeight: number): void {
    const result = this.symbols.psam_composite_update_layer_weight(this.handle, layerId, newWeight);
    checkError(result, 'composite_update_layer_weight', this.lib);
  }

  listLayers(maxLayers: number = 16): CompositeLayerInfo[] {
    if (maxLayers <= 0) {
      return [];
    }

    const buffer = new Uint8Array(maxLayers * COMPOSITE_LAYER_INFO_SIZE);
    const count = this.symbols.psam_composite_list_layers(this.handle, buffer, BigInt(maxLayers));
    if (count < 0) {
      checkError(count, 'composite_list_layers', this.lib);
      return [];
    }

    const layers: CompositeLayerInfo[] = [];
    const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

    for (let i = 0; i < Math.min(count, maxLayers); i++) {
      const baseOffset = i * COMPOSITE_LAYER_INFO_SIZE;
      let end = baseOffset;
      while (end < baseOffset + PSAM_LAYER_ID_MAX && buffer[end] !== 0) {
        end++;
      }
      const idBytes = buffer.slice(baseOffset, end);
      const id = textDecoder.decode(idBytes);
      const weight = view.getFloat32(baseOffset + PSAM_LAYER_ID_MAX, true);
      layers.push({ id, weight });
    }

    return layers;
  }

  predict(context: TokenId[], maxPredictions: number = this.baseTopK): InferenceResult {
    if (context.length === 0) {
      return { ids: [], scores: new Float32Array() };
    }

    const outBuffer = new Uint8Array(maxPredictions * PREDICTION_SIZE);
    const contextArray = new Uint32Array(context);

    const count = this.symbols.psam_composite_predict(
      this.handle,
      contextArray,
      BigInt(contextArray.length),
      outBuffer,
      BigInt(maxPredictions)
    );

    if (count < 0) {
      checkError(count, 'composite_predict', this.lib);
      return { ids: [], scores: new Float32Array() };
    }

    const ids: TokenId[] = [];
    const scores = new Float32Array(count);
    const view = new DataView(outBuffer.buffer, outBuffer.byteOffset, outBuffer.byteLength);

    for (let i = 0; i < count; i++) {
      const offset = i * PREDICTION_SIZE;
      ids.push(view.getUint32(offset, true));
      scores[i] = view.getFloat32(offset + 4, true);
    }

    return { ids, scores };
  }

  sample(context: TokenId[], temperature: number = 1.0): TokenId {
    const result = this.predict(context, this.baseTopK);
    if (result.ids.length === 0) {
      throw new Error('No predictions available');
    }

    const logits = Array.from(result.scores).map(score => score / temperature);
    const maxLogit = Math.max(...logits);
    const expScores = logits.map(l => Math.exp(l - maxLogit));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    const probs = expScores.map(e => e / sumExp);

    const rand = Math.random();
    let acc = 0;
    for (let i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (rand < acc) {
        return result.ids[i];
      }
    }

    return result.ids[0];
  }

  static loadFromFile(path: string, verifyIntegrity: boolean = true, libraryPath?: string): LayeredCompositeNative {
    const lib = loadLibrary(libraryPath);
    const symbols = lib.symbols || lib;
    const handle = symbols.psam_composite_load_file(path, verifyIntegrity ? 1 : 0);
    if (!handle) {
      throw new Error(`Failed to load composite from ${path}`);
    }
    return new LayeredCompositeNative(lib, handle, undefined, 32);
  }
}

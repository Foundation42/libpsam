/**
 * libpsam - JavaScript/TypeScript bindings
 *
 * Auto-detects the best available implementation:
 * 1. Native (FFI) - Fastest, requires libpsam.so
 * 2. WASM - Browser-compatible, moderate performance
 */

export * from './types.js';
export { PSAMNative, LayeredCompositeNative, isNativeAvailable } from './native.js';
export { PSAMWASM, isWASMAvailable } from './wasm.js';

import { PSAMNative, LayeredCompositeNative, isNativeAvailable, saveComposite as nativeSaveComposite } from './native.js';
import type { SaveCompositeOptions } from './types.js';
import { PSAMWASM, isWASMAvailable } from './wasm.js';
import type { TrainablePSAM } from './types.js';

/**
 * Implementation preference
 */
export type Implementation = 'native' | 'wasm' | 'auto';

/**
 * Get the best available PSAM implementation
 */
export function getBestImplementation(prefer: Implementation = 'auto'): typeof PSAMNative | typeof PSAMWASM {
  if (prefer === 'native') {
    if (!isNativeAvailable()) {
      throw new Error('Native implementation not available. Build libpsam.so first.');
    }
    return PSAMNative;
  }

  if (prefer === 'wasm') {
    if (!isWASMAvailable()) {
      throw new Error('WASM implementation not available.');
    }
    return PSAMWASM;
  }

  // Auto mode: try native first, fall back to WASM
  if (isNativeAvailable()) {
    return PSAMNative;
  }

  if (isWASMAvailable()) {
    return PSAMWASM;
  }

  throw new Error('No PSAM implementation available. Build native library or WASM module.');
}

/**
 * Create a PSAM model using the best available implementation
 */
export async function createPSAM(
  vocabSize: number,
  window: number,
  topK: number,
  prefer: Implementation = 'auto'
): Promise<TrainablePSAM> {
  const Implementation = getBestImplementation(prefer);

  // PSAMWASM uses async create(), PSAMNative uses constructor
  if (Implementation === PSAMWASM) {
    return await PSAMWASM.create(vocabSize, window, topK);
  } else {
    return new PSAMNative(vocabSize, window, topK);
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
  nativeSaveComposite(options, libraryPath);
}

/**
 * Default export for convenience
 */
export default {
  createPSAM,
  loadComposite,
  saveComposite,
  getBestImplementation,
  PSAMNative,
  LayeredCompositeNative,
  PSAMWASM,
  isNativeAvailable,
  isWASMAvailable,
};

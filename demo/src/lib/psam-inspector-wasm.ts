/**
 * WASM bindings for PSAM Inspector
 *
 * Provides functions to load and inspect .psam files using the WASM module.
 */

export interface PSAMEdge {
  source: number;
  target: number;
  offset: number;
  weight: number;
  observations: number;
}

export interface PSAMConfig {
  vocabSize: number;
  window: number;
  topK: number;
  alpha: number;
  minEvidence: number;
  enableIdf: boolean;
  enablePpmi: boolean;
  edgeDropout: number;
}

export interface PSAMStats {
  vocabSize: number;
  rowCount: number;
  edgeCount: number;
  totalTokens: number;
  memoryBytes: number;
}

export class PSAMInspectorWASM {
  private Module: any;

  constructor(Module: any) {
    this.Module = Module;
  }

  /**
   * Load a PSAM model from a memory buffer (e.g., from File API)
   */
  loadFromMemory(uint8Array: Uint8Array): number {
    const dataPtr = this.Module._malloc(uint8Array.length);
    this.Module.HEAPU8.set(uint8Array, dataPtr);

    const psam_load_from_memory = this.Module.cwrap(
      'psam_load_from_memory',
      'number',
      ['number', 'number']
    );

    const handle = psam_load_from_memory(dataPtr, uint8Array.length);
    this.Module._free(dataPtr);

    if (!handle) {
      throw new Error('Failed to load PSAM from memory');
    }

    return handle;
  }

  /**
   * Extract edges from a model for visualization
   *
   * @param handle Model handle
   * @param sourceToken Filter by source token (0xFFFFFFFF for all)
   * @param minWeight Minimum edge weight threshold
   * @param maxEdges Maximum number of edges to return
   */
  getEdges(
    handle: number,
    sourceToken: number = 0xFFFFFFFF,
    minWeight: number = 0.0,
    maxEdges: number = 1000
  ): PSAMEdge[] {
    console.log(`[WASM] getEdges(handle=${handle}, source=${sourceToken.toString(16)}, minWeight=${minWeight}, max=${maxEdges})`);

    const EDGE_SIZE = 20; // sizeof(psam_edge_t)
    const edgesPtr = this.Module._malloc(maxEdges * EDGE_SIZE);

    const psam_get_edges = this.Module.cwrap(
      'psam_get_edges',
      'number',
      ['number', 'number', 'number', 'number', 'number']
    );

    const numEdges = psam_get_edges(
      handle,
      sourceToken,
      minWeight,
      maxEdges,
      edgesPtr
    );

    console.log(`[WASM] psam_get_edges returned: ${numEdges}`);

    if (numEdges < 0) {
      this.Module._free(edgesPtr);
      throw new Error(`psam_get_edges failed with code ${numEdges}`);
    }

    const edges: PSAMEdge[] = [];
    const view = new DataView(this.Module.HEAPU8.buffer, edgesPtr, numEdges * EDGE_SIZE);

    for (let i = 0; i < numEdges; i++) {
      const offset = i * EDGE_SIZE;
      edges.push({
        source: view.getUint32(offset, true),
        target: view.getUint32(offset + 4, true),
        offset: view.getInt16(offset + 8, true),
        weight: view.getFloat32(offset + 12, true),
        observations: view.getUint32(offset + 16, true),
      });
    }

    this.Module._free(edgesPtr);
    return edges;
  }

  /**
   * Get model configuration
   */
  getConfig(handle: number): PSAMConfig {
    const configPtr = this.Module._malloc(32); // sizeof(psam_config_t)

    const psam_get_config = this.Module.cwrap(
      'psam_get_config',
      'number',
      ['number', 'number']
    );

    const err = psam_get_config(handle, configPtr);

    if (err !== 0) {
      this.Module._free(configPtr);
      throw new Error(`psam_get_config failed with code ${err}`);
    }

    // Use DataView for correct struct layout (28 bytes)
    // Layout: vocab(0), window(4), topK(8), alpha(12), minEv(16), idf(20), ppmi(21), [pad], dropout(24)
    const view = new DataView(this.Module.HEAPU8.buffer, configPtr, 28);

    const config: PSAMConfig = {
      vocabSize: view.getUint32(0, true),
      window: view.getUint32(4, true),
      topK: view.getUint32(8, true),
      alpha: view.getFloat32(12, true),
      minEvidence: view.getFloat32(16, true),
      enableIdf: view.getUint8(20) !== 0,
      enablePpmi: view.getUint8(21) !== 0,
      edgeDropout: view.getFloat32(24, true),
    };

    this.Module._free(configPtr);
    return config;
  }

  /**
   * Debug: Check model internal state
   */
  debugModel(handle: number): void {
    console.log('[WASM Debug] Checking model internals...');

    // Try calling psam_get_stats to see if model is valid
    const statsPtr = this.Module._malloc(32);
    const psam_get_stats = this.Module.cwrap('psam_get_stats', 'number', ['number', 'number']);
    const err = psam_get_stats(handle, statsPtr);

    if (err !== 0) {
      console.error('[WASM Debug] psam_get_stats failed:', err);
    } else {
      const view32 = new Uint32Array(this.Module.HEAPU32.buffer, statsPtr, 8);
      console.log('[WASM Debug] Stats:', {
        vocabSize: view32[0],
        rowCount: view32[1],
        edgeCount: Number(BigInt(view32[2]) | (BigInt(view32[3]) << 32n)),
      });
    }
    this.Module._free(statsPtr);
  }

  /**
   * Get model statistics (using existing API)
   */
  getStats(handle: number): PSAMStats {
    const statsPtr = this.Module._malloc(32);
    const psam_get_stats = this.Module.cwrap('psam_get_stats', 'number', ['number', 'number']);
    psam_get_stats(handle, statsPtr);

    const view32 = new Uint32Array(this.Module.HEAPU32.buffer, statsPtr, 8);

    const stats: PSAMStats = {
      vocabSize: view32[0],
      rowCount: view32[1],
      edgeCount: Number(BigInt(view32[2]) | (BigInt(view32[3]) << 32n)),
      totalTokens: Number(BigInt(view32[4]) | (BigInt(view32[5]) << 32n)),
      memoryBytes: Number(BigInt(view32[6]) | (BigInt(view32[7]) << 32n)),
    };

    this.Module._free(statsPtr);
    return stats;
  }

  /**
   * Generate predictions for a context (token IDs)
   */
  predict(handle: number, contextTokens: number[], maxPreds: number = 20): Array<{
    token: number;
    score: number;
    probability: number;
  }> {
    const contextArray = new Uint32Array(contextTokens);
    const contextPtr = this.Module._malloc(contextArray.length * 4);
    this.Module.HEAPU32.set(contextArray, contextPtr / 4);

    const PREDICTION_SIZE = 20; // sizeof(psam_prediction_t) = 20 bytes
    const predsPtr = this.Module._malloc(maxPreds * PREDICTION_SIZE);

    const psam_predict = this.Module.cwrap(
      'psam_predict',
      'number',
      ['number', 'number', 'number', 'number', 'number']
    );

    const numPreds = psam_predict(handle, contextPtr, contextArray.length, predsPtr, maxPreds);

    const predictions: Array<{ token: number; score: number; probability: number }> = [];

    if (numPreds > 0) {
      for (let i = 0; i < numPreds; i++) {
        const base = predsPtr + i * PREDICTION_SIZE;
        const idx32 = base >>> 2;
        predictions.push({
          token: this.Module.HEAPU32[idx32],
          score: this.Module.HEAPF32[idx32 + 1],
          probability: this.Module.HEAPF32[idx32 + 4], // offset 16 bytes / 4 = index 4
        });
      }
    }

    this.Module._free(contextPtr);
    this.Module._free(predsPtr);

    return predictions;
  }

  /**
   * Explain prediction: get contributing terms for a candidate token
   */
  explain(handle: number, context: number[], candidateToken: number, maxTerms: number = 32): {
    candidate: number;
    total: number;
    bias: number;
    termCount: number;
    terms: Array<{
      source: number;
      offset: number;
      weight: number;
      idf: number;
      decay: number;
      contribution: number;
    }>;
  } {
    const contextArray = new Uint32Array(context);
    const contextPtr = this.Module._malloc(contextArray.length * 4);
    this.Module.HEAPU32.set(contextArray, contextPtr / 4);

    const termsPtr = maxTerms > 0 ? this.Module._malloc(maxTerms * 24) : 0; // 24 bytes per term
    const resultPtr = this.Module._malloc(16);

    const psam_explain = this.Module.cwrap('psam_explain', 'number',
      ['number', 'number', 'number', 'number', 'number', 'number', 'number']);
    const err = psam_explain(handle, contextPtr, contextArray.length, candidateToken, termsPtr, maxTerms, resultPtr);

    if (err < 0) {
      this.Module._free(contextPtr);
      if (termsPtr) this.Module._free(termsPtr);
      this.Module._free(resultPtr);
      throw new Error(`psam_explain failed with code ${err}`);
    }

    const base = resultPtr >>> 2;
    const candidate = this.Module.HEAPU32[base];
    const total = this.Module.HEAPF32[base + 1];
    const bias = this.Module.HEAPF32[base + 2];
    const termCount = this.Module.HEAP32[base + 3];

    const terms: any[] = [];

    if (termsPtr && termCount > 0) {
      const written = Math.min(termCount, maxTerms);
      const view = new DataView(this.Module.HEAPU8.buffer, termsPtr, written * 24);

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

    this.Module._free(contextPtr);
    if (termsPtr) this.Module._free(termsPtr);
    this.Module._free(resultPtr);

    return {
      candidate,
      total,
      bias,
      termCount,
      terms,
    };
  }

  /**
   * Destroy a model and free memory
   */
  destroy(handle: number): void {
    const psam_destroy = this.Module.cwrap('psam_destroy', null, ['number']);
    psam_destroy(handle);
  }
}

/**
 * Initialize the WASM module and return an inspector instance
 */
export async function initInspectorWASM(): Promise<PSAMInspectorWASM> {
  // Wait for the WASM module to load (same as existing demo)
  if (typeof window === 'undefined' || !(window as any).createPSAMModule) {
    throw new Error('PSAM WASM module not loaded');
  }

  const Module = await (window as any).createPSAMModule({
    locateFile: (path: string) => {
      const base = (import.meta as any).env?.BASE_URL ?? '/';
      const normalizedBase = base.endsWith('/') ? base.slice(0, -1) : base;
      return path.endsWith('.wasm') ? `${normalizedBase}/wasm/${path}` : path;
    }
  });

  return new PSAMInspectorWASM(Module);
}

/**
 * Type definitions for libpsam JavaScript/TypeScript bindings
 */

export type TokenId = number;

/**
 * Training configuration options
 */
export interface TrainOptions {
  /** Context window size */
  window: number;
  /** Laplace smoothing parameter */
  alpha: number;
  /** Distance decay parameter */
  gamma: number;
  /** IDF smoothing parameter */
  eta: number;
  /** Top-K for function words */
  kFunction: number;
  /** Top-K for content words */
  kContent: number;
  /** Top-K for rare words */
  kRare: number;
  /** Edge dropout probability */
  edgeDropout: number;
  /** Minimum evidence threshold */
  minEvidence: number;
  /** Enable IDF weighting */
  enableIDF: boolean;
  /** Enable PPMI transformation */
  enablePPMI: boolean;
}

/**
 * Default training options
 */
export const DEFAULT_TRAIN_OPTIONS: TrainOptions = {
  window: 8,
  alpha: 0.1,
  gamma: 0.05,
  eta: 1.0,
  kFunction: 8,
  kContent: 32,
  kRare: 64,
  edgeDropout: 0.0,
  minEvidence: 2,
  enableIDF: true,
  enablePPMI: true,
};

/**
 * Prediction result
 */
export interface Prediction {
  /** Token ID */
  tokenId: TokenId;
  /** Raw score */
  score: number;
  /** Calibrated probability (if available) */
  probability?: number;
}

/**
 * Inference result with multiple predictions
 */
export interface InferenceResult {
  /** Predicted token IDs */
  ids: TokenId[];
  /** Raw scores */
  scores: Float32Array;
  /** Calibrated probabilities (optional) */
  probabilities?: Float32Array;
}

/**
 * Explanation term showing why a token was predicted
 */
export interface ExplainTerm {
  /** Source token that contributed */
  sourceToken: TokenId;
  /** Position of source token in context */
  sourcePosition: number;
  /** Relative position offset */
  relativeOffset: number;
  /** Base association weight (PPMI-adjusted) */
  baseWeight: number;
  /** IDF weighting factor */
  idfFactor: number;
  /** Distance decay factor */
  distanceDecay: number;
  /** Final contribution (weight × idf × decay) */
  contribution: number;
}

/**
 * Model statistics
 */
export interface ModelStats {
  /** Vocabulary size */
  vocabSize: number;
  /** Number of CSR rows */
  rowCount: number;
  /** Total number of edges */
  edgeCount: number;
  /** Total tokens processed during training */
  totalTokens?: number;
  /** Memory usage in bytes */
  memoryBytes: number;
}

/**
 * Save/load options
 */
export interface PersistenceOptions {
  /** File format (currently only 'binary' is supported by native lib) */
  format?: 'binary';
}

/**
 * Core PSAM interface
 */
export interface PSAM {
  /**
   * Add an overlay layer for domain adaptation
   */
  addLayer(layerId: string, overlay: PSAM, weight: number): void;

  /**
   * Remove a layer by ID
   */
  removeLayer(layerId: string): void;

  /**
   * Update layer weight
   */
  updateLayerWeight(layerId: string, newWeight: number): void;

  /**
   * Predict next tokens given context
   */
  predict(context: TokenId[], maxPredictions?: number): InferenceResult;

  /**
   * Explain why a specific token was predicted for the given context.
   * Returns the top contributing association terms with full traceability.
   */
  explain(context: TokenId[], candidateToken: TokenId, maxTerms?: number): ExplainTerm[];

  /**
   * Sample a single token from the distribution
   */
  sample(context: TokenId[], temperature?: number): TokenId;

  /**
   * Get model statistics
   */
  stats(): ModelStats;

  /**
   * Save model to file
   */
  save(path: string, options?: PersistenceOptions): void | Promise<void>;

  /**
   * Destroy model and free resources
   */
  destroy?(): void;
}

/**
 * Trainable PSAM interface
 */
export interface TrainablePSAM extends PSAM {
  /**
   * Train on a sequence of tokens
   */
  train(tokens: TokenId[], options?: Partial<TrainOptions>): void;

  /**
   * Process a single token during training
   */
  trainToken(token: TokenId): void;

  /**
   * Finalize training (compute PPMI/IDF, build CSR)
   */
  finalizeTraining(): void;
}

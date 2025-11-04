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
 * Logit transform modes for temperature sampling
 */
export enum LogitTransform {
  /** Raw scores (no normalization) */
  RAW = 0,
  /** Z-score normalization (recommended, default) */
  ZSCORE = 1,
  /** Calibrated (reserved for future use) */
  CALIBRATED = 2,
  /** Legacy mode (pre-1.1 behavior) */
  LEGACY = 3,
}

/**
 * Sampler configuration for temperature control
 */
export interface SamplerConfig {
  /** Logit transform mode (default: ZSCORE) */
  transform?: LogitTransform;
  /** Temperature for sampling (default: 1.0, recommended range: 0.1-2.0 for ZSCORE) */
  temperature?: number;
  /** Top-K filtering (default: 0 = use model default) */
  topK?: number;
  /** Top-P nucleus sampling (default: 0.95) */
  topP?: number;
  /** Random seed for reproducibility (default: random) */
  seed?: number;
}

/**
 * Prediction result
 */
export interface Prediction {
  /** Token ID */
  tokenId: TokenId;
  /** Raw score */
  score: number;
  /** Contextual contribution sum (bias excluded) */
  rawStrength: number;
  /** Number of contributing context tokens */
  supportCount: number;
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
  /** Sum of contextual contributions per token */
  rawStrengths: Float32Array;
  /** Number of supporting context tokens */
  supportCounts: Uint16Array;
  /** Calibrated probabilities (optional) */
  probabilities?: Float32Array;
}

/**
 * Explanation term showing why a token was predicted
 */
export interface ExplainTerm {
  /** Source token that contributed */
  source: TokenId;
  /** Relative position delta (negative => token precedes candidate) */
  offset: number;
  /** Base association weight (PPMI-adjusted) */
  weight: number;
  /** IDF weighting factor */
  idf: number;
  /** Distance decay factor */
  decay: number;
  /** Final contribution (weight × idf × decay) */
  contribution: number;
}

/**
 * Aggregated explanation metadata
 */
export interface ExplainResult {
  /** Token being explained */
  candidate: TokenId;
  /** Final sampler score (bias + contributions) */
  total: number;
  /** Bias component for the candidate */
  bias: number;
  /** Total contributing terms discovered */
  termCount: number;
  /** Top contributing terms (length ≤ requested maxTerms) */
  terms: ExplainTerm[];
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
  /** Optional creator string recorded in composite manifests */
  createdBy?: string;
}

/**
 * Core PSAM interface
 */
export interface PSAM {
  /**
   * Predict next tokens given context
   */
  predict(context: TokenId[], maxPredictions?: number, sampler?: SamplerConfig): InferenceResult;

  /**
   * Explain why a specific token was predicted for the given context.
   * Returns score metadata and the top contributing association terms.
   */
  explain(context: TokenId[], candidateToken: TokenId, maxTerms?: number): ExplainResult;

  /**
   * Sample a single token from the distribution
   * @deprecated Use predict() with sampler config instead for more control
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

  /**
   * Build a layered composite using this model as the base.
   */
  createLayeredComposite?(): LayeredComposite;
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

export interface CompositeLayerInfo {
  id: string;
  weight: number;
  bias?: number;
}

export interface LayeredComposite {
  setBaseWeight(weight: number): void;
  addLayer(layerId: string, overlay: PSAM, weight: number): void;
  removeLayer(layerId: string): void;
  updateLayerWeight(layerId: string, newWeight: number): void;
  updateLayerBias?(layerId: string, newBias: number): void;
  listLayers(maxLayers?: number): CompositeLayerInfo[];
  predict(context: TokenId[], maxPredictions?: number, sampler?: SamplerConfig): InferenceResult;
  sample(context: TokenId[], temperature?: number): TokenId;
  destroy(): void;
  save?(options: SaveCompositeOptions): void;
}

export interface CompositeLayerDescriptor {
  id?: string;
  weight?: number;
  path: string;
}

export interface SaveCompositeOptions {
  outPath: string;
  baseModelPath: string;
  baseWeight?: number;
  layers: CompositeLayerDescriptor[];
  createdBy?: string;
  hyperparamsPreset?: 'balanced' | 'fast' | 'accurate' | 'tiny';
}

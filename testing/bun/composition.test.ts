import { afterAll, beforeAll, describe, expect, it } from 'bun:test';

import {
  LayeredCompositeNative,
  LogitTransform,
  PSAMNative,
  createPSAM,
  isNativeAvailable,
  type SamplerConfig,
} from '../../bindings/javascript/src/index.js';

type VocabInfo = {
  tokenToId: Map<string, number>;
  idToToken: Map<number, string>;
  tokenToVocab: Map<string, string>;
};

const VOCAB_SETS: Record<string, string[]> = {
  A: ['apple', 'ant', 'arrow', 'anchor', 'atlas', 'axe', 'angel', 'arch'],
  B: ['ball', 'bat', 'bear', 'boat', 'bell', 'bird', 'bone', 'bread'],
  C: ['cat', 'car', 'cave', 'coin', 'crown', 'cloud', 'cup', 'cliff'],
};

const SAMPLER: SamplerConfig = {
  transform: LogitTransform.ZSCORE,
  temperature: 1.0,
  topP: 0.95,
  topK: 0,
  seed: 4242,
};

function buildVocab(...sets: string[]): VocabInfo {
  const tokens: string[] = [];
  const tokenToVocab = new Map<string, string>();

  for (const name of sets) {
    const subset = VOCAB_SETS[name];
    subset.forEach(token => {
      tokens.push(token);
      tokenToVocab.set(token, name);
    });
  }

  const tokenToId = new Map<string, number>();
  const idToToken = new Map<number, string>();

  tokens.forEach((token, idx) => {
    tokenToId.set(token, idx);
    idToToken.set(idx, token);
  });

  return { tokenToId, idToToken, tokenToVocab };
}

function repeatingSequence(vocab: string[], pattern: number[], length: number): string[] {
  const seq: string[] = [];
  const patternLen = pattern.length;
  for (let i = 0; i < length; i += 1) {
    seq.push(vocab[pattern[i % patternLen]]);
  }
  return seq;
}

function encode(sequence: string[], tokenToId: Map<string, number>): number[] {
  return sequence.map(token => {
    const id = tokenToId.get(token);
    if (id === undefined) {
      throw new Error(`Token "${token}" missing from vocabulary`);
    }
    return id;
  });
}

async function trainModel(tokens: number[], vocabSize: number): Promise<PSAMNative> {
  const model = (await createPSAM(vocabSize, 4, 32, 'native')) as PSAMNative;
  const batch = Uint32Array.from(tokens);
  model.trainBatch(batch);
  model.finalizeTraining();
  return model;
}

function buildDistribution(
  composite: LayeredCompositeNative,
  context: number[],
  vocab: VocabInfo,
  sampler: SamplerConfig,
  topK = 64,
): Map<string, number> {
  const result = composite.predict(context, topK, sampler);
  const { ids, probabilities } = result;
  if (!probabilities) {
    throw new Error('Sampler must return probability estimates');
  }

  const mass = new Map<string, number>();
  for (let i = 0; i < ids.length; i += 1) {
    const token = vocab.idToToken.get(ids[i]);
    if (!token) {
      continue;
    }
    const vocabName = vocab.tokenToVocab.get(token);
    if (!vocabName) {
      continue;
    }
    const current = mass.get(vocabName) ?? 0;
    mass.set(vocabName, current + probabilities[i]);
  }

  const total = Array.from(mass.values()).reduce((acc, value) => acc + value, 0);
  if (total > 0) {
    for (const [key, value] of mass.entries()) {
      mass.set(key, (value / total) * 100);
    }
  }

  return mass;
}

function expectDistribution(mass: Map<string, number>, expected: Record<string, number>, tolerance = 20): void {
  for (const [vocab, target] of Object.entries(expected)) {
    const actual = mass.get(vocab) ?? 0;
    expect(Math.abs(actual - target)).toBeLessThanOrEqual(tolerance);
  }
}

describe('PSAM composition (Bun bindings)', () => {
  if (!isNativeAvailable()) {
    it('skipped because native library is unavailable', () => {
      expect(true).toBe(true);
    });
    return;
  }

  const vocab = buildVocab('A', 'B', 'C');
  let modelA: PSAMNative;
  let modelB: PSAMNative;
  let modelC: PSAMNative;

  beforeAll(async () => {
    const length = 512;
    const seqA = encode(repeatingSequence(VOCAB_SETS.A, [0, 1, 2, 0, 1, 2], length), vocab.tokenToId);
    const seqB = encode(repeatingSequence(VOCAB_SETS.B, [0, 2, 1, 0, 2, 1], length), vocab.tokenToId);
    const seqC = encode(repeatingSequence(VOCAB_SETS.C, [0, 1, 2, 3, 4, 5], length), vocab.tokenToId);

    modelA = await trainModel(seqA, vocab.tokenToId.size);
    modelB = await trainModel(seqB, vocab.tokenToId.size);
    modelC = await trainModel(seqC, vocab.tokenToId.size);
  });

  afterAll(() => {
    modelC.destroy();
    modelB.destroy();
    modelA.destroy();
  });

  it('respects pure dominance', () => {
    const composite = modelA.createLayeredComposite();
    try {
      composite.setBaseWeight(1.0);
      composite.addLayer('overlay_b', modelB, 0.0);

      const context = [vocab.tokenToId.get('apple')!];
      const mass = buildDistribution(composite, context, vocab, SAMPLER);

      expectDistribution(mass, { A: 100, B: 0 });
    } finally {
      composite.destroy();
    }
  });

  it('mixes layers at 50/50', () => {
    const composite = modelA.createLayeredComposite();
    try {
      composite.setBaseWeight(0.5);
      composite.addLayer('overlay_b', modelB, 0.5);

      const context = [vocab.tokenToId.get('apple')!];
      const mass = buildDistribution(composite, context, vocab, SAMPLER);

      expectDistribution(mass, { A: 50, B: 50 });
    } finally {
      composite.destroy();
    }
  });

  it('respects weighted blends', () => {
    const composite = modelA.createLayeredComposite();
    try {
      composite.setBaseWeight(0.7);
      composite.addLayer('overlay_b', modelB, 0.3);

      const context = [vocab.tokenToId.get('apple')!];
      const mass = buildDistribution(composite, context, vocab, SAMPLER);

      expectDistribution(mass, { A: 70, B: 30 });
    } finally {
      composite.destroy();
    }
  });

  it('supports three-layer blends', () => {
    const composite = modelA.createLayeredComposite();
    try {
      composite.setBaseWeight(0.5);
      composite.addLayer('overlay_b', modelB, 0.3);
      composite.addLayer('overlay_c', modelC, 0.2);

      const context = [vocab.tokenToId.get('apple')!];
      const mass = buildDistribution(composite, context, vocab, SAMPLER);

      const total = Array.from(mass.values()).reduce((acc, value) => acc + value, 0);
      expect(Math.abs(total - 100)).toBeLessThan(1e-3);
      const aShare = mass.get('A') ?? 0;
      const bShare = mass.get('B') ?? 0;
      const cShare = mass.get('C') ?? 0;
      expect(aShare).toBeGreaterThan(bShare);
      expect(bShare).toBeGreaterThan(cShare);
      expect(bShare).toBeGreaterThanOrEqual(10);
      expect(cShare).toBeGreaterThanOrEqual(10);
    } finally {
      composite.destroy();
    }
  });

  it('preserves markov transition patterns', async () => {
    // Test 5: Verify that learned transition patterns are preserved in blends
    const testVocab = buildVocab('A', 'B');
    const length = 512;

    // Layer A: Strong apple→ant→arrow cycle
    const patternA = Array(length).fill(0).map((_, i) => i % 3); // [0,1,2,0,1,2,...]
    const seqA = encode(patternA.map(i => VOCAB_SETS.A[i]), testVocab.tokenToId);

    // Layer B: Strong ball→bat→bear cycle
    const patternB = Array(length).fill(0).map((_, i) => i % 3); // [0,1,2,0,1,2,...]
    const seqB = encode(patternB.map(i => VOCAB_SETS.B[i]), testVocab.tokenToId);

    const testModelA = await trainModel(seqA, testVocab.tokenToId.size);
    const testModelB = await trainModel(seqB, testVocab.tokenToId.size);

    try {
      // Test pure A - should show strong apple→ant pattern
      const compositePureA = testModelA.createLayeredComposite();
      try {
        compositePureA.setBaseWeight(1.0);
        compositePureA.addLayer('overlay_b', testModelB, 0.0);

        const context = [testVocab.tokenToId.get('apple')!];
        const result = compositePureA.predict(context, 8, SAMPLER);

        // Find "ant" in predictions (should be top prediction for apple)
        const antId = testVocab.tokenToId.get('ant')!;
        const antIdx = result.ids.indexOf(antId);

        expect(antIdx).not.toBe(-1); // ant should appear
        expect(antIdx).toBeLessThanOrEqual(2); // should be in top 3
      } finally {
        compositePureA.destroy();
      }

      // Test blended - should show BOTH patterns
      const compositeBlend = testModelA.createLayeredComposite();
      try {
        compositeBlend.setBaseWeight(0.5);
        compositeBlend.addLayer('overlay_b', testModelB, 0.5);

        const context = [testVocab.tokenToId.get('apple')!];
        const result = compositeBlend.predict(context, 16, SAMPLER);

        // Both A-pattern and B-pattern tokens should appear
        const predictedTokens = result.ids.map(id => testVocab.idToToken.get(id)!);
        const aTokens = predictedTokens.filter(t => VOCAB_SETS.A.includes(t));
        const bTokens = predictedTokens.filter(t => VOCAB_SETS.B.includes(t));

        expect(aTokens.length).toBeGreaterThan(0);
        expect(bTokens.length).toBeGreaterThan(0);

        // Verify that learned patterns are still visible
        const antId = testVocab.tokenToId.get('ant')!;
        if (result.ids.includes(antId)) {
          const antPosition = result.ids.indexOf(antId);
          expect(antPosition).toBeLessThan(result.ids.length / 2); // should rank in top half
        }
      } finally {
        compositeBlend.destroy();
      }
    } finally {
      testModelB.destroy();
      testModelA.destroy();
    }
  });

  it('exhibits smooth weight sweep linearity', async () => {
    // Test smooth linear transition of probability mass across weight sweep
    const testVocab = buildVocab('A', 'B');
    const length = 512;

    const seqA = encode(repeatingSequence(VOCAB_SETS.A, [0, 1, 2, 0, 1, 2], length), testVocab.tokenToId);
    const seqB = encode(repeatingSequence(VOCAB_SETS.B, [0, 2, 1, 0, 2, 1], length), testVocab.tokenToId);

    const testModelA = await trainModel(seqA, testVocab.tokenToId.size);
    const testModelB = await trainModel(seqB, testVocab.tokenToId.size);

    try {
      const weights = [1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0];
      const aPercentages: number[] = [];

      for (const weightA of weights) {
        const weightB = 1.0 - weightA;
        const composite = testModelA.createLayeredComposite();
        try {
          composite.setBaseWeight(weightA);
          composite.addLayer('overlay_b', testModelB, weightB);

          const context = [testVocab.tokenToId.get('apple')!];
          const mass = buildDistribution(composite, context, testVocab, SAMPLER);

          const aPercent = mass.get('A') ?? 0;
          aPercentages.push(aPercent);

          // Each weight should produce distribution close to target (±20%)
          const expected = weightA * 100;
          expect(Math.abs(aPercent - expected)).toBeLessThanOrEqual(20);
        } finally {
          composite.destroy();
        }
      }

      // Verify monotonic decrease (linearity check)
      for (let i = 0; i < aPercentages.length - 1; i++) {
        // Allow small tolerance for noise, but trend should be clear
        expect(aPercentages[i]).toBeGreaterThanOrEqual(aPercentages[i + 1] - 5.0);
      }
    } finally {
      testModelB.destroy();
      testModelA.destroy();
    }
  });

  it('eliminates zero-weight layer influence', () => {
    // Test 6: Verify that weight=0 completely eliminates layer influence
    const composite = modelA.createLayeredComposite();
    try {
      composite.setBaseWeight(1.0);
      composite.addLayer('overlay_b', modelB, 0.0);

      const context = [vocab.tokenToId.get('apple')!];
      const mass = buildDistribution(composite, context, vocab, SAMPLER);

      // Should be identical to pure A
      const aShare = mass.get('A') ?? 0;
      const bShare = mass.get('B') ?? 0;
      expect(aShare).toBeGreaterThanOrEqual(80);
      expect(bShare).toBeLessThanOrEqual(15);
    } finally {
      composite.destroy();
    }
  });

  it('produces uniform baseline for random sequences', async () => {
    // Test 8: Baseline test with random sequences
    const testVocab = buildVocab('A', 'B');
    const length = 512;

    // Create truly random sequence (no patterns)
    const allTokens = Array.from(testVocab.tokenToId.keys());
    const randomSeq: number[] = [];
    for (let i = 0; i < length; i++) {
      const randomToken = allTokens[Math.floor(Math.random() * allTokens.length)];
      randomSeq.push(testVocab.tokenToId.get(randomToken)!);
    }

    const testModel = await trainModel(randomSeq, testVocab.tokenToId.size);

    try {
      const composite = testModel.createLayeredComposite();
      try {
        composite.setBaseWeight(1.0);

        const context = [testVocab.tokenToId.get('apple')!];
        const result = composite.predict(context, 16, SAMPLER);

        if (!result.probabilities) {
          throw new Error('Sampler must return probability estimates');
        }

        // Calculate entropy (should be relatively high for uniform distribution)
        // H = -Σ(p * log2(p))
        let entropy = 0;
        for (const p of result.probabilities) {
          if (p > 0) {
            entropy -= p * Math.log2(p);
          }
        }

        // Max entropy for 16 predictions would be log2(16) = 4.0
        // We expect at least moderate entropy (>2.0) for random training
        expect(entropy).toBeGreaterThan(2.0);

        // No single token should dominate
        const maxProb = Math.max(...result.probabilities);
        expect(maxProb).toBeLessThan(0.5);
      } finally {
        composite.destroy();
      }
    } finally {
      testModel.destroy();
    }
  });
});

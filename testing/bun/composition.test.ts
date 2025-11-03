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
});

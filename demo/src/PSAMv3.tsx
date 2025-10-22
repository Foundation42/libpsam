import React, { useState, useMemo, useEffect } from 'react';
import { Play, Pause, RotateCcw, Brain, Settings } from 'lucide-react';

const PSAMv3 = () => {
  const [text, setText] = useState("the cat sat on the mat. the dog sat on the rug. the bird sat on the branch. the frog sat on the log.");
  const [trainingStep, setTrainingStep] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [inferenceInput, setInferenceInput] = useState("the dog sat on the");
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationHistory, setGenerationHistory] = useState<any[]>([]);

  // Enhanced parameters
  const [contextWindow, setContextWindow] = useState(8);
  const [topK, setTopK] = useState(32);
  const [minEvidence, setMinEvidence] = useState(1);
  const [idfEnabled, setIdfEnabled] = useState(true);
  const [ppmiEnabled, setPpmiEnabled] = useState(true);
  const [distanceDecay, setDistanceDecay] = useState(0.1);
  const [recencyDecay, setRecencyDecay] = useState(0.05);
  const [edgeDropout, setEdgeDropout] = useState(0.15);
  const [temperature, setTemperature] = useState(1.0);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Tokenize text
  const tokens = useMemo(() => {
    return text.toLowerCase().match(/\w+|[.,!?;]/g) || [];
  }, [text]);

  // Calculate token frequencies for IDF and PPMI
  const tokenStats = useMemo(() => {
    const freq = new Map<string, number>();
    const pairFreq = new Map<string, number>();

    tokens.forEach(token => {
      freq.set(token, (freq.get(token) || 0) + 1);
    });

    // Count co-occurrences for PPMI
    for (let i = 0; i < tokens.length; i++) {
      const source = tokens[i];
      for (let offset = -contextWindow; offset <= contextWindow; offset++) {
        if (offset === 0) continue;
        const targetIdx = i + offset;
        if (targetIdx < 0 || targetIdx >= tokens.length) continue;

        const target = tokens[targetIdx];
        const key = `${source}|${offset}|${target}`;
        pairFreq.set(key, (pairFreq.get(key) || 0) + 1);
      }
    }

    return { tokenFreq: freq, pairFreq, totalTokens: tokens.length };
  }, [tokens, contextWindow]);

  // Calculate IDF weights
  const idfWeight = useMemo(() => {
    const weights = new Map<string, number>();
    const N = tokens.length;
    tokenStats.tokenFreq.forEach((count, token) => {
      const idf = Math.log(N / count);
      weights.set(token, Math.max(0.1, Math.min(1.0, idf / 5)));
    });
    return weights;
  }, [tokenStats, tokens.length]);

  // Distance decay kernel
  const distanceKernel = (delta: number) => {
    return Math.exp(-distanceDecay * Math.abs(delta));
  };

  // Build association graph with PPMI and recency decay
  const { graph, trainingHistory } = useMemo(() => {
    const graph = new Map<string, any[]>();
    const history: any[] = [];
    const evidenceCount = new Map<string, number>();

    for (let step = 0; step <= Math.min(trainingStep, tokens.length - 1); step++) {
      const token = tokens[step];

      if (!graph.has(token)) {
        graph.set(token, []);
      }

      const associations = graph.get(token)!;

      // Apply recency decay to existing edges
      associations.forEach(assoc => {
        assoc.weight *= (1 - recencyDecay);
        assoc.age = (assoc.age || 0) + 1;
      });

      for (let offset = -contextWindow; offset <= contextWindow; offset++) {
        if (offset === 0) continue;

        const targetIdx = step + offset;
        if (targetIdx < 0 || targetIdx >= tokens.length) continue;

        const targetToken = tokens[targetIdx];

        // Track evidence
        const evidenceKey = `${token}|${offset}|${targetToken}`;
        const currentEvidence = (evidenceCount.get(evidenceKey) || 0) + 1;
        evidenceCount.set(evidenceKey, currentEvidence);

        if (currentEvidence < minEvidence) continue;

        // Edge dropout
        if (edgeDropout > 0 && Math.random() < edgeDropout) continue;

        let assoc = associations.find(a =>
          a.targetToken === targetToken && a.relativePos === offset
        );

        if (assoc) {
          // Calculate PPMI-based weight increment
          let increment = 0.1;

          if (ppmiEnabled) {
            const pxy = currentEvidence / tokenStats.totalTokens;
            const px = (tokenStats.tokenFreq.get(token) || 1) / tokenStats.totalTokens;
            const py = (tokenStats.tokenFreq.get(targetToken) || 1) / tokenStats.totalTokens;
            const pmi = Math.log(pxy / (px * py));
            const ppmi = Math.max(0, pmi);
            increment = 0.1 + ppmi * 0.5;
          }

          assoc.weight += increment;
          assoc.count += 1;
          assoc.age = 0;
        } else {
          // New association
          let initialWeight = 1.0;

          if (ppmiEnabled && currentEvidence >= minEvidence) {
            const pxy = currentEvidence / tokenStats.totalTokens;
            const px = (tokenStats.tokenFreq.get(token) || 1) / tokenStats.totalTokens;
            const py = (tokenStats.tokenFreq.get(targetToken) || 1) / tokenStats.totalTokens;
            const pmi = Math.log(pxy / (px * py));
            const ppmi = Math.max(0, pmi);
            initialWeight = 1.0 + ppmi;
          }

          associations.push({
            targetToken,
            relativePos: offset,
            weight: initialWeight,
            count: 1,
            age: 0
          });
        }
      }

      history.push({
        step,
        token,
        associationsAdded: associations.length
      });
    }

    return { graph, trainingHistory: history };
  }, [tokens, trainingStep, contextWindow, minEvidence, edgeDropout, ppmiEnabled, recencyDecay, tokenStats]);

  // Apply top-K pruning per source token
  const prunedGraph = useMemo(() => {
    const pruned = new Map<string, any[]>();

    graph.forEach((associations, token) => {
      const byPosition = new Map<number, any[]>();
      associations.forEach(assoc => {
        if (!byPosition.has(assoc.relativePos)) {
          byPosition.set(assoc.relativePos, []);
        }
        byPosition.get(assoc.relativePos)!.push(assoc);
      });

      const kept: any[] = [];
      byPosition.forEach((assocs, pos) => {
        const sorted = assocs.sort((a, b) => b.weight - a.weight);
        kept.push(...sorted.slice(0, topK));
      });

      if (kept.length > 0) {
        pruned.set(token, kept);
      }
    });

    return pruned;
  }, [graph, topK]);

  // Enhanced inference with IDF, distance decay, and calibration
  const inference = useMemo(() => {
    const inputTokens = inferenceInput.toLowerCase().match(/\w+|[.,!?;]/g) || [];
    const votes = new Map<string, number>();
    const activations: any[] = [];

    inputTokens.forEach((token, idx) => {
      const associations = prunedGraph.get(token) || [];
      const sourceIdf = idfEnabled ? (idfWeight.get(token) || 0.5) : 1.0;

      associations.forEach(assoc => {
        const predictedPos = idx + assoc.relativePos;

        if (predictedPos === inputTokens.length) {
          const distWeight = distanceKernel(assoc.relativePos);
          const score = sourceIdf * distWeight * assoc.weight;

          const currentVote = votes.get(assoc.targetToken) || 0;
          votes.set(assoc.targetToken, currentVote + score);

          activations.push({
            sourceToken: token,
            sourcePos: idx,
            targetToken: assoc.targetToken,
            relativePos: assoc.relativePos,
            weight: assoc.weight,
            idf: sourceIdf,
            distance: distWeight,
            effectiveScore: score
          });
        }
      });
    });

    const predictions = Array.from(votes.entries())
      .map(([token, score]) => ({ token, score, probability: 0 }))
      .sort((a, b) => b.score - a.score);

    // Apply temperature and softmax
    if (predictions.length > 0) {
      const expScores = predictions.map(p => Math.exp(p.score / temperature));
      const sumExp = expScores.reduce((a, b) => a + b, 0);
      predictions.forEach((p, i) => {
        p.probability = expScores[i] / sumExp;
      });
    }

    const confidence = predictions.length >= 2
      ? predictions[0].score / predictions[1].score
      : predictions.length === 1 ? Infinity : 0;

    return {
      predictions: predictions.slice(0, 5),
      activations,
      confidence
    };
  }, [inferenceInput, prunedGraph, idfWeight, idfEnabled, distanceKernel, temperature]);

  const handleTrain = () => {
    if (trainingStep < tokens.length - 1) {
      setTrainingStep(s => s + 1);
    } else {
      setIsTraining(false);
    }
  };

  const handleAutoTrain = () => {
    setIsTraining(!isTraining);
  };

  useEffect(() => {
    if (isTraining) {
      const interval = setInterval(() => {
        setTrainingStep(s => {
          if (s >= tokens.length - 1) {
            setIsTraining(false);
            return s;
          }
          return s + 1;
        });
      }, 50);
      return () => clearInterval(interval);
    }
  }, [isTraining, tokens.length]);

  const handleReset = () => {
    setTrainingStep(0);
    setIsTraining(false);
    setIsGenerating(false);
    setGenerationHistory([]);
  };

  const handleGenerate = () => {
    if (inference.predictions.length === 0) return;

    const topPrediction = inference.predictions[0];
    const newInput = inferenceInput + ' ' + topPrediction.token;

    setGenerationHistory(prev => [...prev, {
      input: inferenceInput,
      predicted: topPrediction.token,
      score: topPrediction.score,
      probability: topPrediction.probability,
      confidence: inference.confidence,
      alternatives: inference.predictions.slice(1, 3)
    }]);

    setInferenceInput(newInput);
  };

  const handleAutoGenerate = () => {
    setIsGenerating(!isGenerating);
  };

  useEffect(() => {
    if (isGenerating && inference.predictions.length > 0) {
      const interval = setInterval(() => {
        const topPrediction = inference.predictions[0];
        const newInput = inferenceInput + ' ' + topPrediction.token;

        setGenerationHistory(prev => [...prev, {
          input: inferenceInput,
          predicted: topPrediction.token,
          score: topPrediction.score,
          probability: topPrediction.probability,
          confidence: inference.confidence,
          alternatives: inference.predictions.slice(1, 3)
        }]);

        setInferenceInput(newInput);

        if (topPrediction.token === '.' || inferenceInput.split(' ').length > 20) {
          setIsGenerating(false);
        }
      }, 500);
      return () => clearInterval(interval);
    }
  }, [isGenerating, inference.predictions, inference.confidence, inferenceInput]);

  const totalAssociations = Array.from(prunedGraph.values())
    .reduce((sum, arr) => sum + arr.length, 0);

  const testScenarios = [
    "the cat sat on the",
    "the dog sat on the",
    "the bird sat on the",
    "the frog sat on the"
  ];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg">
      <div className="flex items-center gap-3 mb-6">
        <Brain className="w-8 h-8 text-indigo-600" />
        <div>
          <h1 className="text-3xl font-bold text-gray-800">PSAM Interactive Demo</h1>
          <p className="text-sm text-gray-600">Position-Specific Association Memory with PPMI + IDF</p>
        </div>
      </div>

      {/* Training Section */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-700">Training Data</h2>
        <textarea
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setTrainingStep(0);
            setGenerationHistory([]);
          }}
          className="w-full p-3 border border-gray-300 rounded-lg mb-4 font-mono text-sm"
          rows={3}
        />

        <div className="flex gap-4 mb-4 items-center flex-wrap">
          <button
            onClick={handleAutoTrain}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition"
          >
            {isTraining ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isTraining ? 'Pause' : 'Auto Train'}
          </button>

          <button
            onClick={handleReset}
            className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>

          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition"
          >
            <Settings className="w-4 h-4" />
            {showAdvanced ? 'Hide' : 'Show'} Parameters
          </button>
        </div>

        {/* Advanced Parameters */}
        {showAdvanced && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-gray-50 rounded-lg mb-4">
            <div>
              <label className="text-xs font-medium text-gray-700 block mb-1">Context Window</label>
              <input
                type="number"
                value={contextWindow}
                onChange={(e) => setContextWindow(Math.max(1, parseInt(e.target.value) || 1))}
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                min="1"
                max="20"
              />
            </div>

            <div>
              <label className="text-xs font-medium text-gray-700 block mb-1">Top-K Pruning</label>
              <input
                type="number"
                value={topK}
                onChange={(e) => setTopK(Math.max(1, parseInt(e.target.value) || 1))}
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                min="1"
                max="128"
              />
            </div>

            <div>
              <label className="text-xs font-medium text-gray-700 block mb-1">Min Evidence</label>
              <input
                type="number"
                value={minEvidence}
                onChange={(e) => setMinEvidence(Math.max(1, parseInt(e.target.value) || 1))}
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                min="1"
                max="5"
              />
            </div>

            <div>
              <label className="text-xs font-medium text-gray-700 block mb-1">Distance Decay (α)</label>
              <input
                type="number"
                value={distanceDecay}
                onChange={(e) => setDistanceDecay(parseFloat(e.target.value))}
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                step="0.05"
                min="0"
                max="1"
              />
            </div>

            <div>
              <label className="text-xs font-medium text-gray-700 block mb-1">Recency Decay (λ)</label>
              <input
                type="number"
                value={recencyDecay}
                onChange={(e) => setRecencyDecay(parseFloat(e.target.value))}
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                step="0.01"
                min="0"
                max="0.5"
              />
            </div>

            <div>
              <label className="text-xs font-medium text-gray-700 block mb-1">Edge Dropout</label>
              <input
                type="number"
                value={edgeDropout}
                onChange={(e) => setEdgeDropout(parseFloat(e.target.value))}
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                step="0.05"
                min="0"
                max="0.5"
              />
            </div>

            <div>
              <label className="text-xs font-medium text-gray-700 block mb-1">Temperature</label>
              <input
                type="number"
                value={temperature}
                onChange={(e) => setTemperature(Math.max(0.1, parseFloat(e.target.value)))}
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                step="0.1"
                min="0.1"
                max="2"
              />
            </div>

            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={idfEnabled}
                  onChange={(e) => setIdfEnabled(e.target.checked)}
                  className="rounded"
                />
                <label className="text-xs font-medium text-gray-700">Enable IDF</label>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={ppmiEnabled}
                  onChange={(e) => setPpmiEnabled(e.target.checked)}
                  className="rounded"
                />
                <label className="text-xs font-medium text-gray-700">Enable PPMI</label>
              </div>
            </div>
          </div>
        )}

        <div className="text-sm text-gray-600 mb-2">
          Progress: {trainingStep} / {tokens.length - 1} |
          Associations: {totalAssociations} |
          Tokens: {prunedGraph.size} |
          Confidence: {inference.confidence.toFixed(2)}x
        </div>

        <div className="font-mono text-sm p-3 bg-gray-50 rounded border border-gray-200 overflow-x-auto">
          {tokens.map((token, idx) => (
            <span
              key={idx}
              className={`mr-1 px-1 rounded ${
                idx === trainingStep
                  ? 'bg-indigo-600 text-white'
                  : idx < trainingStep
                  ? 'bg-green-200'
                  : 'bg-gray-200'
              }`}
            >
              {token}
            </span>
          ))}
        </div>
      </div>

      {/* Quick Test Scenarios */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-700">Quick Tests</h2>
        <div className="grid grid-cols-2 gap-2">
          {testScenarios.map((scenario, idx) => (
            <button
              key={idx}
              onClick={() => {
                setInferenceInput(scenario);
                setGenerationHistory([]);
                setIsGenerating(false);
              }}
              className="px-3 py-2 bg-blue-100 hover:bg-blue-200 rounded text-sm font-mono text-left transition"
            >
              {scenario}
            </button>
          ))}
        </div>
      </div>

      {/* Inference Section */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-700">Inference</h2>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Input context:
          </label>
          <input
            type="text"
            value={inferenceInput}
            onChange={(e) => {
              setInferenceInput(e.target.value);
              setGenerationHistory([]);
              setIsGenerating(false);
            }}
            className="w-full p-3 border border-gray-300 rounded-lg font-mono text-sm"
            placeholder="Enter text to predict next token..."
          />
        </div>

        <div className="flex gap-2 mb-4">
          <button
            onClick={handleGenerate}
            disabled={inference.predictions.length === 0}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition disabled:bg-gray-400"
          >
            Generate Next
          </button>

          <button
            onClick={handleAutoGenerate}
            disabled={inference.predictions.length === 0}
            className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition disabled:bg-gray-400"
          >
            {isGenerating ? 'Stop Auto' : 'Auto Generate'}
          </button>

          <button
            onClick={() => {
              setInferenceInput(testScenarios[0]);
              setGenerationHistory([]);
              setIsGenerating(false);
            }}
            className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition"
          >
            Reset Input
          </button>
        </div>

        <div className="mb-4">
          <h3 className="font-semibold text-gray-700 mb-2">Active Associations (PPMI + Decay):</h3>
          <div className="p-3 bg-gray-50 rounded border border-gray-200 max-h-40 overflow-y-auto text-xs">
            {inference.activations.length === 0 ? (
              <div className="text-gray-500">No associations activated</div>
            ) : (
              inference.activations.map((act: any, idx: number) => (
                <div key={idx} className="mb-1 font-mono">
                  <span className="text-indigo-600">"{act.sourceToken}"</span> @{act.sourcePos}
                  → <span className="text-green-600">"{act.targetToken}"</span>
                  <span className="text-gray-500 ml-2">
                    (w:{act.weight.toFixed(2)} × idf:{act.idf.toFixed(2)} × d:{act.distance.toFixed(2)}
                    = <strong>{act.effectiveScore.toFixed(3)}</strong>)
                  </span>
                </div>
              ))
            )}
          </div>
        </div>

        <div>
          <h3 className="font-semibold text-gray-700 mb-2">Top Predictions (Calibrated):</h3>
          <div className="space-y-2">
            {inference.predictions.length === 0 ? (
              <div className="text-gray-500 text-sm">No predictions available</div>
            ) : (
              inference.predictions.map((pred: any, idx: number) => (
                <div key={idx} className="flex items-center gap-3">
                  <div className="w-32 text-right font-mono text-xs">
                    <div>{pred.score.toFixed(3)}</div>
                    <div className="text-gray-500">({(pred.probability * 100).toFixed(1)}%)</div>
                  </div>
                  <div className="flex-1 bg-gray-200 rounded-full h-8 overflow-hidden">
                    <div
                      className="bg-indigo-600 h-full flex items-center px-3 text-white text-sm font-semibold"
                      style={{ width: `${pred.probability * 100}%` }}
                    >
                      "{pred.token}"
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Generation History */}
      {generationHistory.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4 text-gray-700">Generation History</h2>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {generationHistory.map((item, idx) => (
              <div key={idx} className="p-2 bg-gray-50 rounded border border-gray-200 text-sm">
                <div className="font-mono mb-1">
                  <span className="text-gray-600">{item.input}</span>
                  <span className="text-green-600 font-bold"> {item.predicted}</span>
                </div>
                <div className="text-xs text-gray-500">
                  Score: {item.score.toFixed(3)} |
                  Prob: {(item.probability * 100).toFixed(1)}% |
                  Conf: {item.confidence.toFixed(2)}x
                  {item.alternatives.length > 0 && (
                    <span className="ml-2">
                      | Alt: {item.alternatives.map((a: any) => `"${a.token}" (${(a.probability * 100).toFixed(0)}%)`).join(', ')}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="mt-6 text-center text-sm text-gray-600">
        <p>
          Learn more about PSAM at{' '}
          <a
            href="https://github.com/Foundation42/libpsam"
            target="_blank"
            rel="noopener noreferrer"
            className="text-indigo-600 hover:underline"
          >
            github.com/Foundation42/libpsam
          </a>
        </p>
      </div>
    </div>
  );
};

export default PSAMv3;

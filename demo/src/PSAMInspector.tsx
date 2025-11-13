import { useState, useCallback, useEffect } from 'react';
import { Upload, FileText, Network, BarChart3, Sparkles, Play } from 'lucide-react';
import PSAMRailwayViewer from './components/PSAMRailwayViewer';
import { initInspectorWASM, PSAMInspectorWASM } from './lib/psam-inspector-wasm';

interface Vocab {
  [tokenId: number]: string;
}

interface LoadedPSAM {
  name: string;
  handle?: number;  // WASM handle (when WASM is available)
  vocab?: Vocab;     // Optional vocabulary mapping
  config?: {
    vocabSize: number;
    window: number;
    topK: number;
    alpha: number;
    minEvidence: number;
    enableIdf: boolean;
    enablePpmi: boolean;
    edgeDropout: number;
  };
  stats?: {
    vocabSize: number;
    rowCount: number;
    edgeCount: number;
    totalTokens: number;
    memoryBytes: number;
  };
  edges?: Array<{
    source: number;
    target: number;
    offset: number;
    weight: number;
    observations: number;
  }>;
  // For demo/testing - use mock data
  mockData?: {
    tokens: Array<{ id: number; text: string; salience: number; residual: number; perplexity: number }>;
    connections: Array<{ from: number; to: number; type: string; strength: number }>;
  };
}

const PSAMInspector = () => {
  const [loadedModels, setLoadedModels] = useState<LoadedPSAM[]>([]);
  const [selectedModel, setSelectedModel] = useState<number | null>(null);
  const [activeView, setActiveView] = useState<'network' | 'stats' | 'predict'>('network');
  const [isDragging, setIsDragging] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [contextInput, setContextInput] = useState('');
  const [generatedText, setGeneratedText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [predictions, setPredictions] = useState<Array<{
    token: number;
    word: string;
    score: number;
    probability: number;
  }>>([]);
  const [generationHistory, setGenerationHistory] = useState<Array<{
    token: number;
    word: string;
    contextTokens: number[];
    explanation: {
      total: number;
      bias: number;
      terms: Array<{
        source: number;
        offset: number;
        weight: number;
        contribution: number;
      }>;
    };
  }>>([]);
  const [wasmInspector, setWasmInspector] = useState<PSAMInspectorWASM | null>(null);
  const [wasmError, setWasmError] = useState<string | null>(null);

  // Initialize WASM
  useEffect(() => {
    const loadWASM = async () => {
      try {
        const inspector = await initInspectorWASM();
        setWasmInspector(inspector);
        console.log('✅ WASM Inspector initialized');
      } catch (err) {
        console.warn('⚠️ WASM Inspector not available, using mock data:', err);
        setWasmError(err instanceof Error ? err.message : 'Failed to load WASM');
      }
    };

    loadWASM();
  }, []);

  const loadMockData = useCallback((fileName: string): LoadedPSAM => {
    // Generate mock data for demonstration (until WASM is built)
    const tokens = [
      { id: 0, text: "the", salience: 0.55, residual: 0.42, perplexity: 0.31 },
      { id: 1, text: "quick", salience: 0.71, residual: 0.59, perplexity: 0.38 },
      { id: 2, text: "brown", salience: 0.68, residual: 0.55, perplexity: 0.34 },
      { id: 3, text: "fox", salience: 0.85, residual: 0.73, perplexity: 0.48 },
      { id: 4, text: "jumps", salience: 0.92, residual: 0.81, perplexity: 0.65 },
      { id: 5, text: "over", salience: 0.64, residual: 0.49, perplexity: 0.27 },
      { id: 6, text: "the", salience: 0.58, residual: 0.44, perplexity: 0.25 },
      { id: 7, text: "lazy", salience: 0.76, residual: 0.62, perplexity: 0.42 },
      { id: 8, text: "dog", salience: 0.88, residual: 0.75, perplexity: 0.54 },
      { id: 9, text: "sleeping", salience: 0.82, residual: 0.69, perplexity: 0.51 },
      { id: 10, text: "under", salience: 0.67, residual: 0.52, perplexity: 0.33 },
      { id: 11, text: "a", salience: 0.52, residual: 0.38, perplexity: 0.22 },
      { id: 12, text: "tree", salience: 0.79, residual: 0.66, perplexity: 0.46 }
    ];

    const connections = [
      { from: 1, to: 3, type: "residual", strength: 0.72 },
      { from: 2, to: 3, type: "residual", strength: 0.68 },
      { from: 3, to: 4, type: "dominant", strength: 0.89 },
      { from: 4, to: 8, type: "dominant", strength: 0.85 },
      { from: 5, to: 8, type: "residual", strength: 0.58 },
      { from: 7, to: 8, type: "residual", strength: 0.75 },
      { from: 8, to: 9, type: "dominant", strength: 0.81 },
      { from: 9, to: 12, type: "residual", strength: 0.64 },
      { from: 10, to: 12, type: "residual", strength: 0.71 },
      { from: 3, to: 8, type: "residual", strength: 0.52 },
      { from: 1, to: 7, type: "residual", strength: 0.46 },
      { from: 4, to: 9, type: "perplexity", strength: 0.67 }
    ];

    return {
      name: fileName,
      config: {
        vocabSize: tokens.length,
        window: 8,
        topK: 32,
        alpha: 0.1,
        minEvidence: 1,
        enableIdf: true,
        enablePpmi: true,
        edgeDropout: 0.0
      },
      stats: {
        vocabSize: tokens.length,
        rowCount: tokens.length * 8,
        edgeCount: connections.length,
        totalTokens: 1000,
        memoryBytes: 50000
      },
      mockData: { tokens, connections }
    };
  }, []);

  const loadVocabFile = useCallback(async (vocabFile: File): Promise<Vocab | null> => {
    try {
      const text = await vocabFile.text();
      const vocab: Vocab = {};

      text.split('\n').forEach(line => {
        if (!line.trim()) return;
        const [idStr, word] = line.split('\t');
        if (idStr && word) {
          vocab[parseInt(idStr)] = word;
        }
      });

      console.log(`✅ Loaded vocab with ${Object.keys(vocab).length} entries`);
      return vocab;
    } catch (err) {
      console.error('Failed to load vocab:', err);
      return null;
    }
  }, []);

  const handleFileSelect = useCallback(async (files: FileList) => {
    // Look for .psam and .tsv files
    const psamFiles: File[] = [];
    const vocabFiles: File[] = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (file.name.endsWith('.psam')) {
        psamFiles.push(file);
      } else if (file.name.endsWith('.tsv') || file.name.endsWith('_vocab.tsv')) {
        vocabFiles.push(file);
      }
    }

    for (const file of psamFiles) {

      try {
        console.log(`Loading ${file.name} (${file.size} bytes)`);

        if (wasmInspector) {
          // Use real WASM
          const arrayBuffer = await file.arrayBuffer();
          const uint8Array = new Uint8Array(arrayBuffer);

          const handle = wasmInspector.loadFromMemory(uint8Array);
          const config = wasmInspector.getConfig(handle);
          const stats = wasmInspector.getStats(handle);

          // Debug model state
          wasmInspector.debugModel(handle);

          // Extract edges for visualization
          const edges = wasmInspector.getEdges(handle, 0xFFFFFFFF, 0.0, 500);

          console.log(`✅ Loaded ${file.name}:`, { config, stats, edgeCount: edges.length });

          // Try to find matching vocab file
          const baseName = file.name.replace('.psam', '');
          const vocabFile = vocabFiles.find(f =>
            f.name === `${baseName}_vocab.tsv` ||
            f.name === `${baseName}.tsv`
          );

          let vocab: Vocab | null = null;
          if (vocabFile) {
            vocab = await loadVocabFile(vocabFile);
          } else {
            console.warn(`No vocab file found for ${file.name} (looking for ${baseName}_vocab.tsv or ${baseName}.tsv)`);
          }

          // Map edges to visualization format
          // Build a map of all unique tokens from edges
          const tokenMap = new Map<number, { count: number; maxWeight: number }>();
          edges.forEach(edge => {
            if (!tokenMap.has(edge.source)) {
              tokenMap.set(edge.source, { count: 0, maxWeight: 0 });
            }
            if (!tokenMap.has(edge.target)) {
              tokenMap.set(edge.target, { count: 0, maxWeight: 0 });
            }
            const sourceData = tokenMap.get(edge.source)!;
            const targetData = tokenMap.get(edge.target)!;
            sourceData.count++;
            targetData.count++;
            sourceData.maxWeight = Math.max(sourceData.maxWeight, Math.abs(edge.weight));
            targetData.maxWeight = Math.max(targetData.maxWeight, Math.abs(edge.weight));
          });

          // Create tokens array with the actual token IDs used in edges
          const tokens = Array.from(tokenMap.entries()).map(([id, data]) => ({
            id,
            text: vocab?.[id] || `token_${id}`, // Use vocab if available
            salience: Math.min(0.9, data.count / 10), // Based on connection count
            residual: data.maxWeight,
            perplexity: 0.5
          }));

          console.log(`Created ${tokens.length} tokens from ${edges.length} edges`);
          console.log('Token ID range:', Math.min(...tokens.map(t => t.id)), '-', Math.max(...tokens.map(t => t.id)));

          // Create a lookup to map token IDs to array indices
          const tokenIdToIndex = new Map<number, number>();
          tokens.forEach((token, idx) => {
            tokenIdToIndex.set(token.id, idx);
          });

          // Only include connections where both source and target exist in our tokens
          const connections = edges
            .filter(edge => tokenIdToIndex.has(edge.source) && tokenIdToIndex.has(edge.target))
            .map(edge => ({
              from: tokenIdToIndex.get(edge.source)!,
              to: tokenIdToIndex.get(edge.target)!,
              type: Math.abs(edge.weight) > 0.7 ? 'dominant' : 'residual',
              strength: Math.abs(edge.weight)
            }));

          const model: LoadedPSAM = {
            name: file.name,
            handle,
            vocab: vocab || undefined,
            config,
            stats,
            edges,
            mockData: { tokens, connections }
          };

          setLoadedModels(prev => [...prev, model]);
        } else {
          // Fall back to mock data
          console.log(`⚠️ WASM not available, using mock data for ${file.name}`);
          const mockModel = loadMockData(file.name);
          setLoadedModels(prev => [...prev, mockModel]);
        }

        // Select first model automatically
        if (loadedModels.length === 0) {
          setSelectedModel(0);
        }
      } catch (err) {
        console.error(`Failed to load ${file.name}:`, err);
        alert(`Failed to load ${file.name}: ${err instanceof Error ? err.message : 'Unknown error'}`);
      }
    }
  }, [wasmInspector, loadMockData, loadVocabFile, loadedModels.length]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files) {
      handleFileSelect(e.dataTransfer.files);
    }
  }, [handleFileSelect]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const currentModel = selectedModel !== null ? loadedModels[selectedModel] : null;

  // Helper: Convert text to token IDs using vocab
  const textToTokenIds = useCallback((text: string, vocab: Vocab | undefined): number[] | null => {
    if (!vocab) return null;

    // Create reverse vocab lookup
    const wordToId = new Map<string, number>();
    Object.entries(vocab).forEach(([id, word]) => {
      wordToId.set(word.toLowerCase(), parseInt(id));
    });

    // Tokenize by splitting on whitespace and punctuation
    const words = text.trim().split(/\s+/);
    const tokenIds: number[] = [];

    for (const word of words) {
      const cleanWord = word.toLowerCase();
      const id = wordToId.get(cleanWord);
      if (id !== undefined) {
        tokenIds.push(id);
      }
    }

    return tokenIds.length > 0 ? tokenIds : null;
  }, []);

  // Handle prediction
  const handlePredict = useCallback(async () => {
    if (!currentModel || !wasmInspector || !currentModel.handle) return;

    const tokenIds = textToTokenIds(contextInput, currentModel.vocab);
    if (!tokenIds) {
      alert('Could not find tokens in vocabulary. Try simpler words from the text.');
      return;
    }

    try {
      const preds = wasmInspector.predict(currentModel.handle, tokenIds, 20);

      const predictions = preds.map(p => ({
        token: p.token,
        word: currentModel.vocab?.[p.token] || `token_${p.token}`,
        score: p.score,
        probability: p.probability
      }));

      setPredictions(predictions);
      console.log(`Got ${predictions.length} predictions for context:`, tokenIds);
    } catch (err) {
      console.error('Prediction failed:', err);
      alert(`Prediction failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  }, [currentModel, wasmInspector, contextInput, textToTokenIds]);

  // Auto-generate text
  const handleGenerate = useCallback(async () => {
    if (!currentModel || !wasmInspector || !currentModel.handle || isGenerating) return;

    // Use existing generated text if available, otherwise start with context input
    const startingText = generatedText || contextInput;
    const tokenIds = textToTokenIds(startingText, currentModel.vocab);
    if (!tokenIds) {
      alert('Could not find tokens in vocabulary. Try simpler words.');
      return;
    }

    setIsGenerating(true);

    // Only set generated text if it's empty
    if (!generatedText) {
      setGeneratedText(contextInput);
    }

    try {
      const context = [...tokenIds];
      const maxTokens = 50;
      const history: typeof generationHistory = [];

      for (let i = 0; i < maxTokens; i++) {
        const preds = wasmInspector.predict(currentModel.handle, context, 20);
        if (preds.length === 0) break;

        // Sample from top predictions (simple: pick highest)
        const nextToken = preds[0].token;
        const nextWord = currentModel.vocab?.[nextToken] || `<${nextToken}>`;

        // Get explanation for this token
        try {
          const explanation = wasmInspector.explain(currentModel.handle, context, nextToken, 10);
          history.push({
            token: nextToken,
            word: nextWord,
            contextTokens: [...context],
            explanation: {
              total: explanation.total,
              bias: explanation.bias,
              terms: explanation.terms.map(t => ({
                source: t.source,
                offset: t.offset,
                weight: t.weight,
                contribution: t.contribution,
              })),
            },
          });
        } catch (explainErr) {
          console.warn('Failed to get explanation:', explainErr);
        }

        context.push(nextToken);
        setGeneratedText(prev => prev + ' ' + nextWord);

        // Keep context window size reasonable
        if (context.length > (currentModel.config?.window || 8)) {
          context.shift();
        }

        // Small delay for visual effect
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      // Update generation history
      setGenerationHistory(prev => [...prev, ...history]);
    } catch (err) {
      console.error('Generation failed:', err);
      alert(`Generation failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsGenerating(false);
    }
  }, [currentModel, wasmInspector, contextInput, generatedText, textToTokenIds, isGenerating, generationHistory]);

  // Track selected token for synchronization
  const [selectedTokenIndex, setSelectedTokenIndex] = useState<number | null>(null);

  // Convert generation history to railway track data
  const buildRailwayData = useCallback(() => {
    if (!generationHistory.length || !currentModel?.vocab) return null;

    // Create tokens array from generation history
    const tokens = generationHistory.map((item, idx) => ({
      id: idx, // Use array index as ID for the railway viewer
      text: item.word,
      salience: 0.8, // Generated tokens are salient
      residual: Math.abs(item.explanation.total),
      perplexity: 0.5,
    }));

    // Build connections from explanation terms
    const connections: Array<{ from: number; to: number; type: string; strength: number }> = [];

    // Find max contribution across all terms for normalization
    let maxContribution = 0;
    generationHistory.forEach(item => {
      item.explanation.terms.forEach(term => {
        maxContribution = Math.max(maxContribution, Math.abs(term.contribution));
      });
    });

    generationHistory.forEach((item, targetIdx) => {
      // For each term that contributed to this token
      item.explanation.terms.forEach(term => {
        // Find the source token in our history
        // Look backwards through context to find matching token
        const contextPos = item.contextTokens.length + term.offset;
        if (contextPos >= 0 && contextPos < item.contextTokens.length) {
          const sourceToken = item.contextTokens[contextPos];

          // Find source token index in our generated history
          const sourceIdx = generationHistory.findIndex((h, idx) =>
            idx < targetIdx && h.token === sourceToken
          );

          if (sourceIdx !== -1) {
            // Normalize by max contribution to get 0-1 range
            const strength = Math.abs(term.contribution) / Math.max(maxContribution, 1);
            connections.push({
              from: sourceIdx,
              to: targetIdx,
              type: term.contribution > 0 ? 'dominant' : 'residual',
              strength: Math.max(0.1, strength), // Ensure minimum visibility
            });
          }
        }
      });
    });

    // Build explanations array indexed by token position
    const explanations = generationHistory.map((item, idx) => ({
      token: idx,
      word: item.word,
      total: item.explanation.total,
      bias: item.explanation.bias,
      terms: item.explanation.terms,
    }));

    return { tokens, connections, explanations, vocab: currentModel.vocab, selectedTokenIndex };
  }, [generationHistory, currentModel?.vocab, selectedTokenIndex]);

  const railwayData = buildRailwayData();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
      <div className="w-full px-4 py-8">
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">PSAM Inspector</h1>
              <p className="text-slate-300">Load and visualize .psam model files</p>
            </div>
            <div className={`px-4 py-2 rounded-lg text-sm font-medium ${
              wasmInspector
                ? 'bg-green-500/20 text-green-300 border border-green-500/30'
                : 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
            }`}>
              {wasmInspector ? '✓ WASM Ready' : '⚠ Mock Data Mode'}
            </div>
          </div>
        </div>

        {/* File Upload Area */}
        {loadedModels.length === 0 && (
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
              isDragging
                ? 'border-indigo-400 bg-indigo-500/10'
                : 'border-slate-600 bg-slate-800/50'
            }`}
          >
            <Upload className="w-16 h-16 mx-auto mb-4 text-slate-400" />
            <h3 className="text-xl font-semibold text-white mb-2">
              Drop .psam files here
            </h3>
            <p className="text-slate-400 mb-4">
              Drop .psam + .tsv vocab files together, or click to browse
            </p>
            <label className="inline-block cursor-pointer">
              <span className="px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg font-medium transition-colors inline-block">
                Select Files
              </span>
              <input
                type="file"
                multiple
                accept=".psam,.tsv"
                onChange={(e) => e.target.files && handleFileSelect(e.target.files)}
                className="hidden"
              />
            </label>
          </div>
        )}

        {/* Loaded Models */}
        {loadedModels.length > 0 && (
          <div className="flex gap-6">
            {/* Model List Sidebar */}
            {!sidebarCollapsed && (
              <div className="w-64 flex-shrink-0">
                <div className="bg-slate-800 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-white">Models</h3>
                    <div className="flex items-center gap-2">
                      <label className="cursor-pointer">
                        <Upload className="w-5 h-5 text-indigo-400 hover:text-indigo-300" />
                        <input
                          type="file"
                          multiple
                          accept=".psam,.tsv"
                          onChange={(e) => e.target.files && handleFileSelect(e.target.files)}
                          className="hidden"
                        />
                      </label>
                      <button
                        onClick={() => setSidebarCollapsed(true)}
                        className="text-slate-400 hover:text-white transition-colors"
                        title="Hide sidebar"
                      >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
                        </svg>
                      </button>
                    </div>
                  </div>

                  <div className="space-y-2">
                    {loadedModels.map((model, idx) => (
                      <button
                        key={idx}
                        onClick={() => setSelectedModel(idx)}
                        className={`w-full text-left p-3 rounded-lg transition-colors ${
                          selectedModel === idx
                            ? 'bg-indigo-600 text-white'
                            : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <FileText className="w-4 h-4 flex-shrink-0" />
                          <span className="truncate text-sm">{model.name}</span>
                        </div>
                        {model.stats && (
                          <div className="text-xs mt-1 opacity-75">
                            {model.stats.edgeCount.toLocaleString()} edges
                          </div>
                        )}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Sidebar Toggle Button (when collapsed) */}
            {sidebarCollapsed && (
              <button
                onClick={() => setSidebarCollapsed(false)}
                className="w-10 flex-shrink-0 bg-slate-800 rounded-lg hover:bg-slate-700 transition-colors flex items-center justify-center"
                title="Show sidebar"
              >
                <svg className="w-5 h-5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                </svg>
              </button>
            )}

            {/* Main Content Area */}
            <div className="flex-1 min-w-0">
              {currentModel && (
                <>
                  {/* View Tabs */}
                  <div className="bg-slate-800 rounded-lg p-1 mb-6 flex gap-2">
                    <button
                      onClick={() => setActiveView('network')}
                      className={`flex-1 py-3 px-4 rounded-md font-medium transition-colors flex items-center justify-center gap-2 ${
                        activeView === 'network'
                          ? 'bg-indigo-600 text-white'
                          : 'text-slate-300 hover:bg-slate-700'
                      }`}
                    >
                      <Network className="w-5 h-5" />
                      Network
                    </button>
                    <button
                      onClick={() => setActiveView('stats')}
                      className={`flex-1 py-3 px-4 rounded-md font-medium transition-colors flex items-center justify-center gap-2 ${
                        activeView === 'stats'
                          ? 'bg-indigo-600 text-white'
                          : 'text-slate-300 hover:bg-slate-700'
                      }`}
                    >
                      <BarChart3 className="w-5 h-5" />
                      Statistics
                    </button>
                    <button
                      onClick={() => setActiveView('predict')}
                      className={`flex-1 py-3 px-4 rounded-md font-medium transition-colors flex items-center justify-center gap-2 ${
                        activeView === 'predict'
                          ? 'bg-indigo-600 text-white'
                          : 'text-slate-300 hover:bg-slate-700'
                      }`}
                    >
                      <Sparkles className="w-5 h-5" />
                      Predict
                    </button>
                  </div>

                  {/* Content */}
                  {activeView === 'network' && currentModel.mockData && (
                    <PSAMRailwayViewer data={currentModel.mockData} />
                  )}

                  {activeView === 'stats' && (
                    <div className="bg-slate-800 rounded-lg p-6">
                      <h3 className="text-xl font-semibold text-white mb-6">Model Statistics</h3>

                      {currentModel.stats && (
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-8">
                          <div className="bg-slate-700 rounded-lg p-4">
                            <div className="text-slate-400 text-sm mb-1">Vocabulary Size</div>
                            <div className="text-2xl font-bold text-white">{currentModel.stats.vocabSize.toLocaleString()}</div>
                          </div>
                          <div className="bg-slate-700 rounded-lg p-4">
                            <div className="text-slate-400 text-sm mb-1">Edge Count</div>
                            <div className="text-2xl font-bold text-white">{currentModel.stats.edgeCount.toLocaleString()}</div>
                          </div>
                          <div className="bg-slate-700 rounded-lg p-4">
                            <div className="text-slate-400 text-sm mb-1">Total Tokens</div>
                            <div className="text-2xl font-bold text-white">{currentModel.stats.totalTokens.toLocaleString()}</div>
                          </div>
                          <div className="bg-slate-700 rounded-lg p-4">
                            <div className="text-slate-400 text-sm mb-1">Row Count</div>
                            <div className="text-2xl font-bold text-white">{currentModel.stats.rowCount.toLocaleString()}</div>
                          </div>
                          <div className="bg-slate-700 rounded-lg p-4">
                            <div className="text-slate-400 text-sm mb-1">Memory</div>
                            <div className="text-2xl font-bold text-white">
                              {(currentModel.stats.memoryBytes / 1024).toFixed(1)} KB
                            </div>
                          </div>
                        </div>
                      )}

                      {currentModel.config && (
                        <>
                          <h4 className="text-lg font-semibold text-white mb-4">Configuration</h4>
                          <div className="bg-slate-700 rounded-lg p-4 font-mono text-sm">
                            <div className="grid grid-cols-2 gap-3">
                              <div className="text-slate-400">Window:</div>
                              <div className="text-white">{currentModel.config.window}</div>

                              <div className="text-slate-400">Top-K:</div>
                              <div className="text-white">{currentModel.config.topK}</div>

                              <div className="text-slate-400">Alpha:</div>
                              <div className="text-white">{currentModel.config.alpha.toFixed(2)}</div>

                              <div className="text-slate-400">Min Evidence:</div>
                              <div className="text-white">{currentModel.config.minEvidence}</div>

                              <div className="text-slate-400">IDF:</div>
                              <div className="text-white">{currentModel.config.enableIdf ? 'Enabled' : 'Disabled'}</div>

                              <div className="text-slate-400">PPMI:</div>
                              <div className="text-white">{currentModel.config.enablePpmi ? 'Enabled' : 'Disabled'}</div>

                              <div className="text-slate-400">Edge Dropout:</div>
                              <div className="text-white">{(currentModel.config.edgeDropout * 100).toFixed(1)}%</div>
                            </div>
                          </div>
                        </>
                      )}
                    </div>
                  )}

                  {activeView === 'predict' && (
                    <div className="bg-slate-800 rounded-lg p-6">
                      <h3 className="text-xl font-semibold text-white mb-6">Generate Text</h3>

                      {!currentModel.vocab && (
                        <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-4 mb-6 text-amber-200">
                          <p className="text-sm">
                            ⚠️ No vocabulary loaded. Drop a .tsv file to enable text generation.
                          </p>
                        </div>
                      )}

                      {currentModel.vocab && (
                        <>
                          {/* Context Input */}
                          <div className="mb-6">
                            <label className="block text-sm font-medium text-slate-300 mb-2">
                              Context (starting words)
                            </label>
                            <input
                              type="text"
                              value={contextInput}
                              onChange={(e) => setContextInput(e.target.value)}
                              placeholder="Enter some words from the vocabulary..."
                              className="w-full px-4 py-3 bg-slate-700 text-white rounded-lg border border-slate-600 focus:border-indigo-500 focus:outline-none"
                              disabled={isGenerating}
                            />
                            <p className="text-xs text-slate-400 mt-1">
                              Enter 2-5 words that appear in the training text
                            </p>
                          </div>

                          {/* Action Buttons */}
                          <div className="flex gap-3 mb-6">
                            <button
                              onClick={handlePredict}
                              disabled={!contextInput.trim() || isGenerating}
                              className="px-6 py-3 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center gap-2"
                            >
                              <Sparkles className="w-5 h-5" />
                              Predict Next
                            </button>
                            <button
                              onClick={handleGenerate}
                              disabled={!contextInput.trim() || isGenerating}
                              className="px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center gap-2"
                            >
                              {isGenerating ? (
                                <>
                                  <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full" />
                                  Generating...
                                </>
                              ) : (
                                <>
                                  <Play className="w-5 h-5" />
                                  Auto-Generate
                                </>
                              )}
                            </button>
                            {generatedText && (
                              <button
                                onClick={() => {
                                  setGeneratedText('');
                                  setPredictions([]);
                                  setGenerationHistory([]);
                                }}
                                disabled={isGenerating}
                                className="px-6 py-3 bg-slate-600 hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
                              >
                                Clear
                              </button>
                            )}
                          </div>

                          {/* Predictions List */}
                          {predictions.length > 0 && (
                            <div className="mb-6">
                              <h4 className="text-lg font-semibold text-white mb-3">Top Predictions</h4>
                              <div className="bg-slate-700 rounded-lg p-4 space-y-2 max-h-64 overflow-y-auto">
                                {predictions.map((pred, idx) => (
                                  <div
                                    key={idx}
                                    className="flex items-center justify-between py-2 px-3 bg-slate-800 rounded"
                                  >
                                    <div className="flex items-center gap-3">
                                      <span className="text-slate-400 text-sm font-mono w-6">{idx + 1}.</span>
                                      <span className="text-white font-medium">{pred.word}</span>
                                    </div>
                                    <div className="flex items-center gap-4 text-sm">
                                      <span className="text-indigo-300 font-mono">
                                        {(pred.probability * 100).toFixed(1)}%
                                      </span>
                                      <span className="text-slate-400 font-mono">
                                        score: {pred.score.toFixed(3)}
                                      </span>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Generated Text */}
                          {generatedText && generationHistory.length > 0 && (
                            <div className="mb-6">
                              <h4 className="text-lg font-semibold text-white mb-3">Generated Text</h4>
                              <div className="bg-slate-700 rounded-lg p-4 min-h-32">
                                <div className="flex flex-wrap gap-1 text-lg leading-relaxed">
                                  {/* Show initial context words */}
                                  {contextInput.split(/\s+/).map((word, idx) => (
                                    <span
                                      key={`context-${idx}`}
                                      className="px-1.5 py-0.5 rounded bg-slate-600 text-slate-300 cursor-default"
                                    >
                                      {word}
                                    </span>
                                  ))}
                                  {/* Show generated tokens as clickable boxes */}
                                  {generationHistory.map((item, idx) => (
                                    <span
                                      key={idx}
                                      onClick={() => {
                                        // Trigger selection in railway viewer
                                        const event = new CustomEvent('selectToken', { detail: idx });
                                        window.dispatchEvent(event);
                                      }}
                                      className={`px-1.5 py-0.5 rounded cursor-pointer transition-colors ${
                                        railwayData?.selectedTokenIndex === idx
                                          ? 'bg-amber-500 text-white ring-2 ring-amber-300'
                                          : 'bg-indigo-600 text-white hover:bg-indigo-500'
                                      }`}
                                    >
                                      {item.word}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            </div>
                          )}

                          {/* Railway Track Visualization */}
                          {railwayData && (
                            <div>
                              <h4 className="text-lg font-semibold text-white mb-3">
                                Generation Flow
                                <span className="text-sm text-slate-400 font-normal ml-2">
                                  ({railwayData.tokens.length} tokens, {railwayData.connections.length} connections)
                                </span>
                              </h4>
                              <div className="bg-slate-700 rounded-lg overflow-hidden">
                                <PSAMRailwayViewer
                                  data={railwayData}
                                  onTokenSelect={setSelectedTokenIndex}
                                />
                              </div>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PSAMInspector;

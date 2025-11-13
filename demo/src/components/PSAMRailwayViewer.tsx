import { useRef, useEffect, useState } from 'react';

interface Token {
  id: number;
  text: string;
  salience: number;
  residual: number;
  perplexity: number;
}

interface Connection {
  from: number;
  to: number;
  type: string;
  strength: number;
}

interface PSAMData {
  tokens: Token[];
  connections: Connection[];
  explanations?: Array<{
    token: number;
    word: string;
    total: number;
    bias: number;
    terms: Array<{
      source: number;
      offset: number;
      weight: number;
      contribution: number;
    }>;
  }>;
  vocab?: { [tokenId: number]: string };
  selectedTokenIndex?: number | null;
}

interface Props {
  data: PSAMData;
  onTokenSelect?: (index: number | null) => void;
}

const PSAMRailwayViewer = ({ data, onTokenSelect }: Props) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [offset, setOffset] = useState(200);
  const [verticalOffset, setVerticalOffset] = useState(0);
  const [zoom, setZoom] = useState(1.0);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [selectedToken, setSelectedToken] = useState<number | null>(null);
  const [hoveredToken, setHoveredToken] = useState<number | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  // Calculate constants (needed by useEffect hooks)
  const STATION_SPACING = 200 * zoom;
  const CENTER_Y = 300 + verticalOffset;
  const LANE_HEIGHT = 80 * zoom;

  // Sync selected token from parent and center it
  useEffect(() => {
    if (data.selectedTokenIndex !== undefined && data.selectedTokenIndex !== selectedToken) {
      setSelectedToken(data.selectedTokenIndex);

      // Center the selected token
      if (data.selectedTokenIndex !== null) {
        const canvas = canvasRef.current;
        if (canvas) {
          const tokenX = data.selectedTokenIndex * STATION_SPACING + 100;
          const centerX = canvas.width / 2;
          const targetOffset = centerX - tokenX;
          setOffset(targetOffset);
        }
      }
    }
  }, [data.selectedTokenIndex, STATION_SPACING, selectedToken]);

  // Listen for external selection events
  useEffect(() => {
    const handleSelectToken = (e: Event) => {
      const customEvent = e as CustomEvent;
      const tokenIdx = customEvent.detail;
      setSelectedToken(tokenIdx);
      onTokenSelect?.(tokenIdx);

      // Center the token when selected from text
      if (tokenIdx !== null) {
        const canvas = canvasRef.current;
        if (canvas) {
          const tokenX = tokenIdx * STATION_SPACING + 100;
          const centerX = canvas.width / 2;
          const targetOffset = centerX - tokenX;

          // Smooth scroll animation
          const startOffset = offset;
          const distance = targetOffset - startOffset;
          const duration = 500; // ms
          const startTime = Date.now();

          const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const easeProgress = 1 - Math.pow(1 - progress, 3);
            setOffset(startOffset + distance * easeProgress);

            if (progress < 1) {
              requestAnimationFrame(animate);
            }
          };

          requestAnimationFrame(animate);
        }
      }
    };

    window.addEventListener('selectToken', handleSelectToken);
    return () => window.removeEventListener('selectToken', handleSelectToken);
  }, [onTokenSelect, STATION_SPACING, offset]);

  // Heatmap color based on strength (0-1)
  const getHeatmapColor = (strength: number, type: string) => {
    // For negative contributions (residual), use cool colors
    if (type === 'residual') {
      // Blue (weak) to purple (strong)
      const r = Math.floor(100 + strength * 155);
      const g = Math.floor(50 + strength * 50);
      const b = Math.floor(200 + strength * 55);
      return `rgb(${r}, ${g}, ${b})`;
    }

    // For positive contributions (dominant), use warm colors
    // Green (weak) -> Yellow (medium) -> Orange (strong) -> Red (very strong)
    if (strength < 0.33) {
      // Green to yellow
      const t = strength / 0.33;
      const r = Math.floor(50 + t * 205);
      const g = Math.floor(200 - t * 45);
      const b = 50;
      return `rgb(${r}, ${g}, ${b})`;
    } else if (strength < 0.66) {
      // Yellow to orange
      const t = (strength - 0.33) / 0.33;
      const r = 255;
      const g = Math.floor(155 - t * 50);
      const b = 50;
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      // Orange to red
      const t = (strength - 0.66) / 0.34;
      const r = 255;
      const g = Math.floor(105 - t * 105);
      const b = Math.floor(50 - t * 50);
      return `rgb(${r}, ${g}, ${b})`;
    }
  };

  const drawConnection = (
    ctx: CanvasRenderingContext2D,
    from: { x: number; y: number },
    to: { x: number; y: number },
    type: string,
    strength: number,
    laneOffset: number,
    fromTokenText: string
  ) => {
    const fromX = from.x + offset;
    const toX = to.x + offset;
    const verticalY = CENTER_Y + (laneOffset * LANE_HEIGHT);

    const rampDist = 60;

    const color = getHeatmapColor(strength, type);
    const lineWidth = Math.max(2, strength * 6);

    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();
    ctx.moveTo(fromX, from.y);
    ctx.lineTo(fromX + rampDist, verticalY);
    ctx.lineTo(toX - rampDist, verticalY);
    ctx.lineTo(toX, to.y);
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(fromX + rampDist, verticalY, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(toX - rampDist, verticalY, 3, 0, Math.PI * 2);
    ctx.fill();

    const midX = (fromX + toX) / 2;
    const isAboveCenter = verticalY < CENTER_Y;
    const labelY = isAboveCenter ? verticalY - 20 : verticalY + 26;

    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    const labelText = `"${fromTokenText}" ${type} (${strength.toFixed(2)})`;
    const metrics = ctx.measureText(labelText);
    const padding = 6;

    ctx.fillStyle = 'rgba(14, 14, 16, 0.9)';
    ctx.fillRect(
      midX - metrics.width / 2 - padding,
      labelY - 9,
      metrics.width + padding * 2,
      18
    );

    ctx.fillStyle = '#8be28b';
    ctx.textAlign = 'left';
    const tokenText = `"${fromTokenText}"`;
    const tokenWidth = ctx.measureText(tokenText).width;
    ctx.fillText(tokenText, midX - metrics.width / 2, labelY);

    ctx.fillStyle = '#ffffff';
    const restText = ` ${type} (${strength.toFixed(2)})`;
    ctx.fillText(restText, midX - metrics.width / 2 + tokenWidth, labelY);
  };

  const drawToken = (
    ctx: CanvasRenderingContext2D,
    token: Token & { x: number; y: number },
    isSelected: boolean,
    isHovered: boolean
  ) => {
    const x = token.x + offset;
    const y = token.y;

    const baseWidth = 80;
    const baseHeight = 40;
    const width = baseWidth + (token.salience * 40);
    const height = baseHeight + (token.salience * 20);

    if (token.perplexity > 0.6) {
      ctx.shadowBlur = 15;
      ctx.shadowColor = '#ffe066';
    } else {
      ctx.shadowBlur = 0;
    }

    ctx.fillStyle = isSelected ? '#ffd700' : (isHovered ? '#ffb84d' : '#d18b00');
    ctx.strokeStyle = '#2a2214';
    ctx.lineWidth = 2;

    const rectX = x - width / 2;
    const rectY = y - height / 2;

    ctx.beginPath();
    ctx.roundRect(rectX, rectY, width, height, 10);
    ctx.fill();
    ctx.stroke();

    ctx.shadowBlur = 0;

    ctx.fillStyle = '#1a1205';
    ctx.font = '14px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(token.text, x, y);

    return { x: rectX, y: rectY, width, height, token };
  };

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    console.log(`[Railway] Drawing ${data.tokens.length} tokens, ${data.connections.length} connections`);

    const tokens = data.tokens.map((token, idx) => ({
      ...token,
      x: idx * STATION_SPACING + 100,
      y: CENTER_Y
    }));

    ctx.fillStyle = '#0e0e10';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // Center line
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, CENTER_Y);
    ctx.lineTo(canvas.width, CENTER_Y);
    ctx.stroke();

    // Lane assignment
    const lanes = new Map<string, number>();
    const laneOccupancy = new Map<string, Array<{ start: number; end: number }>>();

    const sortedConnections = [...data.connections].sort((a, b) =>
      Math.min(a.from, a.to) - Math.min(b.from, b.to)
    );

    sortedConnections.forEach((conn) => {
      const xStart = Math.min(conn.from, conn.to);
      const xEnd = Math.max(conn.from, conn.to);

      let foundLane: number | null = null;
      for (let laneNum = 1; laneNum <= 10; laneNum++) {
        for (const sign of [1, -1]) {
          const testLane = sign * laneNum;
          const key = `lane_${testLane}`;

          if (!laneOccupancy.has(key)) {
            laneOccupancy.set(key, []);
          }

          const occupied = laneOccupancy.get(key)!;
          const hasCollision = occupied.some(range => {
            if (xEnd <= range.start || xStart >= range.end) {
              return false;
            }
            return true;
          });

          if (!hasCollision) {
            foundLane = testLane;
            occupied.push({ start: xStart, end: xEnd });
            break;
          }
        }
        if (foundLane !== null) break;
      }

      lanes.set(`${conn.from}-${conn.to}`, foundLane || 1);
    });

    // Draw connections
    data.connections.forEach((conn) => {
      const fromToken = tokens[conn.from];
      const toToken = tokens[conn.to];
      const laneOffset = lanes.get(`${conn.from}-${conn.to}`) || 0;

      const isHighlighted = selectedToken === conn.from || selectedToken === conn.to;
      if (isHighlighted) {
        ctx.globalAlpha = 1.0;
      } else if (selectedToken !== null) {
        ctx.globalAlpha = 0.2;
      } else {
        ctx.globalAlpha = 0.6;
      }

      drawConnection(ctx, fromToken, toToken, conn.type, conn.strength, laneOffset, fromToken.text);
    });

    ctx.globalAlpha = 1.0;

    // Draw tokens
    tokens.forEach(token => {
      const isSelected = selectedToken === token.id;
      const isHovered = hoveredToken === token.id;
      drawToken(ctx, token, isSelected, isHovered);
    });
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      draw();
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    return () => window.removeEventListener('resize', resizeCanvas);
  }, [offset, verticalOffset, zoom, selectedToken, hoveredToken, data]);

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setMousePos({ x: e.clientX, y: e.clientY });

    if (isDragging) {
      const deltaX = e.clientX - dragStart.x;
      const deltaY = e.clientY - dragStart.y;
      setOffset(prev => prev + deltaX);
      setVerticalOffset(prev => prev + deltaY);
      setDragStart({ x: e.clientX, y: e.clientY });
    } else {
      const tokens = data.tokens.map((token, idx) => ({
        ...token,
        x: idx * STATION_SPACING + 100,
        y: CENTER_Y
      }));

      let foundToken: number | null = null;

      for (const token of tokens) {
        const tx = token.x + offset;
        const width = 80 + (token.salience * 40);
        const height = 40 + (token.salience * 20);
        const rectX = tx - width / 2;
        const rectY = token.y - height / 2;

        if (x >= rectX && x <= rectX + width && y >= rectY && y <= rectY + height) {
          foundToken = token.id;
          break;
        }
      }

      setHoveredToken(foundToken);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleClick = () => {
    if (hoveredToken !== null) {
      const newSelection = selectedToken === hoveredToken ? null : hoveredToken;
      setSelectedToken(newSelection);
      onTokenSelect?.(newSelection);
    }
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.05 : 0.05;
    setZoom(prev => Math.max(0.3, Math.min(3.0, prev + delta)));
  };

  const selectedTokenData = selectedToken !== null ? data.tokens[selectedToken] : null;
  const hoveredTokenData = hoveredToken !== null ? data.tokens[hoveredToken] : null;
  const tooltipToken = hoveredTokenData || selectedTokenData;

  return (
    <div className="bg-slate-900 rounded-lg overflow-hidden flex flex-col" style={{ height: 'calc(100vh - 300px)' }}>
      <div className="bg-slate-800 px-6 py-4 border-b border-slate-700 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div>
              <div className="text-white font-semibold">PSAM Railway Track Viewer</div>
              <div className="text-slate-400 text-xs mt-1">
                Drag to pan • Scroll to zoom • Click token to inspect
              </div>
            </div>

            {/* Heatmap Legend */}
            <div className="flex items-center gap-2">
              <span className="text-slate-400 text-xs">Contribution:</span>
              <div className="flex items-center gap-1">
                <span className="text-slate-400 text-xs">Weak</span>
                <div className="w-24 h-3 rounded-full" style={{
                  background: 'linear-gradient(to right, rgb(50,200,50), rgb(255,155,50), rgb(255,105,50), rgb(255,0,0))'
                }}></div>
                <span className="text-slate-400 text-xs">Strong</span>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="text-slate-400 text-sm">
              Zoom: {(zoom * 100).toFixed(0)}%
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setZoom(prev => Math.max(0.3, prev - 0.1))}
                className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-white rounded text-sm font-medium transition-colors"
              >
                −
              </button>
              <button
                onClick={() => setZoom(1.0)}
                className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-white rounded text-sm font-medium transition-colors"
              >
                Reset
              </button>
              <button
                onClick={() => setZoom(prev => Math.min(3.0, prev + 0.1))}
                className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-white rounded text-sm font-medium transition-colors"
              >
                +
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 relative overflow-hidden">
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onClick={handleClick}
          onWheel={handleWheel}
          className="w-full h-full cursor-grab active:cursor-grabbing"
        />
      </div>

      {/* Tooltip */}
      {tooltipToken && hoveredToken !== null && (
        <div
          style={{
            position: 'fixed',
            left: mousePos.x + 10,
            top: mousePos.y + 10,
          }}
          className="bg-slate-800 border border-slate-600 rounded-lg p-3 text-white text-sm font-mono pointer-events-none z-50 min-w-[180px]"
        >
          <div className="font-semibold mb-2 text-amber-400">{tooltipToken.text}</div>
          <div className="space-y-1 text-xs">
            <div>Salience: {tooltipToken.salience.toFixed(2)}</div>
            <div>Residual: {tooltipToken.residual.toFixed(2)}</div>
            <div>Perplexity: {tooltipToken.perplexity.toFixed(2)}</div>
          </div>
        </div>
      )}

      {/* Bottom Panel - Explanation Table */}
      {selectedToken !== null && data.explanations && data.explanations[selectedToken] ? (
        <div className="bg-slate-800 border-t border-slate-700 flex-shrink-0 flex flex-col" style={{ maxHeight: '300px' }}>
          <div className="px-6 pt-4 pb-2 flex-shrink-0">
            <h3 className="text-white font-semibold text-lg">
              Prediction Explanation for <span className="text-amber-400 font-mono">{data.explanations[selectedToken].word}</span>
            </h3>
            <div className="text-slate-400 text-sm mt-1">
              Total Score: {data.explanations[selectedToken].total.toFixed(3)} |
              Bias: {data.explanations[selectedToken].bias.toFixed(3)}
            </div>
          </div>

          {data.explanations[selectedToken].terms.length > 0 ? (
            <div className="flex-1 overflow-auto px-6 pb-4">
              <div className="bg-slate-900 rounded-lg overflow-hidden">
                <table className="w-full text-sm">
                  <thead className="bg-slate-700 text-slate-300 sticky top-0">
                    <tr>
                      <th className="px-4 py-2 text-left">Source Token</th>
                      <th className="px-4 py-2 text-left">Offset</th>
                      <th className="px-4 py-2 text-right">Weight</th>
                      <th className="px-4 py-2 text-right">Contribution</th>
                      <th className="px-4 py-2 text-right">% of Total</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-700">
                    {data.explanations[selectedToken].terms
                      .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
                      .map((term, idx) => {
                        // Try to find in generated tokens first, then fall back to vocab
                        const sourceToken = data.tokens.find(t => t.id === term.source);
                        const sourceText = sourceToken?.text ||
                                          (data.vocab?.[term.source]) ||
                                          `token_${term.source}`;
                        const pct = (term.contribution / data.explanations![selectedToken].total) * 100;
                        const isPositive = term.contribution > 0;

                        return (
                          <tr key={idx} className="hover:bg-slate-700/50">
                            <td className="px-4 py-2 text-white font-mono">
                              {sourceText}
                            </td>
                            <td className="px-4 py-2 text-slate-400 font-mono">
                              {term.offset}
                            </td>
                            <td className="px-4 py-2 text-right font-mono">
                              <span className={isPositive ? 'text-green-400' : 'text-red-400'}>
                                {term.weight.toFixed(3)}
                              </span>
                            </td>
                            <td className="px-4 py-2 text-right font-mono">
                              <span className={isPositive ? 'text-green-400' : 'text-red-400'}>
                                {term.contribution.toFixed(3)}
                              </span>
                            </td>
                            <td className="px-4 py-2 text-right font-mono text-slate-300">
                              {pct.toFixed(1)}%
                            </td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="text-slate-400 text-center py-4 px-6">
              No contributing terms found
            </div>
          )}
        </div>
      ) : (
        <div className="bg-slate-800 border-t border-slate-700 p-6 flex-shrink-0">
          <div className="text-slate-400 text-center text-sm">
            Click on a token in the visualization to see its prediction explanation
          </div>
        </div>
      )}
    </div>
  );
};

export default PSAMRailwayViewer;

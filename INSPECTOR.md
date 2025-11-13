# PSAM Inspector

Interactive web-based tool for visualizing and analyzing PSAM models.

## Features

### 🎯 Model Loading
- Drag-and-drop `.psam` model files
- Auto-load matching `.tsv` vocabulary files
- Support for multiple models simultaneously
- Browser-based WASM execution (no server required)

### 🚂 Railway Track Visualization
- **Metaphor**: Tokens as stations, associations as connecting tracks
- **Interactive Canvas**: Pan (drag), zoom (scroll/buttons), click to inspect
- **Heatmap Colors**: Green (weak) → Yellow → Orange → Red (strong)
- **Line Thickness**: Proportional to contribution strength
- **Full-screen Support**: Collapsible sidebar, responsive layout

### 📊 Explanation Analysis
- Click any token to see prediction breakdown
- Table shows:
  - Source tokens that influenced prediction
  - Position offset in context
  - Edge weight and contribution
  - Percentage of total score
- Sortable by contribution strength
- Scrollable with sticky headers

### 🎮 Text Generation
- Real-time Shakespearean text generation
- Captures explanation for every token
- Auto-generate up to 50 tokens
- Clear and restart functionality
- View predictions before generating

### ✨ Interactive Token Highlighting
- **Bidirectional Sync**: Click tokens in visualization or text to highlight both
- **Clickable Token Boxes**: Generated text displayed as interactive token boxes
- **Auto-Centering**: Smooth scroll animation centers selected tokens in view
- **Visual States**:
  - Context words: Gray boxes (non-interactive)
  - Generated tokens: Indigo boxes (clickable)
  - Selected token: Amber with ring effect
- **Smooth Animation**: 500ms ease-out cubic scroll to center

## Architecture

### C Core (`core/src/core/inspect.c`)
```c
psam_model_t* psam_load_from_memory(const void* buffer, size_t size);
int psam_get_edges(const psam_model_t* model, uint32_t source_token,
                   float min_weight, size_t max_edges, psam_edge_t* out_edges);
psam_error_t psam_get_config(const psam_model_t* model, psam_config_t* out_config);
```

### WASM Bindings (`demo/src/lib/psam-inspector-wasm.ts`)
- TypeScript wrappers with correct struct alignment
- 20-byte prediction struct handling
- Memory management with malloc/free

### React Components
- **PSAMInspector**: Main app with state management and text generation
- **PSAMRailwayViewer**: Canvas visualization with selection sync
- **Bidirectional Communication**:
  - Props: `selectedTokenIndex` flows down via `data` prop
  - Callbacks: `onTokenSelect` flows up to parent
  - Custom Events: Cross-component `selectToken` events for text clicks

## Quick Start

```bash
cd demo
npm run build
npm run dev
```

Navigate to `/inspector` route.

## Technical Notes

**Struct Alignment**: `psam_config_t` is 28 bytes (not 32).

**Normalization**: Connections normalized by max contribution.

**Colors**:
- Positive: Green → Yellow → Orange → Red
- Negative: Blue → Purple

**Animation**: Smooth scrolling uses `requestAnimationFrame` with ease-out cubic easing for 500ms duration.

**Token Selection**: Synchronized via React state (`selectedTokenIndex`) and DOM custom events (`selectToken`).

## Files Added

- `core/src/core/inspect.c` (360 lines)
- `demo/src/PSAMInspector.tsx` (888 lines)
- `demo/src/components/PSAMRailwayViewer.tsx` (567 lines)
- `demo/src/lib/psam-inspector-wasm.ts` (349 lines)

**Total**: ~2,254 lines

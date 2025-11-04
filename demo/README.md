# PSAM Interactive Demo

Live interactive demonstration of PSAM (Position-Specific Association Memory).

🌐 **Live Demo**: https://foundation42.github.io/libpsam/

## Features

- ✨ **Interactive Training** - Watch PSAM learn token associations in real-time
- 🎛️ **Adjustable Parameters** - Tweak PPMI, IDF, distance decay, and more
- 🔮 **Live Predictions** - See predictions update as you type
- 📊 **Visualization** - View active associations and their weights
- 🎯 **Auto-generation** - Generate sequences automatically

## Local Development

### Prerequisites

- Node.js ≥ 18 or Bun ≥ 1.0

### Setup

```bash
# Install dependencies
npm install
# or
bun install

# Sync WASM assets (copies from bindings/wasm/build or downloads the latest release)
npm run wasm:sync
# or
bun run wasm:sync

# Start development server
npm run dev
# (Bun users: run `bun run wasm:sync` first, then `bun run dev`)

# Build for production
npm run build
# (Bun users: run `bun run wasm:sync` first, then `bun run build`)
```

## How to Use

1. **Training Phase**:
   - Edit the training text or use the default
   - Click "Auto Train" to watch PSAM learn
   - Adjust parameters in the "Show Parameters" panel

2. **Inference Phase**:
   - Enter a context in the inference input
   - Watch predictions appear with scores and probabilities
   - Click "Generate Next" to extend the sequence
   - Use "Auto Generate" for automatic sequence generation

3. **Quick Tests**:
   - Click any quick test button to try pre-configured scenarios
   - See how PSAM learns different patterns

## Parameters Explained

### Core Parameters

- **Context Window** (default: 8)
  - How many tokens before/after to consider during training
  - Larger = capture longer-range dependencies

- **Top-K Pruning** (default: 32)
  - Maximum associations to keep per token/position
  - Smaller = faster, less memory
  - Larger = more comprehensive, better recall

- **Min Evidence** (default: 1)
  - Minimum times a pattern must appear before creating association
  - Higher = filter out noise, require more data

### Weighting Parameters

- **Distance Decay α** (default: 0.1)
  - How quickly influence decays with distance
  - Higher = nearby tokens matter much more

- **Recency Decay λ** (default: 0.05)
  - How quickly old associations fade during training
  - Higher = adapt faster to recent patterns

- **Temperature** (default: 1.0)
  - Prediction diversity
  - Lower = more confident (peaked distribution)
  - Higher = more diverse (flatter distribution)

### Features

- **Enable PPMI** (default: on)
  - Use Positive Pointwise Mutual Information weighting
  - Measures association strength beyond chance co-occurrence

- **Enable IDF** (default: on)
  - Use Inverse Document Frequency weighting
  - Gives rare tokens more influence than common ones

## Architecture

This demo uses a pure JavaScript implementation of PSAM for easy browser compatibility. The full libpsam library includes:

- **Native C library** - 20-200× faster
- **WASM version** - Near-native speed in browser
- **Multi-language bindings** - JavaScript, Python, and more

See the [main repository](https://github.com/Foundation42/libpsam) for the full library.

## Demo vs. Production

| Feature | Demo | libpsam (C) |
|---------|------|-------------|
| **Speed** | ~1-10 inferences/sec | 10,000-100,000/sec |
| **Model Size** | Small (< 10KB vocab) | Large (50K+ vocab) |
| **Persistence** | Browser only | Binary files |
| **Layers** | Single model | Multi-layer composition |
| **Platform** | Browser | Cross-platform |

## Building for GitHub Pages

```bash
# Build the static site
npm run build

# Preview the build
npm run preview
```

The built site will be in `dist/` and can be deployed to GitHub Pages.

## Deployment

This demo is automatically deployed to GitHub Pages via GitHub Actions when changes are pushed to the main branch.

See `.github/workflows/deploy-demo.yml` for the deployment configuration.

## Technologies

- **React** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Lucide React** - Icons

## Learn More

- [PSAM Theory](../docs/PSAM.md) - How PSAM works
- [API Documentation](../docs/API.md) - Using the library
- [Examples](../examples/) - Code examples
- [Main Repository](https://github.com/Foundation42/libpsam)

## License

MIT - see [LICENSE](../LICENSE)
# Updated Wed Oct 22 10:45:14 BST 2025

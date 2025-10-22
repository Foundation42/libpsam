# libpsam Documentation

Complete documentation for libpsam - the Position-Specific Association Memory library.

## Core Documentation

### [📖 PSAM.md](./PSAM.md) - **Understanding PSAM**
Learn how PSAM works, including detailed explanations of:
- Core concept and architecture
- PPMI (Positive Pointwise Mutual Information)
- IDF (Inverse Document Frequency)
- Distance decay and scoring
- Mathematical foundations
- Comparison to transformers and other approaches

**Start here** if you want to understand the theory behind PSAM.

### [🔧 API.md](./API.md) - **API Reference**
Complete API documentation for all language bindings:
- C library functions and types
- JavaScript/TypeScript interfaces
- Python classes and methods
- Code examples for each language
- Common patterns and best practices

**Use this** as your reference when coding with libpsam.

### [🏗️ BUILDING.md](./BUILDING.md) - **Build Guide**
Comprehensive build instructions:
- Building the C library (CMake)
- JavaScript/TypeScript bindings
- Python bindings
- WebAssembly compilation
- Platform-specific notes (Linux, macOS, Windows)
- Troubleshooting

**Follow this** to build libpsam from source.

## Quick Links

### Getting Started
1. [Main README](../README.md) - Quick start and overview
2. [PSAM.md](./PSAM.md) - Understand how it works
3. [BUILDING.md](./BUILDING.md) - Build the library
4. [Examples](../examples/) - See it in action

### Language-Specific
- [C Library](../core/README.md)
- [JavaScript](../bindings/javascript/README.md)
- [Python](../bindings/python/README.md)
- [WASM](../bindings/wasm/README.md)

### Contributing
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines

## Documentation Structure

```
docs/
├── README.md (this file)    # Documentation index
├── PSAM.md                  # How PSAM works (theory)
├── API.md                   # Complete API reference
└── BUILDING.md              # Build instructions
```

## External Resources

- **GitHub**: https://github.com/Foundation42/libpsam
- **Issues**: https://github.com/Foundation42/libpsam/issues
- **Discussions**: https://github.com/Foundation42/libpsam/discussions

## Key Concepts

### PPMI (Positive Pointwise Mutual Information)
Measures how much more often two tokens appear together than expected by chance. See [PSAM.md](./PSAM.md#ppmi-positive-pointwise-mutual-information) for details.

### IDF (Inverse Document Frequency)
Weights tokens by their rarity - rare tokens get more influence than common ones. See [PSAM.md](./PSAM.md#idf-inverse-document-frequency) for details.

### Distance Decay
Nearby tokens have more influence than distant ones, with exponential decay. See [PSAM.md](./PSAM.md#distance-decay) for details.

### Layer Composition
Hot-swappable domain-specific layers for adaptation without retraining. See [PSAM.md](./PSAM.md#layer-composition) for details.

## Examples by Use Case

### Basic Usage
- [C Example](../examples/c/basic_usage.c)
- [JavaScript Example](../examples/javascript/node-example.js)
- [Python Example](../examples/python/basic_usage.py)

### Domain Adaptation
- [C Layer Composition](../examples/c/layer_composition.c)
- [Python Domain Adaptation](../examples/python/domain_adaptation.py)

### Browser/WASM
- [Browser Example](../examples/javascript/browser-example.html)

## Frequently Asked Questions

### What is PSAM?
PSAM (Position-Specific Association Memory) is a novel sequence prediction approach using explicit association graphs rather than dense neural networks. See [PSAM.md](./PSAM.md#what-is-psam).

### How does it compare to transformers?
PSAM is 20-200× faster, uses KB-MB instead of GB, and is fully interpretable. See [Comparison](./PSAM.md#comparison-to-other-approaches).

### Can I use it in production?
Yes! The library is production-ready with:
- Complete test coverage
- Binary serialization
- Thread-safe inference
- Multi-language support

### How do I get started?
1. Read [PSAM.md](./PSAM.md) to understand the concept
2. Follow [BUILDING.md](./BUILDING.md) to build the library
3. Check [examples](../examples/) to see it in action
4. Reference [API.md](./API.md) while coding

## Need Help?

- **Questions**: Open a [GitHub Discussion](https://github.com/Foundation42/libpsam/discussions)
- **Bugs**: File an [Issue](https://github.com/Foundation42/libpsam/issues)
- **Contributing**: See [CONTRIBUTING.md](../CONTRIBUTING.md)

---

**Built with ❤️ and [Claude Code](https://claude.com/claude-code)**

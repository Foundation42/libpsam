# WASM Bindings for libpsam

WebAssembly bindings for browser-based PSAM usage.

## Building

### Prerequisites

Install [Emscripten](https://emscripten.org/docs/getting_started/downloads.html):

```bash
# Linux/macOS
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
```

### Build WASM Module

```bash
cd bindings/wasm
./build.sh
```

This generates:
- `psam.js` - JavaScript loader
- `psam.wasm` - WebAssembly binary

## Usage

### In HTML

```html
<!DOCTYPE html>
<html>
<head>
    <script src="psam.js"></script>
</head>
<body>
    <script>
        createPSAMModule().then(Module => {
            // Create model
            const create = Module.cwrap('psam_create', 'number', ['number', 'number', 'number']);
            const destroy = Module.cwrap('psam_destroy', null, ['number']);

            const model = create(50000, 8, 32);
            console.log('Model created:', model);

            // Use model...

            destroy(model);
        });
    </script>
</body>
</html>
```

### With TypeScript Bindings

Use the high-level TypeScript wrapper from `@foundation42/libpsam`:

```typescript
import { PSAMWASM } from '@foundation42/libpsam/wasm';

const psam = await PSAMWASM.create(50000, 8, 32);
psam.trainBatch([1, 2, 3, 4, 5]);
psam.finalizeTraining();

const predictions = psam.predict([1, 2, 3], 10);
console.log(predictions);
```

## Performance

WASM provides near-native performance in the browser:

- **5-20× faster** than pure JavaScript
- **No server required** - runs entirely client-side
- **Memory efficient** - uses native linear memory

## Browser Support

Requires WebAssembly support:
- Chrome 57+
- Firefox 52+
- Safari 11+
- Edge 16+

## File Size

The WASM module is very small:
- `psam.wasm`: ~20-30 KB (uncompressed)
- `psam.js`: ~10 KB (loader)

With gzip compression, total transfer is typically under 20 KB.

## Notes

- File I/O (`psam_save`/`psam_load`) uses Emscripten's virtual filesystem
- For persistence, use browser storage APIs or server uploads
- Multi-threading requires SharedArrayBuffer (not enabled by default)

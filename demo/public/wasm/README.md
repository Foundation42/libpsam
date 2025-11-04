# libpsam WASM

WebAssembly build of libpsam for browser usage.

## Files

> ⚠️ These files are generated. Run `npm run wasm:sync` (or `bun run wasm:sync`) to populate this folder.

- `psam.wasm` - WebAssembly module
- `psam.js` - Emscripten glue code
- `psam-bindings.js` - TypeScript bindings (ESM)
- `psam-bindings.d.ts` - TypeScript type definitions
- `types.d.ts` - Shared type definitions

## Usage

### In HTML

```html
<script src="psam.js"></script>
<script type="module">
  import { PSAMWASM } from './psam-bindings.js';

  const psam = await PSAMWASM.create(1000, 8, 32);
  psam.trainBatch([1, 2, 3, 4, 5]);
  psam.finalizeTraining();

  const result = psam.predict([1, 2, 3]);
  console.log('Predictions:', result);
</script>
```

### With a bundler

```typescript
import { PSAMWASM } from '@foundation42/libpsam/wasm';

const psam = await PSAMWASM.create(1000, 8, 32);
```

## Documentation

See https://github.com/Foundation42/libpsam for full documentation.

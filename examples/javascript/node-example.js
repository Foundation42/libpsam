/**
 * Node.js example for libpsam
 *
 * Demonstrates basic usage with native bindings
 */

import { PSAMNative, isNativeAvailable } from '@foundation42/libpsam/native';

console.log('╔════════════════════════════════════════════════════════════╗');
console.log('║          libpsam - Node.js Example                        ║');
console.log('╚════════════════════════════════════════════════════════════╝\n');

// Check if native library is available
if (!isNativeAvailable()) {
    console.error('❌ Native library not available');
    console.error('   Build libpsam.so and set LIBPSAM_PATH environment variable');
    process.exit(1);
}

console.log('✓ Native library loaded\n');

// Configuration
const VOCAB_SIZE = 100;
const WINDOW = 8;
const TOP_K = 10;

console.log('📦 Creating PSAM model...');
console.log(`   - Vocabulary size: ${VOCAB_SIZE}`);
console.log(`   - Window: ${WINDOW}`);
console.log(`   - Top-K: ${TOP_K}\n`);

const psam = new PSAMNative(VOCAB_SIZE, WINDOW, TOP_K);

// Training data: "the quick brown fox jumps over the lazy dog"
const tokens = [1, 2, 3, 4, 5, 6, 1, 7, 8];

console.log('📚 Training on sequence...');
console.log(`   Tokens: [${tokens.join(', ')}]\n`);

psam.trainBatch(tokens);
psam.finalizeTraining();

console.log('✓ Training complete!\n');

// Get statistics
const stats = psam.stats();
console.log('📊 Model Statistics:');
console.log(`   - Vocabulary: ${stats.vocabSize} tokens`);
console.log(`   - Rows: ${stats.rowCount}`);
console.log(`   - Edges: ${stats.edgeCount}`);
console.log(`   - Memory: ${stats.memoryBytes} bytes (${(stats.memoryBytes / 1024).toFixed(1)} KB)\n`);

// Make predictions
console.log('🔮 Making predictions...');
const context = [1, 2, 3];  // "the quick brown"
console.log(`   Context: [${context.join(', ')}]\n`);

const predictions = psam.predict(context, 5);

console.log('   Predictions:');
predictions.ids.forEach((tokenId, i) => {
    console.log(`   ${i + 1}. Token ${tokenId} (score: ${predictions.scores[i].toFixed(3)})`);
});
console.log('');

// Sample from distribution
console.log('🎲 Sampling with different temperatures...');
for (const temp of [0.5, 1.0, 2.0]) {
    const sampled = psam.sample(context, temp);
    console.log(`   T=${temp}: Token ${sampled}`);
}
console.log('');

// Save model
const filepath = 'example_model.psam';
console.log(`💾 Saving model to '${filepath}'...`);

psam.save(filepath);
console.log('✓ Model saved!\n');

// Destroy original model
psam.destroy();

// Load model back
console.log(`📂 Loading model from '${filepath}'...`);
const loaded = PSAMNative.load(filepath);
console.log('✓ Model loaded!\n');

// Verify loaded model works
console.log('🔮 Testing loaded model...');
const loadedPreds = loaded.predict(context, 3);
console.log(`✓ Loaded model works! First prediction: Token ${loadedPreds.ids[0]}\n`);

// Cleanup
loaded.destroy();

console.log('🎉 Example complete!\n');
console.log(`libpsam version: ${PSAMNative.version()}`);

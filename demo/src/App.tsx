import { useState } from 'react';
import PSAMv3 from './PSAMv3';
import PSAMWasmDemo from './PSAMWasmDemo';

function App() {
  const [activeTab, setActiveTab] = useState<'js' | 'wasm'>('wasm');

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Tab Selector */}
      <div className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-4">
            <button
              onClick={() => setActiveTab('wasm')}
              className={`px-6 py-4 font-medium text-sm border-b-2 transition-colors ${
                activeTab === 'wasm'
                  ? 'border-indigo-600 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              ⚡ WASM Demo (Real libpsam)
            </button>
            <button
              onClick={() => setActiveTab('js')}
              className={`px-6 py-4 font-medium text-sm border-b-2 transition-colors ${
                activeTab === 'js'
                  ? 'border-indigo-600 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              📚 JS Demo (Pure JavaScript)
            </button>
          </div>
        </div>
      </div>

      {/* Demo Content */}
      <div className="py-8">
        {activeTab === 'wasm' ? <PSAMWasmDemo /> : <PSAMv3 />}
      </div>
    </div>
  );
}

export default App;

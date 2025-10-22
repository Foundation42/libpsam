#!/bin/bash
# Helper script for Docker-based builds and tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

show_help() {
    cat << EOF
Usage: ./scripts/docker-build.sh [COMMAND]

Commands:
    wasm        Build WASM module using Emscripten container
    native      Build native library and run tests
    demo        Build WASM and start demo dev server
    test        Run all tests in containers
    clean       Clean build artifacts and containers
    shell       Open a shell in the WASM build container
    help        Show this help message

Examples:
    ./scripts/docker-build.sh wasm       # Build WASM module
    ./scripts/docker-build.sh demo       # Build and run demo locally
    ./scripts/docker-build.sh test       # Run all tests

EOF
}

case "${1:-}" in
    wasm)
        echo "🔨 Building WASM module in container..."
        docker-compose run --rm wasm-builder
        echo ""
        echo "✅ WASM build complete!"
        echo "   Output: bindings/wasm/build/psam.{js,wasm}"
        ;;

    native)
        echo "🔨 Building native library in container..."
        docker-compose run --rm native-builder
        ;;

    demo)
        echo "🔨 Building WASM and starting demo server..."
        docker-compose run --rm wasm-builder

        # Copy WASM files to demo
        mkdir -p demo/public/wasm
        cp bindings/wasm/build/psam.wasm demo/public/wasm/
        cp bindings/wasm/build/psam.js demo/public/wasm/

        echo ""
        echo "🚀 Starting demo server..."
        echo "   Demo will be available at: http://localhost:5173"
        echo "   Press Ctrl+C to stop"
        docker-compose up demo-dev
        ;;

    test)
        echo "🧪 Running tests..."
        echo ""
        echo "Building native library..."
        docker-compose run --rm native-builder
        echo ""
        echo "Building WASM module..."
        docker-compose run --rm wasm-builder
        echo ""
        echo "✅ All builds successful!"
        ;;

    clean)
        echo "🧹 Cleaning build artifacts..."
        docker-compose down -v
        rm -rf bindings/wasm/build
        rm -rf build
        rm -rf demo/public/wasm
        echo "✅ Clean complete!"
        ;;

    shell)
        echo "🐚 Opening shell in WASM build container..."
        docker-compose run --rm wasm-builder bash
        ;;

    help|--help|-h|"")
        show_help
        ;;

    *)
        echo "❌ Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac

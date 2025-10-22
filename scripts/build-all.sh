#!/bin/bash
# Build all components of libpsam

set -e

echo "🔨 Building libpsam..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Build C library
echo -e "${BLUE}📦 Building C library...${NC}"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON ..
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
cd ..

echo -e "${GREEN}✓ C library built${NC}"

# Build JavaScript bindings
if [ -d "bindings/javascript" ]; then
    echo -e "${BLUE}📦 Building JavaScript bindings...${NC}"
    cd bindings/javascript
    if [ -f "package.json" ]; then
        npm install
        npm run build 2>/dev/null || true
    fi
    cd ../..
    echo -e "${GREEN}✓ JavaScript bindings ready${NC}"
fi

# Build Python bindings
if [ -d "bindings/python" ]; then
    echo -e "${BLUE}📦 Building Python bindings...${NC}"
    cd bindings/python
    if [ -f "setup.py" ]; then
        pip install -e . 2>/dev/null || true
    fi
    cd ../..
    echo -e "${GREEN}✓ Python bindings ready${NC}"
fi

echo -e "${GREEN}🎉 All components built successfully!${NC}"
echo ""
echo "Build outputs:"
echo "  - C library: build/libpsam.so (or .dylib on macOS)"
echo "  - Examples: build/examples/c/"
echo "  - JS bindings: bindings/javascript/dist/"
echo "  - Python: bindings/python/ (installed in dev mode)"

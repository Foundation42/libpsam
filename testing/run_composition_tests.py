#!/bin/bash
#
# PSAM Composition Testing Workflow
# 
# This script automates the complete testing pipeline for PSAM composition.
#

set -e  # Exit on error

SUITE_DIR="synthetic_test_suite"
MODELS_DIR="composition_test_models"
RESULTS_DIR="composition_test_results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================="
echo "PSAM Composition Test Workflow"
echo "=================================="
echo

# Create directories
mkdir -p "$MODELS_DIR"
mkdir -p "$RESULTS_DIR"

# Function to train a model
train_model() {
    local test_name=$1
    local layer_name=$2
    local vocab_file=$3
    
    local input_file="$SUITE_DIR/$test_name/${layer_name}.txt"
    local output_file="$MODELS_DIR/${test_name}_${layer_name}.psam"
    
    if [ -f "$output_file" ]; then
        echo "  ✓ Model already exists: $output_file"
        return 0
    fi
    
    echo "  Training: $test_name / $layer_name"
    
    # Adjust these parameters based on your PSAM CLI
    psam train \
        --input "$input_file" \
        --vocab "$vocab_file" \
        --output "$output_file" \
        --n-gram 3 \
        --quiet || {
            echo -e "${RED}  ✗ Training failed${NC}"
            return 1
        }
    
    echo -e "${GREEN}  ✓ Trained: $output_file${NC}"
}

# Function to create a composition
create_composition() {
    local test_name=$1
    shift
    local layers=("$@")
    
    local output_file="$MODELS_DIR/${test_name}_composed.psamc"
    
    if [ -f "$output_file" ]; then
        echo "  ✓ Composition already exists: $output_file"
        return 0
    fi
    
    echo "  Composing: $test_name"
    
    local cmd="psam compose"
    for layer_spec in "${layers[@]}"; do
        cmd="$cmd --layer $layer_spec"
    done
    cmd="$cmd --output $output_file"
    
    eval $cmd || {
        echo -e "${RED}  ✗ Composition failed${NC}"
        return 1
    }
    
    echo -e "${GREEN}  ✓ Created: $output_file${NC}"
}

# Function to test predictions
test_predictions() {
    local test_name=$1
    local model_file=$2
    local prompt=$3
    
    local output_file="$RESULTS_DIR/${test_name}_predictions.txt"
    
    echo "  Testing: $test_name"
    echo "  Prompt: \"$prompt\""
    
    psam predict \
        --model "$model_file" \
        --prompt "$prompt" \
        --top-k 20 \
        > "$output_file" || {
            echo -e "${RED}  ✗ Prediction failed${NC}"
            return 1
        }
    
    # Analyze results
    python verify_composition.py \
        --test "$test_name" \
        --input "$output_file"
    
    echo -e "${GREEN}  ✓ Results saved: $output_file${NC}"
}

# ================================================================
# TEST 1: Pure Dominance
# ================================================================
echo "Test 1: Pure Layer Dominance"
echo "----------------------------"

train_model "test_01_pure_dominance" "layer_a" "$SUITE_DIR/vocabs/vocab_A_B.tsv"
train_model "test_01_pure_dominance" "layer_b" "$SUITE_DIR/vocabs/vocab_A_B.tsv"

create_composition "test_01_pure_dominance" \
    "$MODELS_DIR/test_01_pure_dominance_layer_a.psam 1.0" \
    "$MODELS_DIR/test_01_pure_dominance_layer_b.psam 0.0"

test_predictions "test_01_pure_dominance" \
    "$MODELS_DIR/test_01_pure_dominance_composed.psamc" \
    "apple"

echo

# ================================================================
# TEST 2: Equal Blend
# ================================================================
echo "Test 2: Equal Blend (50/50)"
echo "----------------------------"

# Reuse models from test 1
create_composition "test_02_equal_blend" \
    "$MODELS_DIR/test_01_pure_dominance_layer_a.psam 0.5" \
    "$MODELS_DIR/test_01_pure_dominance_layer_b.psam 0.5"

test_predictions "test_02_equal_blend" \
    "$MODELS_DIR/test_02_equal_blend_composed.psamc" \
    "apple"

echo

# ================================================================
# TEST 3: Weighted Blend
# ================================================================
echo "Test 3: Weighted Blend (70/30)"
echo "-------------------------------"

create_composition "test_03_weighted_blend" \
    "$MODELS_DIR/test_01_pure_dominance_layer_a.psam 0.7" \
    "$MODELS_DIR/test_01_pure_dominance_layer_b.psam 0.3"

test_predictions "test_03_weighted_blend" \
    "$MODELS_DIR/test_03_weighted_blend_composed.psamc" \
    "apple"

echo

# ================================================================
# TEST 4: Three-Layer Blend
# ================================================================
echo "Test 4: Three-Layer Blend (50/30/20)"
echo "-------------------------------------"

train_model "test_04_three_layer" "layer_a" "$SUITE_DIR/vocabs/vocab_A_B_C.tsv"
train_model "test_04_three_layer" "layer_b" "$SUITE_DIR/vocabs/vocab_A_B_C.tsv"
train_model "test_04_three_layer" "layer_c" "$SUITE_DIR/vocabs/vocab_A_B_C.tsv"

create_composition "test_04_three_layer" \
    "$MODELS_DIR/test_04_three_layer_layer_a.psam 0.5" \
    "$MODELS_DIR/test_04_three_layer_layer_b.psam 0.3" \
    "$MODELS_DIR/test_04_three_layer_layer_c.psam 0.2"

test_predictions "test_04_three_layer" \
    "$MODELS_DIR/test_04_three_layer_composed.psamc" \
    "apple"

echo

# ================================================================
# Weight Sweep Experiment
# ================================================================
echo "Bonus: Weight Sweep Experiment"
echo "-------------------------------"
echo "Testing smooth transition from pure A to pure B..."
echo

for weight_a in 1.0 0.8 0.6 0.5 0.4 0.2 0.0; do
    weight_b=$(echo "1.0 - $weight_a" | bc -l)
    
    sweep_name="sweep_${weight_a}_${weight_b}"
    
    create_composition "$sweep_name" \
        "$MODELS_DIR/test_01_pure_dominance_layer_a.psam $weight_a" \
        "$MODELS_DIR/test_01_pure_dominance_layer_b.psam $weight_b"
    
    psam predict \
        --model "$MODELS_DIR/${sweep_name}_composed.psamc" \
        --prompt "apple" \
        --top-k 20 \
        > "$RESULTS_DIR/${sweep_name}_predictions.txt"
    
    echo "Weight A: $weight_a, Weight B: $weight_b"
    python verify_composition.py --input "$RESULTS_DIR/${sweep_name}_predictions.txt" | grep -A 5 "Probability Mass"
    echo
done

echo
echo "=================================="
echo "Testing Complete!"
echo "=================================="
echo
echo "Results saved to: $RESULTS_DIR/"
echo "Models saved to: $MODELS_DIR/"
echo
echo "You can now:"
echo "  1. Review prediction outputs in $RESULTS_DIR/"
echo "  2. Run additional tests manually"
echo "  3. Visualize weight sweep results"
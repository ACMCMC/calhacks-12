#!/bin/bash

# PrivAds Quick Start Script
# This script sets up the environment and runs a complete training pipeline

set -e  # Exit on error

echo "=========================================="
echo "PrivAds: Quick Start Setup"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt

echo "✓ Dependencies installed"

# Create data directory
mkdir -p data
mkdir -p checkpoints
echo "✓ Directories created"

# Generate synthetic data
echo ""
echo "=========================================="
echo "Step 1: Generating Synthetic Data"
echo "=========================================="
python generate_data.py

# Train model
echo ""
echo "=========================================="
echo "Step 2: Training Model"
echo "=========================================="
python train.py \
    --data data/train_clicks.csv \
    --num-users 5000 \
    --epochs 10 \
    --batch-size 64 \
    --user-dim 256 \
    --lr 0.001 \
    --use-triplets \
    --save-dir checkpoints

echo ""
echo "=========================================="
echo "✅ Training Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Check training results in: checkpoints/"
echo "  2. Run inference with: python inference.py"
echo "  3. Visualize embeddings (coming soon)"
echo ""

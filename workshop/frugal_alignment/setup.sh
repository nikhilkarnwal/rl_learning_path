#!/bin/bash

# Quick start script for Llama-3 fine-tuning

echo "🚀 Setting up Llama-3 Fine-tuning Environment"
echo "=============================================="

# Check if Python 3.10 is installed
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    PYTHON_VERSION=$(python3.10 --version 2>&1 | awk '{print $2}')
    echo "✅ Found Python 3.10: $PYTHON_VERSION"
else
    echo "⚠️  Python 3.10 not found. Installing..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew is not installed. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    # Install Python 3.10
    echo "📦 Installing Python 3.10 via Homebrew..."
    brew install python@3.10
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Python 3.10"
        exit 1
    fi
    
    PYTHON_CMD="python3.10"
    echo "✅ Python 3.10 installed successfully"
fi

# Verify Python version is 3.10+


# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
$PYTHON_CMD -m venv venv-llama


# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv-llama/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Install PyTorch with CUDA
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo "📚 Installing dependencies..."
pip install transformers datasets peft trl bitsandbytes accelerate wandb pyyaml

# Install Unsloth
echo "⚡ Installing Unsloth..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To get started:"
echo "  1. Activate the environment: source venv-llama/bin/activate"
echo "  2. Edit config.yaml to customize training"
echo "  3. Run training: python train.py"
echo ""
echo "For Weights & Biases tracking, run: wandb login"

#!/bin/bash
# Quick setup script for the RL learning path

cd "$(dirname "$0")"

echo "Setting up RL Learning Path..."

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    /opt/homebrew/bin/python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements-simple.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the setup, run:"
echo "  python src/main.py --test-env"

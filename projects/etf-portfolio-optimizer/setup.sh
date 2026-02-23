#!/bin/bash
# ETF Portfolio Optimizer Setup Script

set -e  # Exit on error

echo "========================================="
echo "ETF Portfolio Optimizer Setup"
echo "========================================="

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo -e "\nCreating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo -e "\nUpgrading pip..."
pip install --upgrade pip

# Install dependencies
echo -e "\nInstalling dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo -e "\nCreating directories..."
mkdir -p data outputs logs

# Test the installation
echo -e "\nTesting installation..."
python3 -c "import yfinance, pandas, numpy, scipy; print('✓ Core libraries imported successfully')"

# Make scripts executable
chmod +x run_optimizer.py

echo -e "\n========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run optimization:"
echo "  python run_optimizer.py"
echo ""
echo "For quick test:"
echo "  python run_optimizer.py --quick"
echo ""
echo "To run with backtesting:"
echo "  python run_optimizer.py --backtest"
echo ""
echo "Output files will be saved to outputs/ directory"
echo ""
echo "========================================="
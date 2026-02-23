#!/bin/bash
# Investment Monitor Setup Script

set -e  # Exit on error

echo "========================================="
echo "Investment Monitor Setup"
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
mkdir -p data/cache reports logs

# Copy example config if config doesn't exist
if [ ! -f "config/config.yaml" ]; then
    echo -e "\nCopying example configuration..."
    cp config/config.example.yaml config/config.yaml
    echo "Please edit config/config.yaml with your settings"
fi

# Test the installation
echo -e "\nTesting installation..."
python3 -c "import yfinance, pandas, numpy; print('✓ Core libraries imported successfully')"

# Make scripts executable
chmod +x run_monitor.py

echo -e "\n========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run a test:"
echo "  python run_monitor.py --test"
echo ""
echo "To run full analysis:"
echo "  python run_monitor.py"
echo ""
echo "To set up scheduled daily runs, add to crontab:"
echo "  0 18 * * * cd /path/to/investment-monitor && /path/to/venv/bin/python run_monitor.py >> logs/cron.log 2>&1"
echo ""
echo "Don't forget to:"
echo "1. Edit config/config.yaml with your preferences"
echo "2. Set up API keys if needed"
echo "3. Test the system with: python run_monitor.py --test"
echo "========================================="
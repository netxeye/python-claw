#!/bin/bash
# ETF Portfolio Manager - Complete System Setup

set -e  # Exit on error

echo "========================================="
echo "ETF Portfolio Manager Setup"
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
mkdir -p data outputs logs reports weekly_scans backtest_outputs web_dashboard/templates

# Test the installation
echo -e "\nTesting installation..."
python3 -c "import yfinance, pandas, numpy, flask, schedule; print('✓ All core libraries imported successfully')"

# Make scripts executable
chmod +x run_system.py

echo -e "\n========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "Quick Start Options:"
echo ""
echo "1. Run 5-year backtest:"
echo "   python run_system.py --backtest"
echo ""
echo "2. Run weekly scan:"
echo "   python run_system.py --scan"
echo ""
echo "3. Start automated system:"
echo "   python run_system.py --schedule"
echo ""
echo "4. Generate system report:"
echo "   python run_system.py --report"
echo ""
echo "5. Start web dashboard:"
echo "   cd web_dashboard && python app.py"
echo "   Then open: http://localhost:5000"
echo ""
echo "System Features:"
echo "• 5-year historical backtesting"
echo "• Weekly portfolio scanning"
echo "• Real-time web dashboard"
echo "• Automated scheduling"
echo "• 15% annual return optimization"
echo ""
echo "Output directories:"
echo "• reports/ - System reports"
echo "• weekly_scans/ - Weekly scan results"
echo "• backtest_outputs/ - Backtest results"
echo "• logs/ - System logs"
echo ""
echo "========================================="
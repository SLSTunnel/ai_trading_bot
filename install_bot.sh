#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print section headers
print_section() {
    echo -e "\n${YELLOW}=== $1 ===${NC}"
}

# Function to print success message
print_success() {
    echo -e "\n${GREEN}$1${NC}"
}

# Update system
print_section "Updating System Packages"
sudo apt update && sudo apt upgrade -y

# Install system dependencies
print_section "Installing System Dependencies"
sudo apt install -y python3-pip python3-venv git build-essential

# Install TA-Lib dependencies
print_section "Installing TA-Lib Dependencies"
sudo apt install -y libffi-dev libssl-dev libxml2-dev libxslt1-dev zlib1g-dev

# Install TA-Lib
print_section "Installing TA-Lib"
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make -j$(nproc)
sudo make install
cd ..
rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Create and activate virtual environment
print_section "Setting Up Python Environment"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install wheel
pip install --upgrade pip wheel

# Install Python dependencies
print_section "Installing Python Dependencies"
pip install -r requirements.txt

# Install MetaTrader 5
pip install MetaTrader5 --no-cache-dir

# Create configuration directory
mkdir -p config

# Create a default .env file if it doesn't exist
if [ ! -f ".env" ]; then
    cat > .env << EOL
# MT5 Web Credentials (Will be set through web interface)
MT5_ACCOUNT=
MT5_PASSWORD=
MT5_SERVER=

# Trading Parameters
SYMBOLS=EURUSD,GBPUSD,USDJPY
TIMEFRAME=H1
RISK_PERCENT=1.0
LEVERAGE=30

# Model Settings
MODEL_PATH=ai_model.joblib
MODEL_RETRAIN_DAYS=7

# Risk Management
STOP_LOSS_PIPS=20
TAKE_PROFIT_PIPS=40
MAX_DAILY_LOSS=5.0

# Logging
LOG_LEVEL=INFO
LOG_FILE=bot.log
EOL
    print_success "Created default .env file. Please configure your settings through the web interface."
fi

# Make run script executable
chmod +x run_bot.py

# Create a desktop entry for easy startup
if [ -d "/usr/share/applications" ]; then
    cat > /usr/share/applications/ai-trading-bot.desktop << EOL
[Desktop Entry]
Name=AI Trading Bot
Exec=$(pwd)/run_bot.py
Icon=utilities-terminal
Terminal=true
Type=Application
Categories=Finance;
Path=$(pwd)
EOL
    print_success "Created desktop entry for AI Trading Bot"
fi

# Display completion message
print_success "\n✅ Installation completed successfully!"
echo -e "\nNext steps:"
echo "1. Start the bot: ${YELLOW}./run_bot.py${NC}"
echo "2. Open your web browser to: ${YELLOW}http://localhost:8050${NC}"
echo "3. Configure your MT5 account through the web interface"
echo -e "\nFor support, please refer to the documentation or open an issue.\n"

# Install dashboard requirements
print_section "Installing Dashboard Dependencies"
pip install -r dashboard_requirements.txt

# Create necessary directories
mkdir -p logs

# Create systemd service for auto-start
print_section "Setting Up Auto-Start"
if [ -d "/etc/systemd/system" ]; then
    cat > /etc/systemd/system/ai-trading-bot.service << EOL
[Unit]
Description=AI Trading Bot with Dashboard
After=network.target

[Service]
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin"
ExecStart=$(pwd)/venv/bin/python dashboard.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOL

    # Enable and start the service
    sudo systemctl daemon-reload
    sudo systemctl enable ai-trading-bot.service
    sudo systemctl start ai-trading-bot.service
    
    print_success "Trading bot service has been configured to start on boot"
fi

# Display completion message
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✅ AI Trading Bot Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\n📊 ${BOLD}Dashboard URL:${NC} http://$(hostname -I | awk '{print $1}'):5000"
echo -e "\nTo manage the service:"
echo -e "   Start:  ${BOLD}sudo systemctl start ai-trading-bot${NC}"
echo -e "   Stop:   ${BOLD}sudo systemctl stop ai-trading-bot${NC}"
echo -e "   Status: ${BOLD}sudo systemctl status ai-trading-bot${NC}"
echo -e "   Logs:   ${BOLD}sudo journalctl -u ai-trading-bot -f${NC}"

# Make the installation script executable
chmod +x install_bot.sh

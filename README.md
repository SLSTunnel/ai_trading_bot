# AI-Powered Trading Bot for MetaTrader 5

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MetaTrader5](https://img.shields.io/badge/Platform-MetaTrader%205-00a69c.svg)](https://www.metatrader5.com/)

An advanced algorithmic trading system that leverages machine learning and technical analysis to execute trades on the MetaTrader 5 platform. This bot is designed for forex and CFD trading with robust risk management and real-time performance monitoring.

## 🚀 Key Features

- **Machine Learning Integration**: Uses scikit-learn and XGBoost for predictive modeling
- **Multi-Asset Trading**: Supports multiple currency pairs and timeframes
- **Advanced Risk Management**: Implements position sizing, stop-loss, and take-profit strategies
- **Real-time Analytics**: Live monitoring of trades and account metrics
- **Backtesting Engine**: Validate strategies on historical data
- **Web Dashboard**: Interactive interface for monitoring and control
- **Automated Execution**: Seamless integration with MetaTrader 5 terminal

## 📦 Prerequisites

- Python 3.8 or higher
- MetaTrader 5 account and terminal installed
- Windows OS (required for MetaTrader 5 integration)
- 4GB+ RAM recommended
- Stable internet connection

## 🛠 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-trading-bot.git
   cd ai-trading-bot
   ```

2. **Set up virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/MacOS (MetaTrader 5 requires Windows, but setup can be done on other OS)
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Note: For TA-Lib installation on Windows, download the appropriate wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)

4. **Configure your settings**
   - Copy `.env.example` to `.env`
   - Update with your MetaTrader 5 credentials and trading parameters

## 🚦 Quick Start

1. **Initialize the database**
   ```bash
   python init_db.py
   ```

2. **Run the trading bot**
   ```bash
   python run_bot.py
   ```

3. **Access the web dashboard** (optional)
   ```bash
   python dashboard.py
   ```
   Open `http://localhost:8050` in your browser

## 📊 Project Structure

```
ai-trading-bot/
├── advanced_trading_engine.py  # Core trading logic and execution
├── ai_trading_strategy.py     # Machine learning strategy implementation
├── dashboard.py              # Web-based monitoring dashboard
├── models.py                 # Database models for trade tracking
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## ⚙️ Configuration

Edit the `.env` file to customize your trading parameters:

```ini
# MT5 Account Settings
MT5_ACCOUNT=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server

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
```

## 🤖 AI Strategy

The trading bot uses a combination of technical indicators and machine learning to generate signals:

- **Feature Engineering**:
  - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
  - Price action patterns
  - Volatility measures
  - Market sentiment (if available)

- **Machine Learning**:
  - Random Forest classifier for signal generation
  - Model retraining on a scheduled basis
  - Confidence thresholding for trade execution

## 📈 Performance Monitoring

Track your bot's performance through:

1. **Console Logs**: Real-time trade execution and status updates
2. **Web Dashboard**: Visualize performance metrics and account statistics
3. **Log Files**: Detailed records of all trading activity

## ⚠️ Risk Warning

- Trading financial instruments carries a high level of risk
- Past performance is not indicative of future results
- Only trade with capital you can afford to lose
- Test thoroughly in demo mode before live trading

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For questions or support, please open an issue on GitHub.

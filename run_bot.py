import MetaTrader5 as mt5
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
import pytz
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_trading_engine import AdvancedTradingBot, OrderType
from ai_trading_strategy import AITradingStrategy
from config import MT5_CONFIG, TRADING_CONFIG, LOGGING_CONFIG

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
LOG_FILE = 'trading_bot.log'
SYMBOLS = TRADING_CONFIG.get('symbols', ['EURUSD', 'GBPUSD', 'USDJPY'])
TIMEFRAME = TRADING_CONFIG.get('timeframe', 'H1')
MAGIC_NUMBER = 123456

def initialize_mt5() -> bool:
    """Initialize MetaTrader 5 connection."""
    if not mt5.initialize():
        logger.error("Failed to initialize MT5")
        return False
    
    # Log in to the account
    authorized = mt5.login(
        login=MT5_CONFIG['login'],
        password=MT5_CONFIG['password'],
        server=MT5_CONFIG['server']
    )
    
    if not authorized:
        logger.error(f"Failed to connect to account #{MT5_CONFIG['login']}")
        return False
    
    logger.info(f"Connected to account #{MT5_CONFIG['login']}")
    return True

def main():
    """Main function to run the trading bot."""
    logger.info("Starting Advanced Trading Bot...")
    
    # Initialize MT5 connection
    if not initialize_mt5():
        return
    
    try:
        # Create and configure the trading bot
        bot = AdvancedTradingBot()
        
        # Create and add AI trading strategy
        ai_strategy = AITradingStrategy(model_path='ai_trading_model.joblib')
        bot.add_strategy(ai_strategy, symbols=SYMBOLS, timeframe=TIMEFRAME)
        
        # Train the AI model with historical data (optional)
        # Uncomment the following lines to enable training
        # for symbol in SYMBOLS:
        #     rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, f"TIMEFRAME_{TIMEFRAME}"), 0, 5000)
        #     if rates is not None and len(rates) > 0:
        #         df = pd.DataFrame(rates)
        #         df['time'] = pd.to_datetime(df['time'], unit='s')
        #         logger.info(f"Training AI model for {symbol}...")
        #         ai_strategy.train(df)
        
        # Run the bot
        logger.info("Starting trading bot...")
        bot.run()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        # Shut down
        mt5.shutdown()
        logger.info("Trading bot stopped")

if __name__ == "__main__":
    main()

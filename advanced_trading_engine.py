import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import pytz
import talib
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)

class OrderType(Enum):
    BUY = auto()
    SELL = auto()
    BUY_LIMIT = auto()
    SELL_LIMIT = auto()
    BUY_STOP = auto()
    SELL_STOP = auto()

class OrderStatus(Enum):
    PENDING = auto()
    FILLED = auto()
    REJECTED = auto()
    CANCELLED = auto()
    PARTIAL = auto()

@dataclass
class Order:
    symbol: str
    order_type: OrderType
    volume: float
    price: float
    sl: float
    tp: float
    magic: int = 0
    comment: str = ""
    ticket: int = 0
    status: OrderStatus = OrderStatus.PENDING
    open_time: datetime = None
    close_time: datetime = None
    commission: float = 0.0
    swap: float = 0.0
    profit: float = 0.0

class RiskManager:
    """Manages risk across all trades and strategies."""
    
    def __init__(self, max_risk_per_trade: float = 0.01, max_daily_drawdown: float = 0.02):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_drawdown = max_daily_drawdown
        self.daily_pnl = 0.0
        self.last_reset = datetime.now()
    
    def calculate_position_size(self, symbol: str, entry: float, stop_loss: float, risk_percent: float) -> float:
        """Calculate position size based on account balance and risk parameters."""
        try:
            # Get account balance
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return 0.0
                
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Failed to get symbol info for {symbol}")
                return 0.0
                
            # Calculate risk amount in account currency
            risk_amount = account_info.balance * risk_percent
            
            # Calculate pip value
            point = symbol_info.point
            pip_value = symbol_info.trade_tick_value / symbol_info.trade_tick_size * point
            
            # Calculate stop loss in pips
            stop_pips = abs(entry - stop_loss) / point
            
            # Calculate position size
            position_size = (risk_amount / stop_pips) / pip_value
            
            # Normalize to lot size
            lot_step = symbol_info.volume_step
            position_size = round(position_size / lot_step) * lot_step
            
            # Ensure position size is within limits
            position_size = max(symbol_info.volume_min, min(symbol_info.volume_max, position_size))
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def check_daily_drawdown(self) -> bool:
        """Check if daily drawdown limit is reached."""
        # Reset daily P&L if it's a new day
        now = datetime.now()
        if now.date() > self.last_reset.date():
            self.daily_pnl = 0.0
            self.last_reset = now
            
        # Get current account balance
        account_info = mt5.account_info()
        if account_info is None:
            return False
            
        # Calculate drawdown
        if account_info.balance > 0:
            drawdown = 1 - (account_info.equity / account_info.balance)
            return drawdown >= self.max_daily_drawdown
            
        return False

class AdvancedTradingEngine:
    """Advanced trading engine with position management and risk control."""
    
    def __init__(self, risk_manager: RiskManager = None):
        self.risk_manager = risk_manager or RiskManager()
        self.positions = {}
        self.orders = {}
        self.lock = threading.RLock()
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    def start(self):
        """Start the trading engine."""
        self.is_running = True
        logger.info("Trading engine started")
        
        # Start background tasks
        self.executor.submit(self._monitor_positions)
        self.executor.submit(self._monitor_orders)
    
    def stop(self):
        """Stop the trading engine."""
        self.is_running = False
        self.executor.shutdown(wait=True)
        logger.info("Trading engine stopped")
    
    def _monitor_positions(self):
        """Monitor open positions and manage them."""
        while self.is_running:
            try:
                with self.lock:
                    # Get all open positions
                    positions = mt5.positions_get()
                    if positions is None:
                        time.sleep(1)
                        continue
                        
                    # Update positions dictionary
                    current_positions = {pos.ticket: pos for pos in positions}
                    
                    # Check for closed positions
                    for ticket in list(self.positions.keys()):
                        if ticket not in current_positions:
                            # Position was closed
                            pos = self.positions.pop(ticket)
                            logger.info(f"Position closed: {pos.ticket} - {pos.symbol} - {pos.volume} lots")
                    
                    # Update existing positions and add new ones
                    for ticket, pos in current_positions.items():
                        if ticket in self.positions:
                            # Update existing position
                            self.positions[ticket] = pos
                        else:
                            # New position
                            self.positions[ticket] = pos
                            logger.info(f"New position opened: {pos.ticket} - {pos.symbol} - {pos.volume} lots")
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in position monitoring: {e}")
                time.sleep(5)
    
    def _monitor_orders(self):
        """Monitor pending orders."""
        while self.is_running:
            try:
                with self.lock:
                    # Get all pending orders
                    orders = mt5.orders_get()
                    if orders is None:
                        time.sleep(1)
                        continue
                        
                    # Update orders dictionary
                    current_orders = {order.ticket: order for order in orders}
                    
                    # Check for filled or cancelled orders
                    for ticket in list(self.orders.keys()):
                        if ticket not in current_orders:
                            # Order was filled or cancelled
                            order = self.orders.pop(ticket)
                            logger.info(f"Order {order.ticket} - {order.symbol} - {order.volume} lots was filled or cancelled")
                    
                    # Update existing orders and add new ones
                    for ticket, order in current_orders.items():
                        if ticket in self.orders:
                            self.orders[ticket] = order
                        else:
                            self.orders[ticket] = order
                            logger.info(f"New order placed: {order.ticket} - {order.symbol} - {order.volume} lots")
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in order monitoring: {e}")
                time.sleep(5)
    
    def open_position(self, symbol: str, order_type: OrderType, volume: float, 
                     sl: float = 0.0, tp: float = 0.0, magic: int = 0, comment: str = "") -> Optional[Order]:
        """Open a new position."""
        try:
            # Check if we've hit daily drawdown
            if self.risk_manager.check_daily_drawdown():
                logger.warning("Daily drawdown limit reached. No new positions will be opened.")
                return None
            
            # Get current price
            symbol_info = mt5.symbol_info_tick(symbol)
            if symbol_info is None:
                logger.error(f"Failed to get tick data for {symbol}")
                return None
                
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "sl": sl,
                "tp": tp,
                "deviation": 10,
                "magic": magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            # Set order type and price
            if order_type == OrderType.BUY:
                request["type"] = mt5.ORDER_TYPE_BUY
                request["price"] = symbol_info.ask
            elif order_type == OrderType.SELL:
                request["type"] = mt5.ORDER_TYPE_SELL
                request["price"] = symbol_info.bid
            else:
                logger.error(f"Unsupported order type: {order_type}")
                return None
            
            # Send order
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.comment}")
                return None
                
            # Create order object
            order = Order(
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                price=result.price,
                sl=sl,
                tp=tp,
                magic=magic,
                comment=comment,
                ticket=result.order,
                status=OrderStatus.FILLED,
                open_time=datetime.now(),
                commission=result.commission,
                swap=result.swap,
                profit=result.profit
            )
            
            # Add to orders dictionary
            with self.lock:
                self.orders[order.ticket] = order
                
            return order
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return None
    
    def close_position(self, ticket: int, volume: float = None, comment: str = "") -> bool:
        """Close an open position."""
        try:
            # Get position
            position = mt5.positions_get(ticket=ticket)
            if not position:
                logger.error(f"Position {ticket} not found")
                return False
                
            position = position[0]
            
            # If volume is not specified, close the entire position
            if volume is None or volume > position.volume:
                volume = position.volume
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "volume": volume,
                "deviation": 10,
                "magic": position.magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            # Set order type and price
            if position.type == mt5.POSITION_TYPE_BUY:
                request["type"] = mt5.ORDER_TYPE_SELL
                request["price"] = mt5.symbol_info_tick(position.symbol).bid
            else:
                request["type"] = mt5.ORDER_TYPE_BUY
                request["price"] = mt5.symbol_info_tick(position.symbol).ask
            
            # Send close order
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Close position failed: {result.comment}")
                return False
                
            logger.info(f"Position {ticket} closed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position {ticket}: {e}")
            return False
    
    def close_all_positions(self, symbol: str = None, magic: int = None) -> int:
        """Close all open positions."""
        try:
            # Get all open positions
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
                
            if positions is None:
                return 0
                
            # Filter by magic number if specified
            if magic is not None:
                positions = [p for p in positions if p.magic == magic]
                
            # Close positions
            closed = 0
            for position in positions:
                if self.close_position(position.ticket):
                    closed += 1
                    
            return closed
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return 0

class AdvancedTradingBot:
    """Advanced trading bot with AI strategy and risk management."""
    
    def __init__(self):
        self.engine = AdvancedTradingEngine()
        self.strategies = {}
        self.symbols = []
        self.timeframe = mt5.TIMEFRAME_H1
        self.magic = 123456  # Magic number for this bot
        
    def add_strategy(self, strategy, symbols: list, timeframe: str):
        """Add a trading strategy."""
        self.strategies[strategy.__class__.__name__] = {
            'strategy': strategy,
            'symbols': symbols,
            'timeframe': timeframe
        }
        self.symbols = list(set(self.symbols + symbols))
        
    def run(self):
        """Run the trading bot."""
        # Initialize MT5
        if not mt5.initialize():
            logger.error("Failed to initialize MT5")
            return False
            
        # Start trading engine
        self.engine.start()
        
        try:
            while True:
                try:
                    # Check market conditions for each symbol and strategy
                    for strategy_name, config in self.strategies.items():
                        strategy = config['strategy']
                        symbols = config['symbols']
                        timeframe = config['timeframe']
                        
                        for symbol in symbols:
                            try:
                                # Get market data
                                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1000)
                                if rates is None or len(rates) == 0:
                                    logger.error(f"No data for {symbol}")
                                    continue
                                    
                                df = pd.DataFrame(rates)
                                df['time'] = pd.to_datetime(df['time'], unit='s')
                                
                                # Get signal from strategy
                                signal = strategy.generate_signal(df, symbol)
                                
                                if signal and signal.get('action') == 'buy':
                                    # Calculate position size based on risk
                                    volume = self.engine.risk_manager.calculate_position_size(
                                        symbol, signal['price'], signal['stop_loss'], 0.01
                                    )
                                    
                                    if volume > 0:
                                        # Open buy position
                                        self.engine.open_position(
                                            symbol=symbol,
                                            order_type=OrderType.BUY,
                                            volume=volume,
                                            sl=signal['stop_loss'],
                                            tp=signal['take_profit'],
                                            magic=self.magic,
                                            comment=f"{strategy_name} signal"
                                        )
                                        
                                elif signal and signal.get('action') == 'sell':
                                    # Calculate position size based on risk
                                    volume = self.engine.risk_manager.calculate_position_size(
                                        symbol, signal['price'], signal['stop_loss'], 0.01
                                    )
                                    
                                    if volume > 0:
                                        # Open sell position
                                        self.engine.open_position(
                                            symbol=symbol,
                                            order_type=OrderType.SELL,
                                            volume=volume,
                                            sl=signal['stop_loss'],
                                            tp=signal['take_profit'],
                                            magic=self.magic,
                                            comment=f"{strategy_name} signal"
                                        )
                                        
                            except Exception as e:
                                logger.error(f"Error processing {symbol} with {strategy_name}: {e}")
                                
                    # Sleep for a while before next check
                    time.sleep(60)
                    
                except KeyboardInterrupt:
                    logger.info("Shutting down...")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(60)
                    
        finally:
            # Clean up
            self.engine.stop()
            mt5.shutdown()
            
        return True

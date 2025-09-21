import numpy as np
import pandas as pd
import talib
from typing import Dict, Any, Optional
import joblib
import os
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class AITradingStrategy:
    """
    Advanced AI-powered trading strategy using multiple technical indicators
    and machine learning for trade signal generation.
    """
    
    def __init__(self, model_path: str = 'ai_trading_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.load_model()
    
    def load_model(self):
        """Load a pre-trained model or initialize a new one."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logger.info("Loaded pre-trained AI model")
                return
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
        
        # Initialize a new model if loading failed or no model exists
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        logger.info("Initialized new AI model")
    
    def save_model(self):
        """Save the trained model to disk."""
        try:
            joblib.dump(self.model, self.model_path)
            logger.info(f"Saved AI model to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators and features for the AI model."""
        df = df.copy()
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving Averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{window}'] = talib.SMA(df['close'], timeperiod=window)
            df[f'ema_{window}'] = talib.EMA(df['close'], timeperiod=window)
        
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # MACD
        macd, signal, _ = talib.MACD(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = macd - signal
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = (upper - lower) / middle
        
        # ATR (Average True Range)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Stochastic Oscillator
        stoch_k, stoch_d = talib.STOCH(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # ADX (Average Directional Index)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # OBV (On-Balance Volume)
        df['obv'] = talib.OBV(df['close'], df['tick_volume'])
        
        # Volume features
        df['volume_ma'] = talib.SMA(df['tick_volume'], timeperiod=20)
        df['volume_ratio'] = df['tick_volume'] / df['volume_ma'].replace(0, 1)
        
        # Price momentum
        df['momentum'] = talib.MOM(df['close'], timeperiod=10)
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        """Prepare features and labels for training the model."""
        # Calculate features
        df_features = self.calculate_features(df)
        
        # Create target variable (1 if next return is positive, 0 otherwise)
        df_features['target'] = (df_features['close'].shift(-1) > df_features['close']).astype(int)
        
        # Select feature columns (all numeric columns except the target)
        feature_cols = [col for col in df_features.select_dtypes(include=[np.number]).columns 
                       if col not in ['target', 'time', 'tick_volume', 'spread', 'real_volume']]
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        # Prepare features and target
        X = df_features[feature_cols].values
        y = df_features['target'].values
        
        return X, y
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train the AI model on historical data."""
        logger.info("Starting AI model training...")
        
        try:
            # Prepare training data
            X, y = self.prepare_training_data(df)
            
            if len(X) < 100:  # Minimum samples required for training
                logger.warning(f"Insufficient data for training. Got {len(X)} samples, need at least 100.")
                return {"status": "error", "message": "Insufficient training data"}
            
            # Split data into train and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train the model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate the model
            train_accuracy = self.model.score(X_train_scaled, y_train)
            val_accuracy = self.model.score(X_val_scaled, y_val)
            
            # Save the trained model
            self.save_model()
            
            logger.info(f"Model trained successfully. Train accuracy: {train_accuracy:.2f}, Val accuracy: {val_accuracy:.2f}")
            
            return {
                "status": "success",
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "n_samples": len(X),
                "feature_importance": dict(zip(self.feature_columns, self.model.feature_importances_))
            }
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return {"status": "error", "message": str(e)}
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Dict[str, Any]]:
        """Generate trading signals using the AI model."""
        if len(df) < 100:  # Minimum samples required for prediction
            logger.warning("Insufficient data for prediction")
            return None
            
        try:
            # Calculate features for the latest data point
            df_features = self.calculate_features(df)
            
            if len(df_features) == 0:
                logger.warning("No features calculated")
                return None
            
            # Get the most recent data point for prediction
            latest = df_features.iloc[-1:]
            
            # If feature columns aren't set yet, use all numeric columns
            if self.feature_columns is None:
                self.feature_columns = [col for col in latest.select_dtypes(include=[np.number]).columns 
                                      if col not in ['target', 'time', 'tick_volume', 'spread', 'real_volume']]
            
            # Select features and scale
            X = latest[self.feature_columns].values
            if X.shape[1] != len(self.feature_columns):
                logger.error("Feature dimension mismatch")
                return None
                
            X_scaled = self.scaler.transform(X)
            
            # Get prediction probabilities
            proba = self.model.predict_proba(X_scaled)[0]
            
            # Get the predicted class (1 for long, 0 for short)
            prediction = self.model.predict(X_scaled)[0]
            
            # Calculate confidence (probability of the predicted class)
            confidence = proba[prediction]
            
            # Only trade if confidence is above threshold
            if confidence < 0.6:  # 60% confidence threshold
                return None
            
            # Get current price and volatility
            current_price = latest['close'].values[0]
            atr = latest['atr'].values[0]
            
            # Set stop loss and take profit based on ATR
            if prediction == 1:  # Long signal
                stop_loss = current_price - (atr * 1.5)
                take_profit = current_price + (atr * 2.5)
                action = 'buy'
            else:  # Short signal
                stop_loss = current_price + (atr * 1.5)
                take_profit = current_price - (atr * 2.5)
                action = 'sell'
            
            return {
                'action': action,
                'symbol': symbol,
                'price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': float(confidence),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None

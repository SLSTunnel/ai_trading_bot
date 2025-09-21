from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    mt5_login = db.Column(db.String(50), nullable=False)
    mt5_password = db.Column(db.String(100), nullable=False)
    mt5_server = db.Column(db.String(100), nullable=False)
    symbols = db.Column(db.String(255), nullable=False)
    timeframe = db.Column(db.String(10), default='H1')
    risk_percent = db.Column(db.Float, default=1.0)
    lot_size = db.Column(db.Float, default=0.1)
    max_drawdown = db.Column(db.Float, default=2.0)
    trading_active = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'mt5': {
                'login': self.mt5_login,
                'password': '********' if self.mt5_password else '',
                'server': self.mt5_server
            },
            'trading': {
                'symbols': self.symbols.split(','),
                'timeframe': self.timeframe,
                'max_risk_percent': self.risk_percent,
                'lot_size': self.lot_size,
                'max_drawdown': self.max_drawdown,
                'trading_active': self.trading_active
            }
        }

import os
import json
import threading
import time
from datetime import datetime
import os
from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
import pandas as pd
import plotly.express as px
import plotly.utils
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import MetaTrader5 as mt5
from models import db, Settings

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///trading_bot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Create database tables
with app.app_context():
    db.create_all()

# Initialize Dash app
dash_app = dash.Dash(
    __name__,
    server=app,
    url_base_pathname='/dashboard/',
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

def get_or_create_settings():
    """Get settings from database or create default"""
    settings = Settings.query.first()
    if not settings:
        # Create default settings
        settings = Settings(
            mt5_login='',
            mt5_password='',
            mt5_server='',
            symbols='EURUSD,GBPUSD,USDJPY',
            timeframe='H1',
            risk_percent=1.0,
            lot_size=0.1,
            max_drawdown=2.0,
            trading_active=False
        )
        db.session.add(settings)
        db.session.commit()
    return settings

# Global variables
trading_active = True
trades = []
balance_history = []

# Initialize MT5
if not mt5.initialize():
    print("MT5 initialization failed")
    mt5.shutdown()

# Connect to MT5 account
if not mt5.login(
    login=config['mt5']['login'],
    password=config['mt5']['password'],
    server=config['mt5']['server']
):
    print("MT5 login failed")
    mt5.shutdown()

# Dashboard layout
dash_app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("AI Trading Bot Dashboard"), className="text-center my-4")
    ]),
    
    # Stats Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Balance", className="card-title"),
                    html.H2("$0.00", id="balance-display")
                ])
            ], className="mb-4")
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Active Trades", className="card-title"),
                    html.H2("0", id="active-trades")
                ])
            ], className="mb-4")
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Win Rate", className="card-title"),
                    html.H2("0%", id="win-rate")
                ])
            ], className="mb-4")
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Daily P/L", className="card-title"),
                    html.H2("$0.00", id="daily-pl")
                ])
            ], className="mb-4")
        ], md=3)
    ]),
    
    # Control Panel and Trades Table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Trading Controls"),
                dbc.CardBody([
                    dbc.Button(
                        "Stop Trading", 
                        id="toggle-trading", 
                        color="danger", 
                        className="w-100 mb-3"
                    ),
                    dbc.Button(
                        "Close All Trades", 
                        id="close-trades", 
                        color="warning", 
                        className="w-100"
                    )
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Trading Pairs"),
                dbc.CardBody([
                    html.Div(id="symbols-list")
                ])
            ])
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Active Trades"),
                dbc.CardBody([
                    html.Div(id="trades-table")
                ])
            ])
        ], md=9)
    ]),
    
    # Charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Balance History"),
                dbc.CardBody([
                    dcc.Graph(id="balance-chart")
                ])
            ], className="mt-4")
        ])
    ])
], fluid=True)

# Update dashboard data
@socketio.on('update_dashboard')
def handle_update():
    # Get account info
    account_info = mt5.account_info()
    balance = account_info.balance if account_info else 0
    
    # Get open positions
    positions = mt5.positions_get()
    active_trades = len(positions) if positions else 0
    
    # Calculate win rate (simplified)
    win_rate = calculate_win_rate()
    
    # Update dashboard
    socketio.emit('dashboard_data', {
        'balance': f"${balance:.2f}",
        'active_trades': active_trades,
        'win_rate': f"{win_rate}%",
        'daily_pl': calculate_daily_pl()
    })
    
    # Update trades table
    update_trades_table(positions)
    
    # Update balance history
    update_balance_history(balance)

def calculate_win_rate():
    # Implement your win rate calculation logic here
    # This is a placeholder
    return 75.5

def calculate_daily_pl():
    # Implement daily P/L calculation
    return "+$125.50"

def update_trades_table(positions):
    # Format positions data for display
    trades_data = []
    if positions:
        for position in positions:
            trades_data.append({
                'symbol': position.symbol,
                'type': 'Buy' if position.type == 0 else 'Sell',
                'volume': position.volume,
                'open_price': position.price_open,
                'current_price': position.price_current,
                'profit': position.profit,
                'ticket': position.ticket
            })
    
    socketio.emit('update_trades', {'trades': trades_data})

def update_balance_history(balance):
    global balance_history
    
    # Add current balance to history
    balance_history.append({
        'time': datetime.now().strftime('%H:%M:%S'),
        'balance': balance
    })
    
    # Keep only last 100 data points
    if len(balance_history) > 100:
        balance_history = balance_history[-100:]
    
    # Update chart
    update_balance_chart()

def update_balance_chart():
    if not balance_history:
        return
    
    df = pd.DataFrame(balance_history)
    fig = px.line(df, x='time', y='balance', title='Account Balance History')
    fig.update_layout(showlegend=False)
    
    socketio.emit('update_chart', {'chart': fig.to_dict()})

# API Endpoints
@app.route('/')
def index():
    settings = get_or_create_settings()
    
    # Get the current trading status and other data
    current_status = {
        'trading_active': settings.trading_active,
        'balance': session.get('balance', 0),
        'open_trades': len(trades) if trades else 0,
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return render_template('index.html', status=current_status)

@app.route('/settings')
def settings_page():
    return render_template('settings.html')

@app.route('/api/toggle_trading', methods=['POST'])
def toggle_trading():
    settings = get_or_create_settings()
    settings.trading_active = not settings.trading_active
    db.session.commit()
    
    return jsonify({
        'status': 'success',
        'trading_active': settings.trading_active
    })

@app.route('/api/close_all_trades', methods=['POST'])
def close_all_trades():
    # Implement close all trades logic
    return jsonify({
        'status': 'success',
        'message': 'All open trades have been closed'
    })

@app.route('/settings')
def settings_page():
    return render_template('settings.html')

# Start the dashboard
def run_dashboard():
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def initialize_mt5():
    """Initialize MT5 connection with current config"""
    global mt5_initialized
    
    if not mt5.initialize():
        print("MT5 initialization failed")
        mt5.shutdown()
        return False
    
    # Get config from session or file
    current_config = session.get('config', load_config())
    
    # Connect to MT5 account
    if not mt5.login(
        login=current_config['mt5']['login'],
        password=current_config['mt5']['password'],
        server=current_config['mt5']['server']
    ):
        print("MT5 login failed")
        mt5.shutdown()
        return False
    
    mt5_initialized = True
    return True

def run():
    # Initialize MT5 connection
    if not initialize_mt5():
        print("Failed to initialize MT5. Please check your credentials in the settings.")
    
    # Start dashboard in a separate thread
    dashboard_thread = threading.Thread(target=run_dashboard)
    dashboard_thread.daemon = True
    dashboard_thread.start()
    
    # Start the main app
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)

if __name__ == '__main__':
    run()

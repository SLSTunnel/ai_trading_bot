import os
import json
import shutil
from datetime import datetime
from flask import Blueprint, request, jsonify, session, current_app
from models import db, Settings
import MetaTrader5 as mt5

settings_bp = Blueprint('settings', __name__)
CONFIG_BACKUP_DIR = 'instance/backups'

def create_backup(settings):
    """Create a backup of the current settings"""
    if not os.path.exists(CONFIG_BACKUP_DIR):
        os.makedirs(CONFIG_BACKUP_DIR)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = os.path.join(CONFIG_BACKUP_DIR, f'settings_backup_{timestamp}.json')
    
    backup_data = {
        'mt5': {
            'login': settings.mt5_login,
            'server': settings.mt5_server
        },
        'trading': {
            'symbols': settings.symbols,
            'timeframe': settings.timeframe,
            'risk_percent': settings.risk_percent,
            'lot_size': settings.lot_size,
            'max_drawdown': settings.max_drawdown,
            'trading_active': settings.trading_active
        },
        'backup_timestamp': timestamp
    }
    
    with open(backup_file, 'w') as f:
        json.dump(backup_data, f, indent=2)
    
    return backup_file

def validate_mt5_credentials(login, password, server):
    """Validate MT5 credentials by attempting to log in"""
    if not mt5.initialize():
        return False, "Failed to initialize MT5"
    
    try:
        authorized = mt5.login(login=int(login), password=password, server=server)
        if not authorized:
            return False, f"MT5 login failed: {mt5.last_error()}"
        return True, "Credentials are valid"
    except Exception as e:
        return False, f"Error validating credentials: {str(e)}"
    finally:
        mt5.shutdown()

@settings_bp.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current settings"""
    try:
        settings = Settings.query.first()
        if not settings:
            return jsonify({
                'status': 'error',
                'message': 'No settings found. Please configure the bot first.'
            }), 404
            
        return jsonify({
            'status': 'success',
            'settings': settings.to_dict()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to load settings: {str(e)}'
        }), 500

@settings_bp.route('/api/settings', methods=['POST'])
def update_settings():
    """Update settings and reconnect to MT5"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['mt5_login', 'mt5_password', 'mt5_server', 'symbols', 'risk_percent', 'timeframe']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required field: {field}'
                }), 400
                
        # Get or create settings
        settings = Settings.query.first()
        if not settings:
            settings = Settings()
            db.session.add(settings)
        
        # Create backup before updating
        backup_file = create_backup(settings)
        
        # Update settings
        settings.mt5_login = data['mt5_login']
        settings.mt5_password = data['mt5_password']
        settings.mt5_server = data['mt5_server']
        settings.symbols = ','.join([s.strip().upper() for s in data['symbols'].split(',')])
        settings.timeframe = data['timeframe']
        settings.risk_percent = float(data['risk_percent'])
        
        # Optional fields with defaults
        if 'lot_size' in data:
            settings.lot_size = float(data['lot_size'])
        if 'max_drawdown' in data:
            settings.max_drawdown = float(data['max_drawdown'])
        
        # Validate MT5 credentials
        is_valid, message = validate_mt5_credentials(
            settings.mt5_login,
            settings.mt5_password,
            settings.mt5_server
        )
        
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': f'MT5 validation failed: {message}'
            }), 400
            
        # Save changes
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Settings updated and MT5 reconnected successfully',
            'backup': backup_file,
            'reload': True,  # Tell frontend to reload the page
            'settings': settings.to_dict()
        })
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': f'Invalid input: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to update settings: {str(e)}'
        }), 500

# Add this to your main app.py or dashboard.py:
# from settings_api import settings_bp
# app.register_blueprint(settings_bp)

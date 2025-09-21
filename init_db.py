from dashboard import app, db
from models import Settings

def init_db():
    with app.app_context():
        # Create all database tables
        db.create_all()
        
        # Create default settings if they don't exist
        if not Settings.query.first():
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
            print("Database initialized with default settings.")
        else:
            print("Database already initialized.")

if __name__ == '__main__':
    init_db()

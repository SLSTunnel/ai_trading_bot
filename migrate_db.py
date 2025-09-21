from flask_migrate import Migrate
from dashboard import app, db

migrate = Migrate(app, db)

if __name__ == '__main__':
    with app.app_context():
        # Create all database tables
        db.create_all()
        print("Database tables created.")
        
        # Initialize Alembic for migrations
        from flask_migrate import upgrade as _upgrade, init as _init, migrate as _migrate, stamp as _stamp
        
        # Initialize the migration repository
        _init()
        
        # Create an initial migration
        _migrate(message='Initial migration')
        
        # Apply the migration
        _upgrade()
        
        # Stamp the database with the current migration
        _stamp()
        
        print("Database migration completed.")

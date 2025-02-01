from sqlalchemy import create_engine, text
import os

def migrate_database():
    DB_URL = os.getenv("DB_URL")
    if not DB_URL:
        raise ValueError("Database URL not found in environment variables")
    
    engine = create_engine(DB_URL)
    
    # SQL commands to add missing columns
    migration_commands = [
        """
        ALTER TABLE agents
        ADD COLUMN IF NOT EXISTS pdf_urls VARCHAR[] DEFAULT '{}',
        ADD COLUMN IF NOT EXISTS website_urls VARCHAR[] DEFAULT '{}'
        """,
    ]
    
    with engine.connect() as connection:
        for command in migration_commands:
            connection.execute(text(command))
            connection.commit()

if __name__ == "__main__":
    try:
        migrate_database()
        print("Migration completed successfully")
    except Exception as e:
        print(f"Migration failed: {str(e)}")
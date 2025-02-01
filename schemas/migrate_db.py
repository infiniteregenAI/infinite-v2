from sqlalchemy import create_engine, text
import os

def migrate_database():
    DB_URL = os.getenv("DB_URL")
    if not DB_URL:
        raise ValueError("Database URL not found in environment variables")
    
    engine = create_engine(DB_URL)
    
    # SQL commands for migrating instructions to array type
    migration_commands = [
        """
        -- First create temporary column
        ALTER TABLE agents 
        ADD COLUMN IF NOT EXISTS instructions_array VARCHAR[] DEFAULT '{}'
        """,
        
        """
        -- Copy data from existing instructions to array format
        UPDATE agents 
        SET instructions_array = string_to_array(instructions, E'\n')
        WHERE instructions IS NOT NULL AND instructions != ''
        """,
        
        """
        -- Drop the old column
        ALTER TABLE agents 
        DROP COLUMN IF EXISTS instructions
        """,
        
        """
        -- Rename the new array column
        ALTER TABLE agents 
        RENAME COLUMN instructions_array TO instructions
        """,
        
        """
        -- Set the default value
        ALTER TABLE agents 
        ALTER COLUMN instructions SET DEFAULT '{}'
        """
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
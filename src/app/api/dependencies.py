from psycopg2.extensions import connection
from fastapi import Depends
from app.services.db.connection import get_db_connection
import psycopg2

def get_db() -> connection:
    """
    FastAPI dependency for database connection.
    """
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()
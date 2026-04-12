import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
import os

load_dotenv()

def get_db_connection():
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost")
    )

    cur = conn.cursor()
    cur.execute("SET search_path TO public, extensions;")
    cur.close()

    register_vector(conn)
    return conn
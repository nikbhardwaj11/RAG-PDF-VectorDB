"""
Simple database connection test.
Run this to verify PostgreSQL + pgvector is working.
"""
from database import get_connection

try:
    conn = get_connection()
    print("[OK] PostgreSQL connected successfully!")

    with conn.cursor() as cur:
        cur.execute("SELECT version();")
        version = cur.fetchone()
        print(f"Database version: {version[0]}")

        # Check pgvector extension
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        if cur.fetchone():
            print("[OK] pgvector extension is installed")
        else:
            print("[WARN] pgvector extension not found - run init_db() to create it")

    conn.close()
    print("[OK] Connection test passed!")

except Exception as e:
    print(f"[FAIL] Database connection failed: {e}")
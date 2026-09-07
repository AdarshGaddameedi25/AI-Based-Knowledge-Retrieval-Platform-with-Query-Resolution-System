import os
import psycopg2
from urllib.parse import urlparse
from config.settings import settings


def check_postgres():
    db_url = os.getenv("DATABASE_URL", settings.database_url)
    parsed = urlparse(db_url)
    target_db = parsed.path.lstrip("/") or "ai_knowledge_retrieval"
    user_pass = f"{parsed.username}:{parsed.password}@" if parsed.username and parsed.password else ""
    port = parsed.port or 5432
    host = parsed.hostname or "localhost"

    postgres_url = f"postgresql://{user_pass}{host}:{port}/postgres"

    conn = psycopg2.connect(postgres_url)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT version()")
    print("Version:", cur.fetchone()[0])

    cur.execute(f"SELECT 1 FROM pg_database WHERE datname='{target_db}'")
    exists = cur.fetchone() is not None
    print(f"{target_db} DB exists:", exists)
    if not exists:
        cur.execute(f"CREATE DATABASE {target_db}")
        print(f"Created {target_db} database")
    cur.close()
    conn.close()

    conn2 = psycopg2.connect(db_url)
    conn2.autocommit = True
    cur2 = conn2.cursor()
    try:
        cur2.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("Extension vector is available and enabled!")
    except Exception as e:
        print("Extension vector error:", e)
    cur2.close()
    conn2.close()


if __name__ == "__main__":
    check_postgres()


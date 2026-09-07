import os
import sys
from urllib.parse import urlparse
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings

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
cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{target_db}'")
exists = cur.fetchone()
if not exists:
    cur.execute(f"CREATE DATABASE {target_db}")
    print(f"Created database: {target_db}")
else:
    print(f"Database already exists: {target_db}")

cur.execute("SELECT version()")
print("PostgreSQL:", cur.fetchone()[0])
conn.close()


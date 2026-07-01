import sqlite3

DB_PATH = r"D:\Intern\medsight-ai\backend\pneumoscan\pneumoscan.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# check current columns
cur.execute("PRAGMA table_info(scans);")
columns = [row[1] for row in cur.fetchall()]
print("Current columns in 'scans':", columns)

if "symptoms" not in columns:
    cur.execute("ALTER TABLE scans ADD COLUMN symptoms TEXT;")
    conn.commit()
    print("Added 'symptoms' column.")
else:
    print("'symptoms' column already exists — issue must be elsewhere (wrong DB file, cached connection, etc).")

# re-check to confirm
cur.execute("PRAGMA table_info(scans);")
columns_after = [row[1] for row in cur.fetchall()]
print("Columns after fix:", columns_after)

conn.close()

"""DB migration: add v2 enrichment columns."""
import sqlite3

conn = sqlite3.connect("ollama.db")
cols = [row[1] for row in conn.execute("PRAGMA table_info(model)").fetchall()]

migrations = {
    "creator_org":      "ALTER TABLE model ADD COLUMN creator_org TEXT",
    "is_multimodal":    "ALTER TABLE model ADD COLUMN is_multimodal INTEGER",
    "huggingface_url":  "ALTER TABLE model ADD COLUMN huggingface_url TEXT",
    "benchmark_scores": "ALTER TABLE model ADD COLUMN benchmark_scores TEXT",
    "parameter_sizes":  "ALTER TABLE model ADD COLUMN parameter_sizes TEXT",
}

for col, sql in migrations.items():
    if col not in cols:
        conn.execute(sql)
        print(f"  Added: {col}")
    else:
        print(f"  Already exists: {col}")

conn.commit()
conn.close()
final = [row[1] for row in sqlite3.connect("ollama.db").execute("PRAGMA table_info(model)").fetchall()]
print(f"\nTotal columns: {len(final)}")
print("Done!")

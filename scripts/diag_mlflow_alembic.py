#!/usr/bin/env python3
"""Diagnose MLflow Alembic migration conflict."""
import os
import re
import sqlite3

# Path to MLflow migrations
migrations_path = '/Users/davidcertan/miniforge3/envs/intel/lib/python3.12/site-packages/mlflow/store/db_migrations/versions/'

# Find the head revision (the one with down_revision = None)
head_revision = None
all_revisions = {}

for filename in os.listdir(migrations_path):
    if filename.endswith('.py') and not filename.startswith('__'):
        filepath = os.path.join(migrations_path, filename)
        with open(filepath, 'r') as f:
            content = f.read()
            rev_match = re.search(r'revision = "([^"]+)"', content)
            down_match = re.search(r'down_revision = "([^"]+)"', content)
            down_none = 'down_revision = None' in content
            if rev_match:
                rev_id = rev_match.group(1)
                down_id = down_match.group(1) if down_match else None
                all_revisions[rev_id] = down_id
                if down_none:
                    head_revision = rev_id

print('=== MLflow 2.10.0 Migration Chain ===')
print(f'Head revision: {head_revision}')
print(f'Total revisions: {len(all_revisions)}')
print()
print('All revision IDs:')
for rev in sorted(all_revisions.keys()):
    print(f'  {rev}')

# Check database revision
print()
print('=== Database State ===')
db_paths = [
    '/Users/davidcertan/Desktop/ml_engine/trained_data/mlruns/mlflow.db',
    '/Users/davidcertan/Desktop/ml_engine/mlruns/mlflow.db'
]

for db_path in db_paths:
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT version_num FROM alembic_version")
            row = cursor.fetchone()
            db_rev = row[0] if row else 'None'
            print(f'{db_path}:')
            print(f'  Current revision: {db_rev}')
            print(f'  Is valid: {db_rev in all_revisions}')
        except sqlite3.OperationalError as e:
            print(f'{db_path}: Error - {e}')
        finally:
            conn.close()

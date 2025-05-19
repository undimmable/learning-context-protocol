#!/usr/bin/env python3
"""
Script to migrate data from memory.jsonl to the PostgreSQL database.

This script reads the memory.jsonl file and inserts each entry into the
memory_entries table in the database.
"""

import datetime as dt
import json
import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add the project root to the Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.db.model.memory_entry import MemoryEntry
from src.db.base import Base, engine

def migrate_jsonl_to_db():
    """Migrate data from memory.jsonl to the database."""
    # Create a session
    Session = sessionmaker(bind=engine)
    session = Session()

    # Path to the memory.jsonl file
    jsonl_path = Path(__file__).parent.parent / "tools" / "memory.jsonl"
    
    if not jsonl_path.exists():
        print(f"JSONL file not found at {jsonl_path}")
        return
    
    # Read the JSONL file and insert each entry into the database
    entries_migrated = 0
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
                
            try:
                entry = json.loads(line)
                
                # Parse the timestamp
                timestamp_str = entry.get('timestamp')
                if timestamp_str:
                    # Remove the 'Z' suffix if present
                    if timestamp_str.endswith('Z'):
                        timestamp_str = timestamp_str[:-1]
                    timestamp = dt.datetime.fromisoformat(timestamp_str)
                else:
                    # Use datetime.now(UTC) instead of utcnow() to avoid deprecation warning
                    timestamp = dt.datetime.now(dt.UTC)
                
                # Create a new MemoryEntry
                memory_entry = MemoryEntry(
                    text=entry.get('text', ''),
                    tags=entry.get('tags', []),
                    timestamp=timestamp
                )
                
                # Add to the session
                session.add(memory_entry)
                entries_migrated += 1
                
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {line}")
            except Exception as e:
                print(f"Error processing entry: {e}")
    
    # Commit the session
    try:
        session.commit()
        print(f"Successfully migrated {entries_migrated} entries to the database")
    except Exception as e:
        session.rollback()
        print(f"Error committing to database: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    # Ensure tables exist
    Base.metadata.create_all(bind=engine)
    
    # Migrate data
    migrate_jsonl_to_db()

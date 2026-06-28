import sqlite3
import os
from typing import Dict, Any, List, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "customer_memory.db")

def init_db():
    """
    Initializes the SQLite database tables.
    """
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table for storing high-level customer interactions (for long-term search/recall)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customer_interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_name TEXT NOT NULL,
        query TEXT,
        issue_type TEXT,
        department TEXT,
        response TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    conn.commit()
    conn.close()

def save_interaction(customer_name: str, query: str, issue_type: str, department: str, response: str):
    """
    Saves an interaction record to the SQLite database.
    """
    if not customer_name:
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO customer_interactions (customer_name, query, issue_type, department, response)
    VALUES (?, ?, ?, ?, ?)
    """, (customer_name.strip(), query, issue_type, department, response))
    conn.commit()
    conn.close()

def get_last_interaction(customer_name: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves the last customer interaction by customer name.
    """
    if not customer_name:
        return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Case-insensitive lookup
    cursor.execute("""
    SELECT query, issue_type, department, response, timestamp 
    FROM customer_interactions 
    WHERE LOWER(customer_name) = LOWER(?) 
    ORDER BY id DESC LIMIT 1
    """, (customer_name.strip(),))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "query": row["query"],
            "issue_type": row["issue_type"],
            "department": row["department"],
            "response": row["response"],
            "timestamp": row["timestamp"]
        }
    return None

if __name__ == "__main__":
    # Test memory script
    print("Testing SQLite Customer Memory...")
    init_db()
    save_interaction("David", "I have a billing issue.", "Refund request", "Billing", "Your refund is pending supervisor approval.")
    last = get_last_interaction("David")
    print(f"Retrieved for David: {last}")

-- Schema for customer_memory.db (Long-term interaction memory)

CREATE TABLE IF NOT EXISTS customer_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name TEXT NOT NULL,
    department TEXT NOT NULL,
    issue_type TEXT NOT NULL,
    customer_query TEXT NOT NULL,
    generated_response TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);


-- Schema for langgraph_checkpoints.db (LangGraph State Checkpointer)
-- Handled automatically by langgraph.checkpoint.sqlite.SqliteSaver:

CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    parent_id TEXT,
    checkpoint BLOB NOT NULL,
    metadata BLOB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS writes (
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    value BLOB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_id, task_id, idx)
);

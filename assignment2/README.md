# AI-Powered Customer Support Automation System

A robust, local AI-powered customer support automation system built using **LangGraph**, **LangChain**, and **SQLite**. The system automatically classifies customer intents, routes queries to specialized support agents (Sales, Technical, Billing, Account), retrieves contextual answers from local documentation via BM25 RAG, persists session state and interaction history, and implements a Human-in-the-Loop (HITL) approval gate for sensitive actions.

---

## 🏗️ System Workflow Architecture

The workflow is designed as a stateful graph where each node processes the state and returns updates.

```
                  [ START ]
                      │
                      ▼
       load_customer_info_and_history
                      │
                      ▼
               classify_intent
                      │
                      ▼
         [ Conditional Routing ]
         ├── Sales    ──► sales_agent ────┐
         ├── Technical ──► technical_agent ─┼──► [ supervisor_agent ] ──► [ END ]
         ├── Account  ──► account_agent ───┼─┘
         └── Billing  ──► billing_agent ───┘
                               │ (Refund / Cancel)
                               ▼
                        [ human_approval ] (HITL Interrupt)
                               │ (Resumed by Supervisor)
                               ▼
                       [ supervisor_agent ] ──► [ END ]
```

---

## 🛠️ Required Dependencies

All packages are listed in `requirements.txt`:
*   `langgraph`: Orchestrates the agent workflow state machine.
*   `langgraph-checkpoint-sqlite`: Provides SQLite state persistence for checkpointers (allowing pauses/interrupts).
*   `langchain-ollama`: Integrates the local LLM (`qwen2.5:3b`).
*   `rank-bm25`: Powers the local token-based document search index.
*   `aiosqlite`: Provides asynchronous SQLite operations.
*   `rich`: Produces a beautiful, styled console UI for tracking node execution.
*   `python-dotenv`: Manages environment variables.

---

## 🚀 Installation & Setup

### 1. Prerequisite: Install Ollama
Ensure you have [Ollama](https://ollama.com) installed and running locally. Pull the required model:
```bash
ollama pull qwen2.5:3b
```

### 2. Install Project Dependencies
Navigate to the assignment directory and run:
```bash
pip install -r requirements.txt
```

---

## 📖 Module Descriptions

*   **`workflow.py`**: Definess the `SupportState` schema, builds the state graph using `StateGraph`, implements intent classification, contains the specialized agents (Sales, Technical, Billing, Account), details conditional routing rules, and configures the Supervisor validation node.
*   **`rag.py`**: Loads company documents, splits them using a character splitter, builds a BM25 index, and performs source-filtered text retrieval.
*   **`memory.py`**: Initializes the SQLite database schemas and manages the saving and querying of customer history interactions.
*   **`demo.py`**: The main executable script. It clears previous database files for reproducibility, sets up a persistent thread session, feeds a sequence of 5 test queries, handles interactive terminal prompt pauses for human approval, and outputs styled UI panels using the `rich` library.
*   **`documents/`**: A folder containing raw text knowledge bases:
    *   `company_policy.txt`: Official company rules on refunds, cancellations, and escalations.
    *   `pricing_guide.txt`: Details on Standard, Professional, and Enterprise plans.
    *   `technical_manual.txt`: Troubleshooters for crashes and password reset guidelines.
    *   `faq_document.txt`: Frequently asked questions (locations, hours, etc.).

---

## 🔍 Key Implementations Explained

### 1. Local RAG Pipeline (BM25)
The system uses the `rank-bm25` algorithm to perform search-retrieval locally without requiring vector search databases or third-party embedding APIs:
*   **Loading & Splitting**: Plaintext documents are loaded and split into small chunks using `RecursiveCharacterTextSplitter`.
*   **Department-Specific Source Filtering**: When an agent requests context, it only retrieves pages from matching documents. For instance, the **Technical Agent** only queries `technical_manual.txt` and `faq_document.txt`. This keeps context sizes minimal and prevents interference.

### 2. SQLite State & Interaction Memory
The system runs two distinct levels of data storage:
*   **Checkpointer Persistence (`langgraph_checkpoints.db`)**: LangGraph's `SqliteSaver` records state history of the thread. This is what permits interrupting a run, modifying state variables, and resuming exactly where it stopped.
*   **Long-Term Memory (`customer_memory.db`)**: Stores the final resolved interaction payload (customer name, department, query, finalized response, timestamp). When a customer runs a "Memory Retrieval" query, this database is searched using their name to fetch what issue they previously filed.

### 3. Human-in-the-Loop (HITL) Gate
High-risk queries (Refunds or subscription cancellations) trigger an interrupt before executing the final action:
*   The intent classifier flags the issue as a refund request.
*   The billing agent drafts a refund ticket, setting `approval_status` to `"Pending"`.
*   The `StateGraph` is compiled with `interrupt_before=["human_approval"]`. When reaching this node, the graph halts.
*   `demo.py` displays the draft and prompts the user for action:
    *   **Approve (`1`)**: Changes state `approval_status` to `"Approved"`. The supervisor agent completes the refund and logs it.
    *   **Reject (`2`)**: Changes state `approval_status` to `"Rejected"`. The supervisor agent informs the user of denial.

---

## 🏃 How to Run the Demo

Run the command from the `assignment2` directory:
```bash
python demo.py
```

### Script Execution Flow
1. **Reset**: The script deletes any existing SQLite files on startup to ensure a clean demo.
2. **Query 1**: Asks about pricing (Routed to **Sales**, answers from `pricing_guide.txt`).
3. **Query 2**: Asks for password reset (Routed to **Account**, answers from manual).
4. **Query 3**: Reports application crash (Routed to **Technical**, troubleshooting rules retrieved).
5. **Query 4**: Requests a refund as **David** (Routed to **Billing**, halts at `human_approval` prompt).
    *   *Interactive:* Enter `1` to approve or `2` to reject the refund.
6. **Query 5**: Asks *"What was my previous support issue?"* (Routed to **Account**, pulls history for **David** from `customer_memory.db` and recalls the refund).

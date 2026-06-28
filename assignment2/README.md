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

*   **`workflow.py`**: Defines the `SupportState` schema, builds the state graph using `StateGraph`, implements intent classification, contains the specialized agents (Sales, Technical, Billing, Account), details conditional routing rules, and configures the Supervisor validation node.
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

### 1. Strict RAG Grounding (Single Source of Truth)
The agents treat the retrieved context as the absolute source of truth. No pricing tiers, refund eligibility window periods, or file size limits are hardcoded within agent prompts. 
*   **No Hallucinations/Prior Knowledge**: All agent prompts are explicitly constrained to prevent guess work, inference, or utilizing pretrained information.
*   **Strict Fallbacks**: If requested information is absent in the retrieved context, the agents are instructed to output verbatim: `"The requested information was not found in the company documentation."`
*   **Prohibition of Hallucinated Details**: Agents are strictly forbidden from fabricating contact details such as URLs, email addresses, phone numbers, websites, or support portals. They may only output them if they are explicitly present in the retrieved context.

### 2. Dynamic Source Selection (Password Reset)
In `account_agent()`, the system implements a dynamic context-sourcing guardrail:
*   If the query references password reset or login issues (e.g., contains keywords like `password`, `forgot password`, `reset password`, or `login issue`), the retrieval logic redirects to query the `["Technical Manual", "FAQ Document"]` documents.
*   Otherwise, it queries `["Company Policy Document", "FAQ Document"]`.

### 3. Customer Interaction Memory & Context Security
*   **Interaction Storage (`customer_memory.db`)**: Captures historical ticket logs securely.
*   **Anti-Leak Protection**: When running memory queries, the prompt and post-processing structures prevent raw SQL metadata, raw queries, database timestamps, or internal columns (like `Final Response`) from leaking into final responses. Output is cleanly formatted to show only:
    - Customer Greeting
    - Department
    - Issue Type
    - Date
    - Summarized resolution
*   **Programmatic Rule Stripping**: To prevent internal supervisor rule lists (e.g. `Rules: 1. If approval status is Approved...`) from leaking, the supervisor node applies a line filter that strips out leaked rules and system constraints.

### 4. Enterprise Identity Mapping
Any placeholders (such as `[Your Company Name]` or `[Your Name]`) are automatically resolved and replaced:
*   `[Your Company Name]` ➔ `ABC Technologies`
*   `[Your Name]` ➔ `ABC Technologies Support Team`
A safety filter programmatically replaces any brackets-enclosed values inside supervisor node outputs to guarantee bracketed placeholders are never exposed.

---

## 🏃 How to Run the Demo

Run the command from the `assignment2` directory:
```bash
python demo.py
```

### Script Execution Flow
1. **Reset**: The script deletes any existing SQLite files on startup to ensure a clean demo.
2. **Query 1**: Asks about pricing (Routed to **Sales**, answers from `pricing_guide.txt` and summarized dynamically).
3. **Query 2**: Asks for password reset (Routed to **Account**, dynamically queries the `Technical Manual` and manual steps).
4. **Query 3**: Reports application crash (Routed to **Technical**, troubleshooting rules retrieved from manual).
5. **Query 4**: Requests a refund as **David** (Routed to **Billing**, halts at `human_approval` prompt for approval).
    *   *Interactive:* Enter `1` to approve or `2` to reject the refund.
6. **Query 5**: Asks *"What was my previous support issue?"* (Routed to **Account**, pulls history for **David** from SQLite memory and summarizes the resolution securely without database context leakage).

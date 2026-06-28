# Screenshots Document Template

This document serves as a structured template for the required assignment screenshots. Once you capture the screenshots as per the guide in `screenshot_instructions.md`, save them in this directory and link/paste them below. You can then print/export this file as a PDF for submission.

---

## 🛡️ 1. Agent Routing
Shows the Intent Classification node routing a query (e.g. Query 1, 2, or 3) to the correct department agent.

*(Insert agent_routing.png here)*

---

## 🛡️ 2. Human Approval (Interrupt)
Shows the execution halting at the `human_approval` node when a billing refund request (Query 4) is initiated.

*(Insert human_approval_interrupt.png here)*

---

## 🛡️ 3. RAG Retrieval
Shows the generated response containing retrieved data from the local company documents (e.g., pricing guide tiers or file upload troubleshooting rules).

*(Insert rag_retrieval.png here)*

---

## 🛡️ 4. Memory Storage
Shows the SQLite `customer_interactions` table populated with the resolved ticket records.

*(Insert memory_storage.png here)*

---

## 🛡️ 5. Memory Recall
Shows the `load_customer_info_and_history` node fetching David's past interaction details from SQLite on Query 5.

*(Insert memory_recall.png here)*

---

## 🛡️ 6. Final Response Generation
Shows the finalized response (Query 5 output panel) addressing David's query about his previous ticket.

*(Insert final_response.png here)*

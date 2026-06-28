import os
import sqlite3
from typing import TypedDict, List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Import RAG and Memory modules
import rag
import memory

# Ensure memory DB is initialized
memory.init_db()

# Define the state structure
class SupportState(TypedDict):
    customer_info: Dict[str, Any]
    customer_query: str
    issue_type: str
    department: str
    retrieved_context: str
    approval_status: str
    generated_response: str
    conversation_history: List[Dict[str, str]]

# Initialize LLM (using the local qwen2.5:3b model with temperature 0.0)
llm = ChatOllama(model="qwen2.5:3b", temperature=0.0)

# ==========================================
# 1. Customer Info & SQLite History Node
# ==========================================
def load_customer_info_and_history(state: SupportState) -> SupportState:
    """
    Initializes customer info, extracts names, and checks SQLite memory for past interactions.
    """
    query = state.get("customer_query", "")
    cust_info = state.get("customer_info", {})
    if not cust_info:
        cust_info = {}
    
    # Try to extract name if not present
    current_name = cust_info.get("name")
    if not current_name or current_name == "Unknown":
        # Prompt LLM to extract customer name
        name_prompt = f"""Extract the customer name from the following query. 
If a person's name is explicitly mentioned (e.g. "My name is David", "This is David", "Hi, I'm Sarah"), output ONLY the first name.
If no name is mentioned, output ONLY the word "Unknown".

Query: {query}
Customer Name:"""
        res = llm.invoke(name_prompt)
        extracted_name = res.content.strip().replace('"', '').replace("'", "")
        if extracted_name.lower() in ["unknown", "none", "no name", ""]:
            cust_info["name"] = "Unknown"
        else:
            cust_info["name"] = extracted_name
    
    state["customer_info"] = cust_info
    name = cust_info.get("name", "Unknown")
    
    # Check if this query is requesting previous issue/history
    history_keywords = ["previous issue", "previous support issue", "past issue", "past support", "what was my previous", "what was my last"]
    is_history_query = any(kw in query.lower() for kw in history_keywords)
    
    if is_history_query:
        state["issue_type"] = "Memory Retrieval"
        state["department"] = "Account" # Route to Account support by default for account history
        
        # Load previous interactions from SQLite
        if name != "Unknown":
            last_interaction = memory.get_last_interaction(name)
            if last_interaction:
                state["retrieved_context"] = (
                    f"Previous interaction found in SQLite database:\n"
                    f"- Query: '{last_interaction['query']}'\n"
                    f"- Department: {last_interaction['department']}\n"
                    f"- Issue Type: {last_interaction['issue_type']}\n"
                    f"- Final Response: '{last_interaction['response']}'\n"
                    f"- Timestamp: {last_interaction['timestamp']}"
                )
            else:
                state["retrieved_context"] = f"No previous interactions found in SQLite database for customer '{name}'."
        else:
            state["retrieved_context"] = "Cannot retrieve previous issues because the customer name is unknown."
            
    return state

# ==========================================
# 2. Intent Classification Node
# ==========================================
def classify_intent(state: SupportState) -> SupportState:
    """
    Classifies the user query into a department and checks if it requires Human approval.
    """
    # If already set as Memory Retrieval, skip classification
    if state.get("issue_type") == "Memory Retrieval":
        return state
        
    query = state.get("customer_query", "")
    
    classify_prompt = f"""You are a customer support query classifier. Your job is to analyze the customer query and classify it into:
1. Department: Must be exactly one of: Sales, Technical Support, Billing, Account.
2. Issue Type: Must be exactly one of: "Refund request", "Subscription cancellation", "Account closure request", "Compensation request", "Escalation to management", or "Normal query".

Rules for Department:
- Sales: pricing, plans, features, licenses, discounts, purchasing.
- Billing: payment issues, billing cycles, invoices, refunds, charge questions.
- Technical Support: crashes, uploads, errors, bugs, app crashes.
- Account: password reset, logins, profile, account closure.

Rules for Issue Type:
- Refund request: customer explicitly asks for a refund.
- Subscription cancellation: customer asks to cancel their subscription.
- Account closure request: customer asks to close or delete their account.
- Compensation request: customer asks for credits, refunds for downtime, or compensation.
- Escalation to management: customer asks to escalate, speak to a manager, or file a complaint.
- Normal query: any other general support question.

Query: {query}

Format your output EXACTLY as:
Department: <department>
Issue Type: <issue_type>
"""
    res = llm.invoke(classify_prompt)
    output = res.content.strip()
    
    # Parse output lines
    dept = "Sales"
    issue = "Normal query"
    
    for line in output.split("\n"):
        if line.lower().startswith("department:"):
            dept = line.split(":", 1)[1].strip()
        elif line.lower().startswith("issue type:") or line.lower().startswith("issue_type:"):
            issue = line.split(":", 1)[1].strip()
            
    # Normalize values
    valid_departments = ["Sales", "Technical Support", "Billing", "Account"]
    if dept not in valid_departments:
        # Fallback based on keywords
        if "pricing" in query.lower() or "cost" in query.lower() or "price" in query.lower():
            dept = "Sales"
        elif "refund" in query.lower() or "billing" in query.lower() or "payment" in query.lower() or "charged" in query.lower():
            dept = "Billing"
        elif "password" in query.lower() or "account" in query.lower() or "login" in query.lower():
            dept = "Account"
        else:
            dept = "Technical Support"
            
    # Clean up issue type
    valid_issues = ["Refund request", "Subscription cancellation", "Account closure request", "Compensation request", "Escalation to management", "Normal query"]
    normalized_issue = "Normal query"
    for v in valid_issues:
        if v.lower() in issue.lower():
            normalized_issue = v
            break
            
    # Check if Human-in-the-Loop is required
    # Required for: Refund requests, Subscription cancellation, Account closure requests, Compensation requests, Escalation to management
    approval_issues = ["Refund request", "Subscription cancellation", "Account closure request", "Compensation request", "Escalation to management"]
    
    state["department"] = dept
    state["issue_type"] = normalized_issue
    
    if normalized_issue in approval_issues:
        state["approval_status"] = "Pending"
    else:
        state["approval_status"] = "Not Required"
        
    return state

# ==========================================
# 3. Department Agents (Specialized Support)
# ==========================================
def sales_agent(state: SupportState) -> SupportState:
    """
    Sales Agent: Retrieves context from Pricing Guide & FAQ, drafts a sales response.
    """
    query = state.get("customer_query", "")
    # Retrieve context from specific Sales sources
    context = rag.get_context_for_sources(query, ["Pricing Guide", "FAQ Document"])
    state["retrieved_context"] = context
    
    prompt = f"""You are a Sales Support Agent. Answer the customer's query using the retrieved context.
If the answer is in the context, be specific and quote pricing or features accurately.
Be professional, friendly, and helpful.

Retrieved Context:
{context}

Customer Query:
{query}

Sales Agent Draft Response:"""
    res = llm.invoke(prompt)
    state["generated_response"] = res.content.strip()
    return state

def technical_agent(state: SupportState) -> SupportState:
    """
    Technical Support Agent: Retrieves context from Technical Manual & FAQ, drafts technical response.
    """
    query = state.get("customer_query", "")
    context = rag.get_context_for_sources(query, ["Technical Manual", "FAQ Document"])
    state["retrieved_context"] = context
    
    prompt = f"""You are a Technical Support Agent. Answer the customer's technical query using the retrieved manual pages and FAQ.
Provide clear step-by-step troubleshooting instructions from the manual if applicable.
Be professional, highly precise, and polite.

Retrieved Context:
{context}

Customer Query:
{query}

Technical Agent Draft Response:"""
    res = llm.invoke(prompt)
    state["generated_response"] = res.content.strip()
    return state

def billing_agent(state: SupportState) -> SupportState:
    """
    Billing Support Agent: Retrieves context from Pricing Guide & Company Policy, drafts billing response.
    """
    query = state.get("customer_query", "")
    context = rag.get_context_for_sources(query, ["Pricing Guide", "Company Policy Document"])
    state["retrieved_context"] = context
    
    prompt = f"""You are a Billing Support Agent. Answer the customer's billing query using the pricing guide and policy context.
Be clear about subscription policies, billing cycles, and refund eligibility.

Retrieved Context:
{context}

Customer Query:
{query}

Billing Agent Draft Response:"""
    res = llm.invoke(prompt)
    state["generated_response"] = res.content.strip()
    return state

def account_agent(state: SupportState) -> SupportState:
    """
    Account Support Agent: Retrieves context from Company Policy & FAQ, drafts account response.
    Can also handle memory lookup answers.
    """
    query = state.get("customer_query", "")
    
    # If this is a SQLite history query, retrieved_context is already populated in load_customer_info_and_history
    if state.get("issue_type") == "Memory Retrieval":
        context = state.get("retrieved_context", "")
        prompt = f"""You are an Account Support Agent. The customer is asking about their previous support issue.
Using the retrieved interaction from our SQLite memory database, inform the customer about their last ticket details.
Summarize what their previous issue was, which department handled it, and the resolution they received.

Retrieved History Context:
{context}

Customer Query:
{query}

Account Agent History Response:"""
    else:
        context = rag.get_context_for_sources(query, ["Company Policy Document", "FAQ Document"])
        state["retrieved_context"] = context
        prompt = f"""You are an Account Support Agent. Answer the customer's account query (e.g. password resets, security, account closure) using the company policies and FAQ.
Provide clear instructions or status details.

Retrieved Context:
{context}

Customer Query:
{query}

Account Agent Draft Response:"""
        
    res = llm.invoke(prompt)
    state["generated_response"] = res.content.strip()
    return state

# ==========================================
# 4. Human-in-the-Loop Node
# ==========================================
def human_approval(state: SupportState) -> SupportState:
    """
    A placeholder node that represents the Human approval decision step.
    The graph halts before this node using LangGraph interrupts.
    When resumed, the supervisor's input (Approved/Rejected) is processed here.
    """
    # This node is a pass-through during resumption, as the state will have been modified by the supervisor.
    return state

# ==========================================
# 5. Supervisor Agent Node
# ==========================================
def supervisor_agent(state: SupportState) -> SupportState:
    """
    Supervisor Agent:
    - Validates generated responses.
    - Improves response quality.
    - Produces the final customer response.
    - Saves the final interaction to SQLite database memory.
    """
    query = state.get("customer_query", "")
    draft = state.get("generated_response", "")
    issue = state.get("issue_type", "Normal query")
    dept = state.get("department", "Account")
    approval = state.get("approval_status", "Not Required")
    name = state.get("customer_info", {}).get("name", "Unknown")
    
    # Prompt the supervisor LLM to review, validate, and refine the response draft
    supervisor_prompt = f"""You are a Customer Support Supervisor. Your role is to review and finalize the agent's draft response to a customer query.

Customer Name: {name}
Customer Query: {query}
Department: {dept}
Issue Type: {issue}
Approval Status: {approval}
Agent Draft Response: {draft}

Rules:
1. If the approval status is 'Approved', ensure the final response confirms that the supervisor has reviewed and approved the action (e.g. 'This request has been approved by our supervisor management team...').
2. If the approval status is 'Rejected', rewrite the response to politely decline the request (e.g., 'Refund requests are only processed within 14 days of purchase. Your request could not be approved because...').
3. If no approval was required, refine the draft response for clarity, tone, and professional grammar, maintaining the same factual information. Do not change facts.
4. Output ONLY the final customer response. Do not add metadata, labels, or extra text.

Final Response:"""
    
    res = llm.invoke(supervisor_prompt)
    final_response = res.content.strip()
    
    # Save the final response to state
    state["generated_response"] = final_response
    
    # Save the interaction to SQLite Customer Memory for future retrieval
    if name != "Unknown" and issue != "Memory Retrieval":
        memory.save_interaction(
            customer_name=name,
            query=query,
            issue_type=issue,
            department=dept,
            response=final_response
        )
        
    return state

# ==========================================
# 6. Graph Routing Logic
# ==========================================
def route_by_department(state: SupportState) -> str:
    """
    Routes the workflow to the appropriate specialized department agent.
    """
    dept = state.get("department", "Account")
    if dept == "Sales":
        return "sales_agent"
    elif dept == "Technical Support":
        return "technical_agent"
    elif dept == "Billing":
        return "billing_agent"
    else:
        return "account_agent"

def route_after_agent(state: SupportState) -> str:
    """
    Routes the workflow to either the human approval loop or directly to the supervisor agent.
    """
    approval = state.get("approval_status", "Not Required")
    if approval == "Pending":
        return "human_approval"
    else:
        return "supervisor_agent"

# ==========================================
# 7. Build and Compile the Graph
# ==========================================
builder = StateGraph(SupportState)

# Add Nodes
builder.add_node("load_customer_info_and_history", load_customer_info_and_history)
builder.add_node("classify_intent", classify_intent)
builder.add_node("sales_agent", sales_agent)
builder.add_node("technical_agent", technical_agent)
builder.add_node("billing_agent", billing_agent)
builder.add_node("account_agent", account_agent)
builder.add_node("human_approval", human_approval)
builder.add_node("supervisor_agent", supervisor_agent)

# Set Entry Point
builder.set_entry_point("load_customer_info_and_history")

# Connect Initialization to Classification
builder.add_edge("load_customer_info_and_history", "classify_intent")

# Route conditionally from classification to department agents
builder.add_conditional_edges(
    "classify_intent",
    route_by_department,
    {
        "sales_agent": "sales_agent",
        "technical_agent": "technical_agent",
        "billing_agent": "billing_agent",
        "account_agent": "account_agent"
    }
)

# Route from all department agents using route_after_agent
for agent in ["sales_agent", "technical_agent", "billing_agent", "account_agent"]:
    builder.add_conditional_edges(
        agent,
        route_after_agent,
        {
            "human_approval": "human_approval",
            "supervisor_agent": "supervisor_agent"
        }
    )

# Connect Human Approval to Supervisor
builder.add_edge("human_approval", "supervisor_agent")

# Supervisor Agent is the terminal node
builder.add_edge("supervisor_agent", END)

# Use SQLite connection for LangGraph checkpointing
# This saves the graph state persistently, enabling interrupts and resumptions
checkpoint_conn = sqlite3.connect("langgraph_checkpoints.db", check_same_thread=False)
graph_memory = SqliteSaver(checkpoint_conn)

# Compile the graph
# We interrupt_before human_approval to pause for supervisor input
graph = builder.compile(
    checkpointer=graph_memory,
    interrupt_before=["human_approval"]
)

# Export the graph compiled object
def get_graph():
    return graph

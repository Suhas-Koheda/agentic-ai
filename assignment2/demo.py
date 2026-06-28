import os
import sys
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.markup import escape

# Delete old database files for a clean reproducible demo BEFORE importing workflow
db_paths = ["customer_memory.db", "langgraph_checkpoints.db"]
for path in db_paths:
    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if os.path.exists(full_path):
        try:
            os.remove(full_path)
        except Exception:
            pass

# Add current folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from workflow import get_graph
import memory

console = Console()

def run_query(query: str, thread_id: str, is_first: bool = False):
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    
    console.print(f"\n[bold magenta]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold magenta]")
    console.print(f"[bold cyan]📥 Customer Input:[/bold cyan] [italic white]\"{escape(query)}\"[/italic white]")
    console.print(f"[bold magenta]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold magenta]")
    
    # 1. Start or resume execution
    # Check if thread already has state to prevent overwriting persistent variables
    state_snapshot = graph.get_state(config)
    if not state_snapshot.values:
        inputs = {
            "customer_query": query,
            "customer_info": {"name": "Unknown"},
            "conversation_history": [],
            "issue_type": "Normal query",
            "department": "",
            "retrieved_context": "",
            "approval_status": "Not Required",
            "generated_response": ""
        }
    else:
        inputs = {
            "customer_query": query
        }
    
    # Run the graph using stream to observe node transitions
    console.print("[bold yellow]⚙️  Workflow Path Execution:[/bold yellow]")
    current_node = None
    
    # We invoke/stream. Note: in LangGraph, streaming events helps trace the path.
    # To avoid double execution side-effects, we can stream the updates.
    try:
        events = graph.stream(inputs, config, stream_mode="updates")
        for event in events:
            for node_name, state_update in event.items():
                current_node = node_name
                # Display node execution
                console.print(f"  [bold green]✔[/bold green] Executed Node: [bold blue]{node_name}[/bold blue]")
                
                # If classification is done, display details
                if node_name == "classify_intent":
                    dept = state_update.get("department", "Unknown")
                    issue = state_update.get("issue_type", "Normal")
                    appr = state_update.get("approval_status", "Not Required")
                    console.print(f"    ├─ [bold]Classified Department:[/bold] {escape(dept)}")
                    console.print(f"    ├─ [bold]Issue Type:[/bold] {escape(issue)}")
                    console.print(f"    └─ [bold]Human Approval Required:[/bold] {'⚠️  YES' if appr == 'Pending' else 'No'}")
                
                # If loading history retrieved anything, display it
                if node_name == "load_customer_info_and_history":
                    name = state_update.get("customer_info", {}).get("name", "Unknown")
                    context = state_update.get("retrieved_context", "")
                    console.print(f"    ├─ [bold]Customer Name:[/bold] {escape(name)}")
                    if context:
                        console.print(f"    └─ [bold]Retrieved Memory History:[/bold] Found past interaction.")
    except Exception as e:
        console.print(f"[bold red]Error during graph execution: {e}[/bold red]")
        return None

    # Check if the graph was interrupted at human_approval
    state_snapshot = graph.get_state(config)
    if state_snapshot.next and "human_approval" in state_snapshot.next:
        console.print(f"\n[bold yellow]🛑 INTERRUPT REACHED: Halted at '{state_snapshot.next[0]}'[/bold yellow]")
        
        # Display draft response that needs approval
        current_state = state_snapshot.values
        draft = current_state.get("generated_response", "")
        issue_type = current_state.get("issue_type", "Refund request")
        dept = current_state.get("department", "Billing")
        
        console.print(Panel(
            f"[bold]Department:[/bold] {escape(dept)}\n"
            f"[bold]Issue Type:[/bold] {escape(issue_type)}\n"
            f"[bold]Draft Response Under Review:[/bold]\n{escape(draft)}",
            title="[bold red]⚠️ SUPERVISOR APPROVAL REQUIRED[/bold red]",
            border_style="yellow"
        ))
        
        # Ask for human interaction
        console.print("[bold yellow]Please enter supervisor action:[/bold yellow]")
        console.print("  [1] Approve request")
        console.print("  [2] Reject request")
        
        choice = ""
        while choice not in ["1", "2"]:
            choice = input("Enter choice (1 or 2): ").strip()
            
        if choice == "1":
            approval_decision = "Approved"
            console.print("[bold green]✔ Response APPROVED by Supervisor.[/bold green]")
        else:
            approval_decision = "Rejected"
            console.print("[bold red]✖ Response REJECTED by Supervisor.[/bold red]")
            
        # Update the state checkpoint with the supervisor's decision
        graph.update_state(
            config,
            {"approval_status": approval_decision},
            as_node="human_approval"
        )
        
        # Resume the workflow
        console.print("\n[bold yellow]⚙️  Resuming Workflow Path:[/bold yellow]")
        events = graph.stream(None, config, stream_mode="updates")
        for event in events:
            for node_name, state_update in event.items():
                console.print(f"  [bold green]✔[/bold green] Executed Node: [bold blue]{node_name}[/bold blue]")

    # Retrieve final state
    final_snapshot = graph.get_state(config)
    final_values = final_snapshot.values
    
    response = final_values.get("generated_response", "No response generated.")
    final_dept = final_values.get("department", "Unknown")
    final_issue = final_values.get("issue_type", "Normal")
    final_name = final_values.get("customer_info", {}).get("name", "Unknown")
    
    console.print(f"\n[bold green]🏁 Workflow Finished Successfully![/bold green]")
    console.print(Panel(
        f"[bold]Customer Name:[/bold] {escape(final_name)}\n"
        f"[bold]Department:[/bold] {escape(final_dept)}\n"
        f"[bold]Issue Type:[/bold] {escape(final_issue)}\n\n"
        f"[bold green]Final Customer Response:[/bold green]\n{escape(response)}",
        title="[bold green]🌟 Final Output 🌟[/bold green]",
        border_style="green"
    ))
    return final_values

def main():
    console.print(Panel(
        "[bold white]AI-Powered Customer Support Automation System[/bold white]\n"
        "LangGraph, SQLite Memory, RAG Pipeline & Human-in-the-Loop",
        title="[bold cyan]Demonstration Script[/bold cyan]",
        border_style="cyan"
    ))
    
    # Initialize/Reset the database for a clean demo run
    memory.init_db()
    
    # We will use a consistent session Thread ID to simulate conversation memory
    thread_id = "customer_session_99"
    
    # Query 1: Sales Support
    q1 = "What are the pricing plans available for your software?"
    run_query(q1, thread_id)
    time.sleep(1)
    
    # Query 2: Account Support
    q2 = "I forgot my account password."
    run_query(q2, thread_id)
    time.sleep(1)
    
    # Query 3: Technical Support
    q3 = "My application crashes whenever I upload a file."
    run_query(q3, thread_id)
    time.sleep(1)
    
    # Query 4: Billing + Human Approval
    # We prepended "My name is David" to ensure his name gets saved in memory
    q4 = "My name is David. I need a refund for my annual subscription."
    run_query(q4, thread_id)
    time.sleep(1)
    
    # Query 5: SQLite Memory Retrieval
    # David asks about his previous issue
    q5 = "What was my previous support issue?"
    run_query(q5, thread_id)
    
    console.print(f"\n[bold magenta]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold magenta]")
    console.print("[bold green]✔ Demonstration Complete![/bold green]")
    console.print(f"[bold magenta]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold magenta]\n")

if __name__ == "__main__":
    main()

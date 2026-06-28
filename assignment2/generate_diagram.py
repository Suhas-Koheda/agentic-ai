import sys
from workflow import get_graph

def main():
    print("Generating workflow.png natively from LangGraph...")
    graph = get_graph()
    try:
        # LangGraph get_graph().draw_mermaid_png() generates a PNG using mermaid.ink API
        png_data = graph.get_graph().draw_mermaid_png()
        with open("workflow.png", "wb") as f:
            f.write(png_data)
        print("Success! Natively generated diagram saved as workflow.png")
    except Exception as e:
        print(f"Error generating native diagram: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

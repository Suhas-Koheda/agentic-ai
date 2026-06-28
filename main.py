from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import (create_tool_calling_agent,AgentExecutor)

from tools import(
    attendance_calculator,
    result_calculator,
    fee_balance_calculator,
    library_fine_calculator,
    hostel_fee_calculator,
    student_information
)

llm=ChatOllama(model="gemma4:e2b",temperature=0.0)
tools=[attendance_calculator,result_calculator,fee_balance_calculator,library_fine_calculator,hostel_fee_calculator,student_information]
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a Smart College Assistant.

            Use the available tools whenever needed.
            Answer clearly and concisely.
            """,
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

print("College Assistant Ready")
print("Type exit to quit")
print("(For multi-line queries, type queries line by line or press Ctrl+D when done)")

while True:
    print("\nStudent: ", end="")
    lines = []
    try:
        while True:
            line = input()
            if line.lower() == "exit":
                print("Goodbye!")
                exit()
            lines.append(line)
            if len(lines) > 1 and line.strip() == "":
                lines.pop()
                break
            if len(lines) == 1:
                break
    except EOFError:
        pass
    
    query = "\n".join(lines)
    
    if not query.strip():
        continue

    response = agent_executor.invoke(
        {
            "input": query
        }
    )

    print("\nAssistant:")
    print(response["output"])
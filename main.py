from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import (create_tool_calling_agent,AgentExecutor)

from tools import(
    attendance_calculator,
    result_calculator,
    fee_balance_calculator,
    library_fine_calucaltor,
    hostel_fee_calculator,
    student_information
)

llm=ChatOllama(model="gemma4:e2b",temperature=0.0)
tools=[attendance_calculator,result_calculator,fee_balance_calculator,library_fine_calucaltor,hostel_fee_calculator,student_information]
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

while True:

    query = input("\nStudent: ")

    if query.lower() == "exit":
        break

    response = agent_executor.invoke(
        {
            "input": query
        }
    )

    print("\nAssistant:")
    print(response["output"])


# I attended 72 classes out of 90. Am I eligible for exams?
# My marks are 95, 90, 88, 91 and 87. What is my grade?
# My course fee is 50000 and I have paid 35000. How much fee is pending?
# I returned a library book 8 days late. What is the fine amount?
# Hostel fee is 6000 per month and I stayed for 5 months.

# I attended 80 classes out of 100.
# My marks are 90, 85, 88, 92 and 95.
# My course fee is 60000 and I paid 45000.

# Provide:
# 1. Attendance Status
# 2. Grade
# 3. Pending Fee
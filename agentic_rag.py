import os

import chromadb

from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM

from crewai.tools import tool


# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is missing."
    )


# ==========================================
# 1. GEMINI LLM
# ==========================================

gemini = LLM(
    model="gemini/gemini-3.5-flash",
    api_key=GEMINI_API_KEY,
)


# ==========================================
# 2. CONNECT TO CHROMADB
# ==========================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

kb = client.get_collection("policies")


# ==========================================
# 3. CHROMADB TOOL
# ==========================================

@tool("Policy Knowledge Search")
def search_policies(query: str) -> str:
    """
    Search company policy documents by semantic meaning.

    Use this tool when you need information about:
    refunds, returns, support response times,
    customer entitlements, or other company policies.

    Use focused queries.

    If a question contains multiple topics,
    call this tool multiple times with one topic per query.

    Args:
        query: A focused question about one topic.
    """

    results = kb.query(
        query_texts=[query],
        n_results=3
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    if not documents:
        return f"Nothing found for '{query}'. Please rephrase."

    lines = [
        f"Results for '{query}':"
    ]

    for i, (document, distance) in enumerate(
        zip(documents, distances),
        start=1
    ):
        score = 1 - distance

        lines.append(
            f"[{i}] ({score:.2f}) {document}"
        )

    return "\n".join(lines)


# ==========================================
# 4. RESEARCHER AGENT
# ==========================================

researcher = Agent(
    role="Policy Researcher",

    goal=(
        "Find every piece of evidence needed to "
        "fully answer the user's question."
    ),

    backstory=(
        "You are a meticulous policy researcher. "
        "You break complex questions into focused "
        "sub-questions and search each topic separately. "
        "You never invent facts. "
        "You use the Policy Knowledge Search tool "
        "whenever evidence is needed."
    ),

    tools=[search_policies],

    llm=gemini,

    allow_delegation=False,

    verbose=True,
)


# ==========================================
# 5. WRITER AGENT
# ==========================================

writer = Agent(
    role="Executive Communications Writer",

    goal=(
        "Turn the researcher's evidence into a "
        "clear and accurate final answer."
    ),

    backstory=(
        "You are an excellent technical writer. "
        "You answer clearly and concisely. "
        "You use ONLY the evidence provided by "
        "the researcher. "
        "You never invent missing information."
    ),

    llm=gemini,

    allow_delegation=False,

    verbose=True,
)


# ==========================================
# 6. GET USER QUESTION
# ==========================================

question = input(
    "\nEnter your question: "
)


# ==========================================
# 7. RESEARCH TASK
# ==========================================

research_task = Task(
    description=f"""
    Research the following question:

    "{question}"

    Identify all separate topics in the question.

    Break the question into focused sub-questions.

    Search the Policy Knowledge Search tool
    separately for each relevant topic.

    Gather all relevant evidence from the
    knowledge base.

    Do not invent information.

    Clearly mention any information that
    could not be found.
    """,

    expected_output=(
        "A list of relevant evidence from the "
        "knowledge base, organized by topic, "
        "including any gaps."
    ),

    agent=researcher,
)


# ==========================================
# 8. WRITING TASK
# ==========================================

writing_task = Task(
    description=f"""
    Using ONLY the evidence produced by the
    Policy Researcher, answer the original question:

    "{question}"

    Do not perform additional searches.

    Do not introduce outside knowledge.

    Do not invent facts.

    Clearly explain the answer and compare
    relevant information when appropriate.

    If evidence is missing, state that clearly.
    """,

    expected_output=(
        "A concise, accurate final answer "
        "supported by the research evidence."
    ),

    agent=writer,

    context=[research_task],
)


# ==========================================
# 9. CREATE CREW
# ==========================================

crew = Crew(
    agents=[
        researcher,
        writer
    ],

    tasks=[
        research_task,
        writing_task
    ],

    process=Process.sequential,

    verbose=True,
)


# ==========================================
# 10. RUN AGENTIC RAG
# ==========================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("AGENTIC RAG")
    print("=" * 70)

    print(f"\nQuestion: {question}")

    print("\nRunning Researcher and Writer agents...\n")

    result = crew.kickoff()

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(result)

    print("=" * 70)
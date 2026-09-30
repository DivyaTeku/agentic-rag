# Agentic RAG using CrewAI

This project converts a traditional RAG pipeline into an
Agentic RAG architecture using CrewAI and ChromaDB.

## Architecture

User Query
    ↓
Researcher Agent
    ↓
Policy Knowledge Search Tool
    ↓
ChromaDB
    ↓
Research Evidence
    ↓
Writer Agent
    ↓
Final Answer

## Agents

### Researcher Agent
- Analyzes the user's question
- Breaks complex questions into sub-queries
- Uses the ChromaDB search tool
- Can perform multiple searches
- Collects evidence
- Does not invent facts

### Writer Agent
- Receives the researcher's evidence
- Does not have database access
- Produces the final response
- Uses only retrieved evidence

## Technologies

- Python
- CrewAI
- ChromaDB
- Gemini 3.5 Flash

## Running

```bash
pip install -r requirements.txt
python seed_chroma.py
python agentic_rag.py

---

# 22. Why the two agents are separated

This is actually something I'd mention in the PR.

The architecture deliberately gives **only the Researcher** access to ChromaDB.

```text
Researcher
   │
   ├── search_policies()
   │
   └── ChromaDB

Writer
   │
   └── NO TOOLS

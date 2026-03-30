"""
Quick manual test — add 3 FPL sentences to the RAG system and query it.

Run with:
    python try_rag.py

The first run downloads the embedding model (~80 MB). Subsequent runs are instant.
"""

from agents.rag_agent import RagAgent, Document

agent = RagAgent(db_path="./try_rag_db")

# ------------------------------------------------------------------
# Add 3 documents
# ------------------------------------------------------------------

print("Adding documents...")

agent.ingest(Document(
    content="Salah is the standout captaincy option for gameweek 28. He has scored in four of his last five home games and faces a weak defence.",
    source="FPL Focus",
    doc_type="opinion",
    author="FPL Focus",
    trust_weight=8,
))

agent.ingest(Document(
    content="Haaland has blanked in three consecutive gameweeks and his next two fixtures are against top-four defences. Avoid as captain.",
    source="Fantasy Football Scout",
    doc_type="opinion",
    author="FFS",
    trust_weight=7,
))

agent.ingest(Document(
    content="The FPL captain earns double points for that gameweek. If your captain does not play, the vice-captain earns double points instead.",
    source="Official FPL Rules",
    doc_type="fact",
    trust_weight=10,
))

print(f"Total chunks stored: {agent.document_count()}\n")

# ------------------------------------------------------------------
# Query
# ------------------------------------------------------------------

query = "who should I captain this week?"
print(f"Query: '{query}'\n")
print("-" * 60)

results = agent.retrieve(query, n_results=3)

for i, chunk in enumerate(results, 1):
    print(f"Result {i}")
    print(f"  Source:    {chunk.source} ({chunk.doc_type})")
    print(f"  Author:    {chunk.author or 'N/A'}")
    print(f"  Trust:     {chunk.trust_weight}/10")
    print(f"  Relevance: {chunk.relevance_score:.2f}")
    print(f"  Boosted:   {chunk.boosted_score:.2f}")
    print(f"  Text:      {chunk.text}")
    print()

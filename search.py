import os
import numpy as np
from datetime import datetime
import streamlit as st
import config
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import initialize_agent, Tool

# Set a local model cache directory inside your app
HF_CACHE_DIR = "./hf_model_cache"
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE_DIR

@st.cache_resource(show_spinner="Loading Chroma vectorstore and embedding model...")
def initialize_search_index():
    """
    Load the Chroma vectorstore and the embedding model.
    Returns:
        db: Chroma vectorstore instance
        embeddings: HuggingFaceEmbeddings instance
    """
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    # Connect to the persistent Chroma DB
    db = Chroma(
        persist_directory=config.PERSIST_DIR,
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings
    )
    return db, embeddings



def search_pdfs(
    query: str,
    db: Chroma,
    k: int = config.SEARCH_RESULT_K,
    alpha: float = 0.0,
    max_snippet_length: int = 500
) -> list[dict]:
    """
    Perform a semantic search using Chroma, with optional date-based weighting and metadata filtering.

    Parameters:
        query: the user query string
        db: the Chroma vectorstore
        embeddings: the embedding model (for embedding the query)
        k: number of results to return
        alpha: decay factor for date weighting (0 = no decay)
        publication_types: list of metadata values to filter on 'publication_type'
        max_snippet_length: truncate snippet to this length

    Returns:
        List of result dicts containing metadata and snippets, sorted by weighted score.
    """
    # Get top-k docs and FAISS-style distances (here, cosine distance)
    docs_and_scores = db.similarity_search_with_score(query, k=k)

    now = datetime.now()
    results = []

    for doc, distance in docs_and_scores:
        meta = doc.metadata

        # Compute date decay weight
        date_weight = 1.0
        pub_date_str = meta.get("publication_date")
        if pub_date_str:
            try:
                pub_date = datetime.fromisoformat(pub_date_str)
            except ValueError:
                try:
                    pub_date = datetime.strptime(pub_date_str, "%B %d, %Y")
                except Exception:
                    pub_date = None
            if pub_date:
                days_diff = (now - pub_date).days
                date_weight = np.exp(-alpha * days_diff / 365)

        # Convert distance to similarity score
        similarity_score = 1 / (1 + distance)
        weighted_score = similarity_score * date_weight

        # Build snippet
        snippet = doc.page_content.replace("\n", " ")
        snippet = snippet[:max_snippet_length]

        results.append({
            "title": meta.get("title"),
            "filename": meta.get("pdf_url", "").split("/")[-1],
            "summary": meta.get("summary"),
            "publication_date": meta.get("publication_date"),
            "publication_type": meta.get("publication_type"),
            "url": meta.get("article_url"),
            "pdf_url": meta.get("pdf_url"),
            "snippet": snippet,
            "score": similarity_score,
            "date_weight": date_weight,
            "weighted_score": weighted_score,
        })

    # Sort by weighted_score descending and take top-k
    results = sorted(results, key=lambda x: x["weighted_score"], reverse=True)[:k]
    return results

def chat_rag(
    prompt: str,
    db,
    k: int = config.SEARCH_RESULT_K,
    alpha: float = 0.0,
    max_snippet_length: int = 500,
    callbacks: list | None = None,
    openai_api_key: str | None = None,
) -> str:
    """
    Performs RAG via a zero-shot agent that invokes `search_pdfs` as a tool only when needed.
    Keeps the same interface signature for modularity.
    Returns the final answer with inline citations.
    """
    # Define a local tool that wraps your existing search_pdfs, capturing db and retrieval parameters
    def _search_pdfs_tool(query: str) -> str:
        docs = search_pdfs(query, db, k=k, alpha=alpha, max_snippet_length=max_snippet_length)
        # Format results as numbered citation snippets
        return "\n".join(
            f"[{i+1}] {d['title']} ({d['publication_date']}): {d['snippet']}"
            for i, d in enumerate(docs)
        )

    search_tool = Tool.from_function(
        func=_search_pdfs_tool,
        name="search_pdfs",
        description="Search the PDF corpus and return top-k relevant snippets numbered for citation."
    )

    # Add a no-op 'none' tool to handle explicit 'Action: None' responses
    none_tool = Tool.from_function(
        func=lambda _: "",
        name="none",
        description="No action. Use when no retrieval is needed."
    )

    # Instantiate the LLM, updating callbacks and API key
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        streaming=True,
        temperature=0,
        callbacks=callbacks or [],
        openai_api_key=openai_api_key,
    )

    # Initialize a zero-shot agent that decides when to call the tool
    agent = initialize_agent(
        tools=[search_tool, none_tool],  # include the none tool to catch 'Action: None'
        llm=llm,
        agent="zero-shot-react-description",
        verbose=False,
    )

    # System prompt to enforce inline citation formatting
    citation_prompt = SystemMessage(content=(
        "You are a research assistant. Whenever you state a fact drawn from our tools, "
        "append a citation number in brackets like '[1]'. At the end, list each source with its number."
    ))

    # Run the agent in streaming mode and return the answer text
    answer = agent.invoke([
        citation_prompt,
        HumanMessage(content=prompt)
    ])

    return answer

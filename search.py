import faiss, numpy as np, pandas as pd
from datetime import datetime
import streamlit as st
import config
from config import INDEX_PATH, MAP_PATH, PAGES_PATH                            # your own config.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI  # If using ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from typing import Generator, List, Tuple
from collections import defaultdict
from operator   import itemgetter
import json
import random
import re

SPINNER_MESSAGES = [
    "Tuning quantum flux capacitors…",
    "Feeding PDFs to our pet AI llama…",
    "Consulting the Book of Infinite Wisdom…",
    "Summoning digital gnomes for indexing…",
    "Injecting caffeine into vector space…",
    "Decoding ancient neural runes…",
    "Polishing the FAISS crystal ball…",
    "Massaging cosine similarities…",
    "Bribing the embeddings to behave…",
    "Unleashing the power of dot products…",
]
# ╭──────────────────────────────────────────────────────────────╮
# │  1. initialise FAISS index + dataframes (cached)            │
# ╰──────────────────────────────────────────────────────────────╯
@st.cache_resource(show_spinner=random.choice(SPINNER_MESSAGES))
def initialize_search_index(openai_api_key: str | None = None):
    """
    Returns
    -------
    index     : faiss.Index
    embeddings: Embedding model (HF or OpenAI)
    mapping   : pd.DataFrame  (index on vector_id)
    pages     : pd.DataFrame  (index on document ID)
    """
    # choose embedding backend
    if "text-embedding" in config.EMBEDDING_MODEL:
        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=openai_api_key
        )
    else:
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    # load artefacts
    index   = faiss.read_index(INDEX_PATH)
    mapping = pd.read_parquet(MAP_PATH).set_index("vector_id")
    pages   = pd.read_parquet(PAGES_PATH).set_index("ID")

    # --- convenience column so we don't have to re-parse later -------------
    def _parse_year(date_str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S",
                    "%B %d, %Y, %I:%M:%S %p", "%B %d, %Y",
                    "%b %d, %Y, %I:%M:%S %p"):
            try:
                return datetime.strptime(date_str, fmt).year
            except Exception:
                continue
        return None

    pages["year"] = pages["Publication Date"].apply(_parse_year)

    tmp = (
        mapping.reset_index()           # vector_id + doc_id
               .merge(pages["year"], left_on="doc_id", right_index=True)
               .dropna(subset=["year"])
    )
    year2vec = {
        int(y): grp["vector_id"].to_numpy(np.int64)
        for y, grp in tmp.groupby("year", sort=True)
    }

    return index, embeddings, mapping, pages, year2vec



def merge_snippets(
        hits: list[dict],
        *,
        max_snippets_per_doc: int = 3,
        joiner: str = " … "
    ) -> list[dict]:
    """
    Parameters
    ----------
    hits  : list of dicts – output of `search_pdfs`
    max_snippets_per_doc : keep at most this many snippets per PDF
    joiner               : string inserted between snippets

    Returns
    -------
    merged : list[dict]  – one entry per document, ordered by best score
    """

    # bucket all snippets by file (or any unique doc key you prefer)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for h in hits:
        buckets[h["pdf_url"]].append(h)

    merged = []
    for fname, lst in buckets.items():
        # sort snippets inside one doc: best weighted_score first
        lst.sort(key=itemgetter("start_char"), reverse=True)

        # concatenate up to max_snippets_per_doc unique snippets
        snippets_combined = joiner.join(
            s["snippet"] for s in lst[:max_snippets_per_doc]
        )

        # start from the best‐scoring dict, overwrite the snippet & score fields
        best = lst[0].copy()
        best["snippet"] = snippets_combined
        # (optional) you may want an aggregate score – e.g. max or mean
        best["combined_score"] = best["weighted_score"]      # keep the max

        merged.append(best)

    # order the final list by the best score per document
    merged.sort(key=itemgetter("combined_score"), reverse=True)
    return merged

# ╭──────────────────────────────────────────────────────────────╮
# │  2.  semantic search with optional date-decay weighting      │
# ╰──────────────────────────────────────────────────────────────╯
def search_pdfs(
        query: str,
        faiss_index,
        embeddings,
        mapping_df: pd.DataFrame,
        pages_df: pd.DataFrame,
        *,
        k: int = config.SEARCH_RESULT_K,
        alpha: float = 0.0,
        max_snippet_length: int = 500,
        threshold: float = 0.0,
        year2vec: dict[int, np.ndarray] | None  = None,
        year: int | None = None
    ) -> list[dict]:
    """
    Parameters
    ----------
    query  : str    – user search string
    faiss_index,
    embeddings,
    mapping_df,
    pages_df : objects returned by `initialize_search_index`
    k      : int    – how many results
    alpha  : float  – 0 → ignore recency, >0 → exponential decay / year
    """

    if query == "":
        return []
    # ➊ embed & search
    q_vec = embeddings.embed_query(f"query: {query.lower()}")

    if year and year2vec:
        # ---------- 0) do we have vectors for this year? ----------
        vec_ids = year2vec.get(year)
        if vec_ids is None or len(vec_ids) == 0:
            return []

        selector = faiss.IDSelectorArray(vec_ids)      # or Batch/Bitmap if huge
        p = faiss.SearchParametersIVF(sel = selector)
        D, I  = faiss_index.search(np.array([q_vec]), k, params=p)            # D = L2 distances
    else:
        D, I  = faiss_index.search(np.array([q_vec]), k)            # D = L2 distances

    now = datetime.now()
    results = []

    for dist, vec_id in zip(D[0], I[0]):


        similarity   = 1 / (1 + dist)      # convert L2 distance → similarity


        # mapping row gives us doc_id and offsets
        meta_row = mapping_df.loc[vec_id]
        doc_row  = pages_df.loc[meta_row["doc_id"]]   # full metadata


        # rebuild chunk text
        text = doc_row["fulltext"]
        snippet_raw = text[int(meta_row["start_char"]) : int(meta_row["end_char"])]
        snippet     = snippet_raw.replace("\n", " ")[:max_snippet_length]

        # date weighting
        pub_date_str = doc_row.get("Publication Date")
        date_weight  = 1.0
        pub_date     = None                              # will hold the parsed dt

        if pub_date_str:
            # ----  robust parsing  --------------------------------------------------
            for fmt in (
                "%Y-%m-%d",                       # 2025-05-01
                "%Y-%m-%dT%H:%M:%S",              # ISO without TZ
                "%B %d, %Y, %I:%M:%S %p",         # Sep 16, 2024, 12:00:00 AM
                "%B %d, %Y",
                "%b %d, %Y, %I:%M:%S %p"
            ):
                try:
                    pub_date = datetime.strptime(pub_date_str, fmt)
                    break
                except ValueError:
                    continue
            if pub_date:
                days_diff   = (now - pub_date).days
                date_weight = np.exp(-alpha * days_diff / 365)

                # **clean representation without the clock**
                pub_date_clean = pub_date.strftime("%Y-%m-%d")   # e.g. "2024-09-16"
            else:
                pub_date_clean = pub_date_str                    # fallback
        else:
            pub_date_clean = None

        # Don't weigh by year for search in a given year
        if year and year2vec:
            weighted = similarity
        else:
            weighted = similarity * date_weight

        if similarity < weighted:
            continue

        results.append({
            "title"            : doc_row.get("Title"),
            "filename"         : doc_row.get("PDF Filename"),
            "summary"          : doc_row.get("Summary"),
            "publication_date" : pub_date_clean,
            "publication_type" : doc_row.get("Publication Type"),
            "url"              : doc_row.get("Article URL"),
            "pdf_url"          : doc_row.get("PDF URL"),
            "snippet"          : snippet,
            "score"            : float(similarity),
            "date_weight"      : float(date_weight),
            "weighted_score"   : float(weighted),
            "start_char"       : int(meta_row["start_char"])
        })

    # sort & trim
    results.sort(key=lambda x: x["weighted_score"], reverse=True)
    return merge_snippets(results, max_snippets_per_doc=10)



def decide_rag(
    prompt: str,
    history: List[dict],
    max_query_tokens: int = 32,
    openai_api_key: str | None = None,
) -> Tuple[bool, str]:
    """
    Return (use_rag, search_query).  Uses a cheap, non-streaming call.
    The model MUST answer with a JSON dict: {"use_rag": bool, "query": str}
    """
    sys = SystemMessage(
        content=(
            "You are a routing controller. "
            "Analyse the user's latest message AND the recent conversation. "
            "If the assistant can answer fully without consulting internal documents, "
            "return: {\"use_rag\": false, \"query\": \"\"}. "

            "Otherwise, set use_rag to true and generate a precise, minimal search query "
            f"(≤ {max_query_tokens} tokens) that would retrieve relevant documents. "

            "Disambiguate similar terms. For example, if the user mentions 'UCO', "
            "treat it as 'Used Cooking Oil' unless explicitly stated otherwise. "
            "Avoid inserting or hallucinating organization names like 'Transport & Environment' "
            "unless the user actually mentioned them. "

            "Avoid bloated or vague queries. Focus on the technical, regulatory, or factual content "
            "needed to answer the user's question. "

            "Only return the JSON output with the keys: use_rag and query."
        )
    )


    # Keep just the last few turns to save tokens
    latest_turns = history[-6:]  # tweak as needed

    hist_msgs = [
        (HumanMessage if m["role"] == "user" else SystemMessage)(content=m["content"])
        for m in latest_turns
    ]

    messages = hist_msgs + [sys, HumanMessage(content=prompt)]

    router = ChatOpenAI(
        model_name="gpt-4.1-nano",   # cheap & fast
        openai_api_key=openai_api_key,
    )

    answer = router.invoke(messages).content
    try:
        j = json.loads(answer)
        return bool(j.get("use_rag")), j.get("query", "")
    except Exception:
        # fail safe: default to RAG with the raw prompt
        return True, prompt

# ───────────────────────────────────────────────────────────────
# 1 ▸ main public function
# ───────────────────────────────────────────────────────────────
def chat_rag(
    prompt: str,
    history: List[dict],
    *,
    faiss_index,
    embeddings,
    mapping_df: pd.DataFrame,
    pages_df: pd.DataFrame,
    k: int = 5,
    alpha: float = 0.0,
    max_snippet_length: int = 500,
    callbacks: list | None = None,
    openai_api_key: str | None = None,
    llm_model: str = "gpt-4o-mini"
) -> Generator[str, None, None]:
    """
    Streaming generator:
      • decides first whether to call RAG
      • if yes, builds a search query from conversation + prompt
      • streams the answer, adding a Sources block when RAG was used
    """

    # --- 1) router ----------------------------------------------------------
    use_rag, search_query = decide_rag(prompt, history, openai_api_key=openai_api_key)

    # --- 2) retrieval if needed ---------------------------------------------
    if use_rag:
        docs = search_pdfs(
            search_query,
            faiss_index,
            embeddings,
            mapping_df,
            pages_df,
            k=k,
            alpha=alpha,
            max_snippet_length=max_snippet_length,
        )

        context_blocks = []
        for idx, d in enumerate(docs, start=1):
            title, url = d.get("title", "Untitled"), d.get("url", "#")
            snippet = d.get("snippet", "").replace("\n", " ").strip()
            context_blocks.append(f"[{idx}] {title} — {url}\n{snippet}")
        context = "\n\n".join(context_blocks)

        sys_ctx = SystemMessage(
            content=(
                "You are a meticulous research assistant for Transport & Environment. "
                "When you quote or paraphrase, add bracketed numbers like [1]."
            )
        )
        sys_docs = SystemMessage(content=f"Context documents:\n\n{context}")
    else:
        docs = []          # empty – will suppress the Sources block later
        sys_ctx = SystemMessage(
            content="You are Transport & Environment’s helpful assistant."
        )
        # no context documents
        sys_docs = None

    # --- 3) build chat messages ---------------------------------------------
    hist_msgs = [
        (HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"])
        for m in history
    ]

    seed_messages = [sys_ctx] + ([sys_docs] if sys_docs else []) + [
        HumanMessage(content=prompt)
    ]
    messages = hist_msgs + seed_messages


    # --- 4) answer -----------------------------------------------------------
    model = ChatOpenAI(
        model_name=llm_model,
        streaming=True,
        callbacks=callbacks,
        openai_api_key=openai_api_key,
        temperature=1 if llm_model != "gpt-4o-mini" else 0
    )

    for token in model.stream(messages):
        yield token
    # --- 5) add sources if RAG was used -------------------------------------
    if use_rag and docs:
        yield "\n\n---\n\n**Sources:**\n\n"
        for idx, result in enumerate(docs, start=1):
            md = (
                f"**[{idx}]. [{result.get('title','Untitled')}]({result.get('url','#')})**: [{result.get('pdf_url','Untitled')}]({result.get('pdf_url','#')}) - \n"
                f"*{result.get('publication_type','Unknown')} – "
                f"{result.get('publication_date','')} – "
                f"{round(result['score']*100,1)}% match*  \n"
                f"> {result.get('snippet','').strip()}  \n\n"
            )
            yield md


def position_timeline(
        topic: str,
        *,
        faiss_index,
        embeddings,
        mapping_df,
        pages_df,
        year2vec,
        openai_api_key: str,
        alpha: float = 0.0,
        max_snippet_length: int = 500,
        k_per_year: int = config.TOP_HITS_PER_YEAR,
        min_score: float = config.SIMILARITY_THRESHOLD,
        triage_model: str = config.TRIAGE_MODEL,
        timeline_model: str = config.TIMELINE_MODEL,
):
    """
    Return a fully-formed Markdown timeline answering
        “What is T&E’s position on <topic> and how did it change?”
    with bracketed citations.

    Steps
      1. router → search query
      2. per-year FAISS search (filtered)
      3. triage hits with a cheap model
      4. big model writes the chronology
    """
    # ---------- 1) router ----------
    use_rag, search_query = decide_rag(
        topic,                       # user prompt
        history=[],                  # no chat
        max_query_tokens=32,
        openai_api_key=openai_api_key,
    )
    # we *always* want RAG here, but decide_rag also disambiguates the query
    if not use_rag:
        search_query = topic

    # ---------- 2) gather hits per year ----------
    now_year = datetime.now().year
    years = [y for y in sorted(year2vec) if now_year - y]
    all_hits: list[dict] = []

    for yr in sorted(years, reverse=True):        # newest → oldest
        hits = search_pdfs(
            search_query,
            faiss_index   = faiss_index,
            embeddings    = embeddings,
            mapping_df    = mapping_df,
            pages_df      = pages_df,
            k             = k_per_year,       # oversample, we'll triage
            alpha         = alpha,
            max_snippet_length = max_snippet_length,
            threshold     = min_score,
            year2vec      = year2vec,
            year          = yr,
        )
        # annotate with year label
        for h in hits:
            h["year"] = yr
        all_hits.extend(hits)

    if not all_hits:
        return (
            "No documents above the similarity threshold were found in the past "
        )

    # ---------- 3) triage with gpt-4.1-nano ----------
    #triage_chunks = []
    #for idx, h in enumerate(all_hits, start=1):
    #    triage_chunks.append(
    #        f"[{idx}] ({h['year']}) {h['title']}\n{h['snippet']}"
    #    )
#
    #triage_prompt = (
    #    "You are a policy analyst at Transport & Environment. "
    #    "Given the snippets below, select **only** those that are somewhat related to T&E’s own *position or stance* on the topic "
    #    f"“{topic}”.\n\n"
    #    "Return a JSON list of the reference numbers that are relevant.\n\n"
    #    "Snippets:\n" + "\n\n".join(triage_chunks)
    #)
#
    #triager = ChatOpenAI(
    #    model_name=triage_model,
    #    openai_api_key=openai_api_key,
    #    temperature=1 if triage_model != "gpt-4o-mini" else 0
    #)
    #try:
    #    triage_reply = triager.invoke([HumanMessage(content=triage_prompt)]).content
    #    keep_ids = set(json.loads(triage_reply))
    #except Exception:
    #    # fall back: keep everything
    #    keep_ids = set(range(1, len(all_hits) + 1))
    keep_ids = set(range(1, len(all_hits) + 1))
    kept_hits = [h for i, h in enumerate(all_hits, start=1) if i in keep_ids]
    if not kept_hits:
        return "No publication explicitly states T&E’s position on this topic."

    # group again by year (newest → oldest)
    kept_hits.sort(key=lambda x: ( -x["year"], -x["weighted_score"]))
    grouped: dict[int, list[dict]] = defaultdict(list)
    for h in kept_hits:
        if len(grouped[h["year"]]) < k_per_year:
            grouped[h["year"]].append(h)

    # ---------- 4) build context and let the big model write ----------
    ctx_blocks = []
    ref_map = {}                      # map global ref → markdown link
    global_idx = 1
    for yr in sorted(grouped, reverse=True):
        for h in grouped[yr]:
            ref = f"{yr}-{global_idx}"
            ref_map[ref] = (
                f"[{h['title']}]({h['url'] or h['pdf_url'] or '#'})"
            )
            ctx_blocks.append(
                f"[{ref}] ({yr}) {h['title']}\n{h['snippet']}"
            )
            h["ref"] = ref
            global_idx += 1

    sys_ctx = SystemMessage(
        content=(
            "You are an expert policy summariser at Transport & Environment. "
            "Using the provided snippets, compose a chronological timeline from earlier to later"
            "explaining how T&E’s **position** on the topic evolved. "
            "For each year, write 1-3 bullet points. "
            "Format the answer in markdown with a title and highlights for headings, years, and bullet points."
            "When you cite, use the bracketed reference ids exactly as given "
            "(e.g. cite exactly like [2022-3] or [2022-3, 2024-5]). Finish with a two-sentence ‘Overall trajectory’ "
            "summary.\n\n"
            "Snippets:\n" + "\n\n".join(ctx_blocks)
        )
    )

    model = ChatOpenAI(
        model_name=timeline_model,
        openai_api_key=openai_api_key,
        temperature=1 if timeline_model != "gpt-4o-mini" else 0,
        streaming=True
    )


    # ------------------------------------------------------------
    # 5)   POST-PROCESS: inline ↔︎ foot-note, build compact sources
    # ------------------------------------------------------------
    #   • Convert every “[YYYY-n]” used by the model into a foot-note marker
    #   • Keep only refs that are *actually present* in the text
    # ------------------------------------------------------------

    # ① recognise
    #    • square OR round brackets
    #    • single id  → [2016-19]
    #    • lists      → [2016-19, 2016-20]  or  (2016-19,2016-20)
    citation_rx = re.compile(
        r'(?<!\[\^)'                       # negative lookbehind to exclude existing footnotes
        r'[\[\(]'                         # opening [ or (
        r'(\d{4}-\d+(?:\s*,\s*\d{4}-\d+)*)'
        r'[\]\)]'                         # closing ] or )
    )

    used_refs = set()
    buffer = ""

    for chunk in model.stream([sys_ctx]):
        token = chunk.content or ""
        buffer += token

        # Optionally: yield every sentence or newline
        if any(p in buffer for p in [". ", "\n"]):
            def replace_citations(text):
                def repl(match):
                    ids = [i.strip() for i in match.group(1).split(',')]
                    used_refs.update(ids)
                    return ', '.join(f'[^{i}]' for i in ids)
                return citation_rx.sub(repl, text)

            to_yield = replace_citations(buffer)
            yield to_yield
            buffer = ""

    # Final leftover buffer
    if buffer:
        yield citation_rx.sub(lambda m: ', '.join(f'[^{i.strip()}]' for i in m.group(1).split(',')), buffer)

    # --- Yield footnotes ---
    yield "\n\n---\n"
    for h in kept_hits:
        if h["ref"] in used_refs:
            link = h["url"] or h["pdf_url"] or "#"
            yield f"[^{h['ref']}]:  {h['publication_date'] or ''} - [{h['title']}]({link}) — {h['publication_type']}\n"

    if not used_refs:
        yield "*The model did not cite any document explicitly.*"

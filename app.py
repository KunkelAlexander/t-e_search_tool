import streamlit as st
import search as search
import time
import inspect
import config as config

# Set page configuration
st.set_page_config(
    page_title="AI Search",  # Title shown in the browser tab
    page_icon="assets/favicon.ico",                    # Emoji or path to a favicon file
    layout="wide",                     # Other options: "centered"
    initial_sidebar_state="expanded"   # Options: "auto", "expanded", "collapsed"
)

# Sidebar with logo and dropdown
# choose the keyword based on what the current API exposes
if "use_container_width" in inspect.signature(st.sidebar.image).parameters:
    st.sidebar.image("assets/logo.png", use_container_width=True)
else:
    st.sidebar.image("assets/logo.png", use_column_width=True)

# --- Default Session-State Values ---
def _init_state(defaults: dict):
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)

_init_state({
    "initialized": False,
    "alpha": 0.05,
    "max_snippet_length": 1500,
    "n_search_results": 10,
    "OPENAI_API_KEY": None,
    "selected_model": "gpt-4o-mini",
    "chat_history": [],
    "n_chrono_search_results": 3,
    "chrono_similarity_threshold": 75
})

with st.sidebar:
    if "use_container_width" in inspect.signature(st.button).parameters:
        if st.button("🔄  Reset", use_container_width=True):
                st.session_state.chat_history = []
    else:
        if st.button("🔄  Reset", use_column_width=True):
                st.session_state.chat_history = []

with st.sidebar.expander("Settings"):


    key = st.text_input(
        "🔑 Enter your OpenAI API key",
        type="password"
        )
    if key:
        st.session_state.OPENAI_API_KEY = key
        st.success("✅ API key saved!")
    st.slider("# Search Results", 5, 100, step=5, key="n_search_results")

    # Dropdown for selecting OpenAI model
    model_options = [
        "gpt-4o-mini",         #
        "o4-mini",             #
        "gpt-4.1-nano",        # 📊 Optimized for math, coding, and visual tasks
        "gpt-4.1-mini",        # 🧠 Efficient STEM-focused reasoning
        "gpt-4.1",             # 🧮 Advanced reasoning with visual input
        "o1-mini",             #
        "o3-mini",             #
        "o3 (expensive)",  #
    ]
    st.selectbox("🤖 Choose OpenAI model", model_options, key="selected_model")

    st.slider("Relevancy of more recent results", 0.0, 0.5, value = 0.01, step=0.01, key="alpha")
    st.slider("Length of snippets", 0, 2000, value=1000, step=50, key="max_snippet_length")

    st.slider("# Search Results", 1, 20, value=3, step=1, key="n_chrono_search_results")
    st.slider(
        "Chronological similarity (%)",
        0, 100,
        value= 75,
        step= 1,
        key="chrono_similarity_threshold"
    )
    group_by_year = st.checkbox("Group results by year", value=True)

    st.slider("Position: min similarity (%)",
            0, 100, value=30, step=5,
            key="position_similarity")
    st.slider("Position: hits per year",
              1, 5, value=3, step=1,
              key="position_hits")

# --- Initialize Search Index Once ---
if not st.session_state.initialized:
    index, embeddings, mapping, pages, year2vec = search.initialize_search_index()
    st.session_state.update({"index": index, "embeddings": embeddings, "mapping": mapping, "pages": pages, "year2vec": year2vec, "initialized": True})


def inject_theme_css():
    st.markdown(
        """
        <style>
        /* ---------- colour tokens ---------- */
        :root {
          /* light mode defaults */
          --card-bg       : #ffffff;
          --card-border   : #cccccc;
          --meta-colour   : #006621;  /* green */
          --link-colour   : #1a0dab;  /* Google‑blue */
          --snippet-colour: #4d5156;  /* grey */

          --disclaimer-fg : black;
          --disclaimer-bg : #fdf6e3;
          --disclaimer-border : #f39c12;
        }

        @media (prefers-color-scheme: dark) {
          :root {
            --card-bg       : #262730;
            --card-border   : #3a3a3a;
            --meta-colour   : #34a853;
            --link-colour   : #8ab4f8;
            --snippet-colour: #e8eaed;

            --disclaimer-fg : white;
            --disclaimer-bg : #333;
            --disclaimer-border : #f39c12;
          }
        }

        /* Footer disclaimer style using tokens */
        .custom-disclaimer {
            color: var(--disclaimer-fg) !important;
            background-color: var(--disclaimer-bg) !important;
            border-left: 4px solid var(--disclaimer-border) !important;
            padding: 1em;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme_css()

def display_result(result):
    similarity_percentage = int(result["weighted_score"] * 100)  # Convert to percentage

    html_content = f"""
    <div style="
        border: 1px solid var(--card-border);
        background: var(--card-bg);
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        font-family: Arial, sans-serif;">

    <p style="color: var(--meta-colour); font-size: 12px; margin: 0;">
        {result['publication_type']} – {result['publication_date']} – {similarity_percentage}% match –
        <a href="{result['pdf_url']}" target="_blank">{result['pdf_url']}</a>
    </p>

    <h3 style="margin: 0; font-size: 18px;">
        <a href="{result['url']}" target="_blank"
            style="color: var(--link-colour); text-decoration: none;">
            {result['title']}
        </a>
    </h3>

    <p style="color: var(--snippet-colour); font-size: 12px; margin-top: 8px;">
        {result['snippet']}
    </p>
    </div>
    """


    st.markdown(html_content, unsafe_allow_html=True)



# --- Layout with Tabs ---
tab_chat, tab_search, tab_chrono, tab_position = st.tabs(["💬 Chat", "❓Search", "📅 Chronological", "📜 Position"])


with tab_search:
    # --- Search Bar ---
    query = st.text_input("Enter your query:", placeholder="Type a sentence or keywords...")

    # --- Perform Search ---
    if query:
        st.write(f"#### Results for: `{query}`")

        # Start timing
        start_time = time.time()

        results = search.search_pdfs(
            query,
            st.session_state.index,
            st.session_state.embeddings,
            st.session_state.mapping,
            st.session_state.pages,
            k                  = st.session_state.n_search_results,
            alpha              = st.session_state.alpha,
            max_snippet_length = st.session_state.max_snippet_length
        )

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time  # Calculate elapsed time

        print(f"🔍 Search completed in **{elapsed_time:.2f} seconds**")

        if results:
            for result in results:
                display_result(result)
        else:
            st.write("No results found.")

def add_message(role: str, content: str):
    st.session_state.chat_history.append(
        {"role": role, "content": content}
    )

with tab_chat:
    if not st.session_state.OPENAI_API_KEY:
        key = st.text_input(
            "🔑 Enter your OpenAI API key",
            type="password",
            key="password_input_2"
        )
        if key:
            st.session_state.OPENAI_API_KEY = key
            st.success("✅ API key saved!")
            st.rerun()
    else:
        # ── get user input ───────────────────────────────────────────
        user_prompt = st.chat_input("Ask me anything about the documents …", key="chatbox")

        # ── 1) one container for every bubble ───────────────────────────
        chat_container = st.container()


        with chat_container:
            # ── 1) render the whole conversation so far (oldest → newest) ──
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])


        if user_prompt:
            # 1 ▸ show it & store
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_prompt)
            add_message("user", user_prompt)


            # 2 ▸ assistant bubble (will be streamed)
            with chat_container.chat_message("assistant"):

                # 2·1 build the context you want to send to the LLM
                #     (you might clip to the last N turns)
                history_for_llm = st.session_state.chat_history[-10:]

                try:
                    # 2·2 fire your RAG wrapper; it returns a generator of chunks
                    stream = search.chat_rag(
                        user_prompt,
                        history      = history_for_llm,                 # NEW
                        faiss_index  = st.session_state.index,
                        embeddings   = st.session_state.embeddings,
                        mapping_df   = st.session_state.mapping,
                        pages_df     = st.session_state.pages,
                        k            = st.session_state.n_search_results,
                        alpha        = st.session_state.alpha,
                        max_snippet_length = st.session_state.max_snippet_length,
                        openai_api_key     = st.session_state.OPENAI_API_KEY,
                        llm_model          = st.session_state.selected_model
                    )

                    # 2·3 stream tokens into the chat bubble
                    assistant_text = st.write_stream(stream) if stream else ""


                # 3  handle a bad / missing key  (or any auth failure during the call)
                except Exception as e:
                    assistant_text = f"❌ {e}\n\nPlease enter a valid key in the sidebar and try again."
                    st.error(assistant_text)

            # 3 ▸ persist the assistant reply
            add_message("assistant", assistant_text)


with tab_chrono:
    chrono_q = st.text_input("Enter your query:", key="chrono_query")

    if chrono_q:
        st.write(f"#### Chronological results for: `{chrono_q}`")
        t0 = time.time()

        # newest → oldest
        years = sorted(st.session_state.year2vec.keys(), reverse=True)
        any_hit = False
        for yr in years:
            hits = search.search_pdfs(
                chrono_q,
                faiss_index   = st.session_state.index,
                embeddings    = st.session_state.embeddings,
                mapping_df    = st.session_state.mapping,
                pages_df      = st.session_state.pages,
                k             = st.session_state.n_chrono_search_results,
                alpha         = st.session_state.alpha,
                max_snippet_length = st.session_state.max_snippet_length,
                threshold     = st.session_state.chrono_similarity_threshold / 100,
                year2vec      = st.session_state.year2vec,
                year          = yr,
            )
            if hits:
                if any_hit:
                    st.markdown(f"---\n## {yr}")
                else:
                    st.markdown(f"## {yr}")

                any_hit = True
                for h in hits:
                    display_result(h)

        if not any_hit:
            st.info("No year contained publications above the threshold.")

        st.caption(f"⏱️ {time.time() - t0:.2f}s")

with tab_position:
    topic_q = st.text_input(
        "What T&E position would you like to trace?",
        placeholder="e.g. indirect land-use change"
    )

    run_timeline = st.button("🔄 Generate timeline", key="run_timeline")

    if run_timeline and topic_q:
        start = time.time()
        timeline_md = search.position_timeline(
            topic_q,
            faiss_index   = st.session_state.index,
            embeddings    = st.session_state.embeddings,
            mapping_df    = st.session_state.mapping,
            pages_df      = st.session_state.pages,
            year2vec      = st.session_state.year2vec,
            openai_api_key= st.session_state.OPENAI_API_KEY,
            alpha         = st.session_state.alpha,
            max_snippet_length = st.session_state.max_snippet_length,
            k_per_year    = st.session_state.position_hits,
            min_score     = st.session_state.position_similarity / 100,
        )
        st.markdown(timeline_md, unsafe_allow_html=True)
        st.caption(f"⏱️ {time.time()-start:.2f}s")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="custom-disclaimer">
        <h4 style="margin-top: 0;">Disclaimer</h4>
        <p style="margin-bottom: 0.5em;">
            The information presented here, including any commentary or analysis, is based solely on publicly available sources.
        </p>
        <p style="margin-bottom: 0.5em;">
            It does not reflect the views, policies, or opinions of <strong>Transport & Environment</strong> or any of its affiliates. This site is not officially associated with or endorsed by Transport & Environment.
        </p>
        <p style="margin-bottom: 0;">
            Use of this site is at your own discretion, and no responsibility is assumed for any reliance on the information provided.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
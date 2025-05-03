import streamlit as st
import search as search
import time

# Set page configuration
st.set_page_config(
    page_title="AI Search",  # Title shown in the browser tab
    page_icon="assets/favicon.ico",                    # Emoji or path to a favicon file
    layout="wide",                     # Other options: "centered"
    initial_sidebar_state="expanded"   # Options: "auto", "expanded", "collapsed"
)

# Sidebar with logo and dropdown
st.sidebar.image("assets/logo.png", use_column_width=True)  # Update with your logo path


# ── 0) CSS for the floating bar (same as before) ────────────────
st.markdown(
    """
    <style>
      /* floating footer */
      .stChatInputContainer {
          position: fixed !important;
          left: 0; right: 0; bottom: 0;
          z-index: 100;
          border-top: 1px solid #eee;
          background: white;
      }
      /* make room so bubbles aren't hidden */
      .block-container { padding-bottom: 6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    "chat_history": []
})

with st.sidebar.expander("Settings"):

    if st.button("🔄  Reset chat", use_container_width=True):
            st.session_state.chat_history = []

            # optional: also clear embeddings / FAISS index, etc.
            # for k in ("index", "embeddings", "mapping", "pages"):
            #     st.session_state.pop(k, None)


    key = st.text_input("🔑 Enter your OpenAI API key", type="password")
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
    st.slider("Length of snippets", 200, 2000, value=1000, step=50, key="max_snippet_length")

# --- Initialize Search Index Once ---
if not st.session_state.initialized:
    index, embeddings, mapping, pages = search.initialize_search_index()
    st.session_state.update({"index": index, "embeddings": embeddings, "mapping": mapping, "pages": pages, "initialized": True})



def display_result(result):
    similarity_percentage = int(result["weighted_score"] * 100)  # Convert to percentage

    # Define the color dynamically (red → orange → green)
    color = "red" if similarity_percentage < 40 else "orange" if similarity_percentage < 70 else "green"

    html_content = f"""
    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-family: Arial, sans-serif;">
            <p style="color: #006621; font-size: 12px; margin: 0;">{result['publication_type']} - {result['publication_date']} - {similarity_percentage}% match - <a href="{result['pdf_url']}" target="_blank"> {result['pdf_url']} </a> </p>
                <h3 style="margin: 0; font-size: 18px;">
                    <a href="{result['url']}" target="_blank" style="color: #1a0dab; text-decoration: none;">
                        {result['title']}
                    </a>
                </h3>
            <p style="color: #4d5156; font-size: 12px; margin-top: 8px;">{result['snippet']}</p>
    </div>
    """

    st.markdown(html_content, unsafe_allow_html=True)

# --- Search Mode Implementation ---


# --- Layout with Tabs ---
tab_search, tab_chat = st.tabs(["❓Search", "💬 Chat"])


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
        key = st.text_input("🔑 Enter your OpenAI API key", type="password", key="password_input_2")
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

            # 3 ▸ persist the assistant reply
            add_message("assistant", assistant_text)



# Footer
st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div style="padding: 1em; border-left: 4px solid #f39c12; background-color: #fdf6e3; border-radius: 5px;">
        <h4 style="margin-top: 0;">Disclaimer</h4>
        <p style="margin-bottom: 0.5em;">
            This website is a personal project developed and maintained by <strong>Alexander Kunkel</strong>. The information presented here, including any commentary or analysis, is based solely on publicly available sources.
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
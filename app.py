import streamlit as st
import search_faiss as search_faiss
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


# --- Session State Initialization ---
if "initialized" not in st.session_state:
    st.session_state.initialized = False  # Tracks if indices are initialized
if "alpha" not in st.session_state:
    st.session_state.alpha = 0.05
if "chunks_before" not in st.session_state:
    st.session_state.chunks_before = 0  # Default within range 0-3
if "chunks_after" not in st.session_state:
    st.session_state.chunks_after = 0  # Default within range 0-3
if "max_snippet_length" not in st.session_state:
    st.session_state.max_snippet_length = 500  # Default within range 200-2000

# --- Filter Options for Publication Type ---
publication_types = ['Briefing', 'Press Release', 'Unknown Type', 'Report', 'Letter',
       'Opinion', 'News', 'Publication', 'Consultation response', 'Internal', 'Spreadsheet']

# Callback for Select All
def select_all():
    st.session_state.selected_types = publication_types

# Callback for Clear All
def clear_all():
    st.session_state.selected_types = []

with st.sidebar.expander("Expert Settings"):
    # Buttons for Select All and Clear All
    st.button("Select All", on_click=select_all)

    st.button("Clear All", on_click=clear_all)

    selected_types = st.multiselect(
        "Filter by Publication Type:",
        options=publication_types,
        default=publication_types,
        key="selected_types"
    )


    alpha = st.slider("Date Decay Factor (alpha)", min_value=0.0, max_value=0.5, step=0.01, value=0.05)


    st.session_state.chunks_before = st.slider(
        "Chunks Before", min_value=0, max_value=3, step=1, value=st.session_state.chunks_before
    )

    st.session_state.chunks_after = st.slider(
        "Chunks After", min_value=0, max_value=3, step=1, value=st.session_state.chunks_after
    )

    st.session_state.max_snippet_length = st.slider(
        "Max Snippet Length", min_value=200, max_value=2000, step=50, value=st.session_state.max_snippet_length
    )

# --- Index Initialization Logic ---
def initialize_index():
    st.sidebar.write("Loading semantic search index...")
    (
        st.session_state.faiss_index,
        st.session_state.embedding_model,
        st.session_state.all_chunks,
        st.session_state.metadata,
        st.session_state.normalise,
    ) = search_faiss.initialize_search_index()
    st.sidebar.success("Semantic search index loaded!")
    st.session_state.initialized = True


# --- Trigger Index Initialization or Update ---
if not st.session_state.initialized:
    # Initialize or Update Index
    initialize_index()


# --- Define Search Function ---
search_function = lambda query: search_faiss.search_pdfs(
    query,
    st.session_state.faiss_index,
    st.session_state.embedding_model,
    st.session_state.all_chunks,
    st.session_state.metadata,
    st.session_state.normalise,
    st.session_state.alpha,
    st.session_state.selected_types,
    st.session_state.chunks_before,
    st.session_state.chunks_after,
    st.session_state.max_snippet_length
)

def display_result(result):
    similarity_percentage = int(result["weighted_score"] * 100)  # Convert to percentage

    # Define the color dynamically (red → orange → green)
    color = "red" if similarity_percentage < 40 else "orange" if similarity_percentage < 70 else "green"

    html_content = f"""
    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-family: Arial, sans-serif;">
            <p style="color: #006621; font-size: 12px; margin: 0;">{result['publication_type']} - {result['publication_date']} - {similarity_percentage}% match </p>
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

# --- Search Bar ---
query = st.text_input("Enter your query:", placeholder="Type a sentence or keywords...")

# --- Perform Search ---
if query:
    st.write(f"#### Results for: `{query}`")

    # Start timing
    start_time = time.time()

    results = search_function(query)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time

    print(f"🔍 Search completed in **{elapsed_time:.2f} seconds**")

    if results:
        for result in results:
            display_result(result)
    else:
        st.write("No results found.")

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
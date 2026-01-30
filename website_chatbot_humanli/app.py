"""
Streamlit App - Main UI for the Website Chatbot
"""

import streamlit as st
import os
from crawler import crawl_website
from chunker import chunk_text
from embeddings import EmbeddingStore
from qa_chain import QAChain
from memory import ChatMemory

# Page configuration
st.set_page_config(
    page_title="Website Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'embedding_store' not in st.session_state:
    st.session_state.embedding_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'memory' not in st.session_state:
    st.session_state.memory = ChatMemory()
if 'indexed' not in st.session_state:
    st.session_state.indexed = False
if 'indexed_url' not in st.session_state:
    st.session_state.indexed_url = ""

# Title
st.title("🤖 Website-Based Chatbot")
st.markdown("Ask questions about any website! First, index a website, then start chatting.")

# Sidebar for website indexing
with st.sidebar:
    st.header("📚 Index Website")
    
    url_input = st.text_input(
        "Enter Website URL",
        placeholder="https://example.com",
        help="Enter the full URL of the website you want to chat about"
    )
    
    if st.button("🔍 Index Website", type="primary", use_container_width=True):
        if not url_input:
            st.error("Please enter a URL!")
        else:
            with st.spinner("Crawling website..."):
                # Crawl website
                result = crawl_website(url_input)
                
                if result:
                    st.success(f"✓ Crawled: {result['title']}")
                    
                    with st.spinner("Chunking text..."):
                        # Chunk text
                        chunks = chunk_text(
                            result['text'],
                            chunk_size=500,
                            chunk_overlap=100,
                            url=result['url'],
                            title=result['title']
                        )
                        st.success(f"✓ Created {len(chunks)} chunks")
                    
                    with st.spinner("Creating embeddings and building index..."):
                        # Create embeddings and build index
                        embedding_store = EmbeddingStore()
                        embeddings = embedding_store.create_embeddings(chunks)
                        embedding_store.build_index(chunks, embeddings)
                        embedding_store.save_index()
                        
                        st.session_state.embedding_store = embedding_store
                        st.session_state.indexed = True
                        st.session_state.indexed_url = url_input
                        
                        # Initialize QA chain
                        st.session_state.qa_chain = QAChain(
                            embedding_store,
                            st.session_state.memory
                        )
                        
                        st.success("✓ Website indexed successfully!")
                        st.balloons()
                else:
                    st.error("Failed to crawl website. Please check the URL and try again.")
    
    # Load existing index
    if st.button("📂 Load Existing Index", use_container_width=True):
        embedding_store = EmbeddingStore()
        if embedding_store.load_index():
            st.session_state.embedding_store = embedding_store
            st.session_state.indexed = True
            st.session_state.qa_chain = QAChain(
                embedding_store,
                st.session_state.memory
            )
            st.success("✓ Index loaded successfully!")
        else:
            st.error("No saved index found. Please index a website first.")
    
    # Clear memory
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.memory.clear()
        st.rerun()
    
    # Status
    st.divider()
    st.subheader("Status")
    if st.session_state.indexed:
        st.success("✓ Website Indexed")
        st.caption(f"URL: {st.session_state.indexed_url}")
    else:
        st.warning("⚠️ No website indexed")

# Main chat area
if st.session_state.indexed:
    st.subheader("💬 Chat")
    
    # Display chat history
    chat_history = st.session_state.memory.get_history()
    
    # Show chat messages
    for i in range(0, len(chat_history), 2):
        if i < len(chat_history):
            # User message
            with st.chat_message("user"):
                st.write(chat_history[i].content if hasattr(chat_history[i], 'content') else str(chat_history[i]))
        
        if i + 1 < len(chat_history):
            # AI message
            with st.chat_message("assistant"):
                st.write(chat_history[i + 1].content if hasattr(chat_history[i + 1], 'content') else str(chat_history[i + 1]))
    
    # Chat input
    user_question = st.chat_input("Ask a question about the website...")
    
    if user_question:
        # Display user message
        with st.chat_message("user"):
            st.write(user_question)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.qa_chain.answer(user_question)
                st.write(answer)
        
        # Rerun to show updated chat history
        st.rerun()
else:
    st.info("👈 Please index a website using the sidebar to start chatting!")
    st.markdown("""
    ### How to use:
    1. Enter a website URL in the sidebar
    2. Click "Index Website" to crawl and process the content
    3. Wait for indexing to complete
    4. Start asking questions!
    
    ### Features:
    - ✅ Website content extraction
    - ✅ Intelligent text chunking
    - ✅ Vector-based similarity search
    - ✅ Context-aware responses
    - ✅ Conversation memory
    """)

# Footer
st.divider()
st.caption("Built with Streamlit, LangChain, and FAISS")


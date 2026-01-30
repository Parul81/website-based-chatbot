# Website-Based Chatbot 🤖

A chatbot that answers questions based solely on content from a specific website. Built with Python, Streamlit, LangChain, and FAISS.

## 🌟 Features

- **Website Crawling**: Extracts clean text content from any website
- **Intelligent Chunking**: Splits text into meaningful chunks with metadata
- **Vector Search**: Uses FAISS for fast similarity search
- **Context-Aware Answers**: Only answers from the indexed website content
- **Conversation Memory**: Remembers context within a session
- **Persistent Storage**: Saves embeddings for reuse

## 🛠️ Tech Stack

- **Python 3.10+**: Programming language
- **Streamlit**: Web UI framework
- **LangChain**: LLM orchestration
- **Sentence Transformers**: Text embeddings
- **FAISS**: Vector similarity search
- **BeautifulSoup4**: HTML parsing
- **OpenAI API**: LLM for generating answers

## 📋 Prerequisites

- Python 3.10 or 3.11
- OpenAI API key (get one from [OpenAI](https://platform.openai.com/api-keys))

## 🚀 Installation

### Step 1: Clone or Download the Project

```bash
cd website_chatbot_humanli
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up OpenAI API Key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual OpenAI API key.

## 🎯 Usage

### Run the Application

```bash
streamlit run app.py
```

The app will open in your browser automatically.

### How to Use

1. **Index a Website**:
   - Enter a website URL in the sidebar
   - Click "Index Website"
   - Wait for the indexing process to complete

2. **Ask Questions**:
   - Once indexed, start asking questions in the chat
   - The bot will answer based only on the website content

3. **Load Existing Index**:
   - Use "Load Existing Index" to reload a previously indexed website

4. **Clear History**:
   - Use "Clear Chat History" to reset conversation memory

## 📁 Project Structure

```
website_chatbot_humanli/
│
├── app.py                 # Streamlit UI
├── crawler.py            # Website content extraction
├── chunker.py            # Text chunking
├── embeddings.py         # Embeddings and FAISS vector store
├── qa_chain.py          # Question answering logic
├── memory.py            # Conversation memory
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── .env                # Environment variables (create this)
│
└── data/
    └── faiss_index/    # Saved embeddings and index
```

## 🧠 How It Works

1. **Crawling**: Extracts text content from the website, removing navigation, headers, and scripts
2. **Chunking**: Splits the text into smaller chunks (500 chars) with overlap (100 chars)
3. **Embedding**: Converts chunks into vector embeddings using sentence transformers
4. **Indexing**: Stores embeddings in FAISS for fast similarity search
5. **Retrieval**: When you ask a question, finds the most relevant chunks
6. **Answering**: Uses OpenAI LLM to generate answers based only on retrieved chunks

## ⚠️ Important Notes

- The chatbot **only answers from the indexed website content**
- If information is not available, it will say: "The answer is not available on the provided website."
- The chatbot does NOT use external knowledge or internet search
- Memory resets when you restart the app

## 🔧 Configuration

You can modify these settings in the code:

- **Chunk size**: Default 500 characters (in `chunker.py`)
- **Chunk overlap**: Default 100 characters (in `chunker.py`)
- **Embedding model**: Default "all-MiniLM-L6-v2" (in `embeddings.py`)
- **LLM model**: Default "gpt-3.5-turbo" (in `qa_chain.py`)
- **Retrieval count**: Default 3 chunks (in `qa_chain.py`)

## 📝 Example

```
User: What is the main topic of this website?
Bot: Based on the website content, the main topic is...

User: Can you tell me more about that?
Bot: [Provides more details from the website]
```

## 🐛 Troubleshooting

**Error: "OPENAI_API_KEY not found"**
- Make sure you created a `.env` file with your API key

**Error: "Failed to crawl website"**
- Check if the URL is correct and accessible
- Some websites may block automated requests

**Error: "No saved index found"**
- Index a website first before trying to load

## 📚 Dependencies

See `requirements.txt` for the complete list of dependencies.

## 🤝 Contributing

This is a learning project. Feel free to experiment and modify!

## 📄 License

This project is for educational purposes.

---

**Built with ❤️ for learning and exploration**


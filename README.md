# Basecode Knowledge Assistant

Fola is an intelligent enterprise chatbot for Basecamp Company. It uses LangChain, a Retrieval-Augmented Generation (RAG) pipeline, and Google's Gemini model to provide smart, context-aware answers from internal company documents.

---

##  Features

Conversational memory (remembers your name, department, work topics)
Document-grounded Q&A using RAG architecture
Personalized responses based on chat history
Gemini-powered language understanding (via `langchain-google-genai`)
Chat saving and loading system
Clean Streamlit UI with sidebar controls
 
---

## Tech Stack

* [Streamlit](https://streamlit.io) – Web UI
* [LangChain](https://www.langchain.com/) – Retrieval & memory chains
* [Google Gemini](https://ai.google.dev) – LLM API (via `langchain-google-genai`)
* [HuggingFace Transformers](https://huggingface.co/) – Embeddings
* [ChromaDB](https://www.trychroma.com/) – Vector store
* `.env` for managing API keys

---

## Directory Structure

```plaintext
.
├── data/                     # Folder containing company knowledge base (.txt files)
├── chroma_db/                # Persisted vector database
├── app.py                    # Main Streamlit app
├── requirements.txt          # dependencies
└── .env                      # API keys (not tracked)
```

---

## .env Configuration

Create a `.env` file in the root with the following keys:

```env
LANGCHAIN_API_KEY=your_langchain_key
HF_TOKEN=your_huggingface_token
GOOGLE_API_KEY=your_google_genai_key
```

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/1sahmuel/Enterprise-Chatbot.git
cd Enterprise-Chatbot
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Add documents**

Put your company `.txt` files inside the `data/` folder.

4. **Run the app**

```bash
streamlit run app.py
```

---

## How It Works

* Loads your `.txt` documents from `./data`
* Splits them into chunks and embeds with HuggingFace
* Stores in Chroma vector DB
* Uses Gemini LLM via LangChain to answer questions based on:

  * Retrieved docs (context)
  * Chat history (memory)
* Answers are strictly grounded and never hallucinate

---

## Screenshot

![image](https://github.com/user-attachments/assets/a15c742d-aad3-46ca-aa78-05fc11136075)


---

## Tips for Better Use

* Start by saying "hi" or introduce yourself: "my name is Sam"
* Ask questions like:

  * “What is our leave policy?”
  * Who is Basecamp
  * I am new employee, tell me bout the health insurance

---

## 🛆 Saving Conversations

You can save or reload past conversations in the sidebar. All sessions are saved in `/conversation_history`.

---

## License

MIT License. See `LICENSE` file.

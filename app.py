import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
import glob
import json
from datetime import datetime
import re

# Configuration
st.set_page_config(
    page_title="Knowledge Base Assistant",
    page_icon="🧠",
    layout="wide"
)

# --- Memory Management Functions ---
def save_conversation_history():
    if "messages" in st.session_state:
        history_data = {
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.messages,
            "user_context": st.session_state.get("user_context", {})
        }
        os.makedirs("./conversation_history", exist_ok=True)
        filename = f"./conversation_history/chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)

def load_conversation_history():
    history_dir = "./conversation_history"
    if not os.path.exists(history_dir):
        return None
    history_files = glob.glob(os.path.join(history_dir, "chat_*.json"))
    if not history_files:
        return None
    latest_file = max(history_files, key=os.path.getctime)
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def extract_user_context_from_conversation():
    context = {}
    if "messages" not in st.session_state:
        return context
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            content = msg["content"].lower()
            if "my name is" in content or "i'm" in content or "i am" in content:
                context["user_mentioned_name"] = True
                context["name_context"] = msg["content"]
            if any(dept in content for dept in ["hr", "it", "finance", "marketing", "sales", "engineering"]):
                context["department_mentioned"] = True
                context["department_context"] = msg["content"]
            if any(word in content for word in ["project", "task", "working on", "assignment"]):
                context["work_context"] = context.get("work_context", [])
                context["work_context"].append(msg["content"])
    return context

def check_data_directory():
    data_dir = "./data"
    if not os.path.exists(data_dir):
        return False, f"Data directory '{data_dir}' does not exist"
    txt_files = glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True)
    if not txt_files:
        return False, f"No .txt files found in '{data_dir}' directory"
    return True, f"Found {len(txt_files)} text files: {[os.path.basename(f) for f in txt_files]}"

def check_environment_variables():
    load_dotenv()
    required_vars = ["GROQ_API_KEY", "LANGCHAIN_API_KEY", "HF_TOKEN", "GOOGLE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        return False, f"Missing environment variables: {missing_vars}"
    return True, "All environment variables found"

@st.cache_resource(show_spinner=False)
def initialize_rag_pipeline():
    env_ok, env_msg = check_environment_variables()
    if not env_ok:
        st.error(f"❌ Environment Error: {env_msg}")
        st.info("💡 Create a `.env` file with your API keys:\n```\nGROQ_API_KEY=your_groq_key\nLANGCHAIN_API_KEY=your_langchain_key\nHF_TOKEN=your_huggingface_token\nGOOGLE_API_KEY=your_google_apikey\n```")
        st.stop()
    data_ok, data_msg = check_data_directory()
    if not data_ok:
        st.error(f"❌ Data Error: {data_msg}")
        st.info("💡 Create a `data` folder in your project root and add your company's .txt files there.")
        st.stop()
    with st.spinner("Initializing knowledge base..."):
        try:
            #lm = ChatGroq(
              #  model_name="gemma2-9b-it",
              #  groq_api_key=os.getenv("GROQ_API_KEY"),
              #  temperature=0.1,
              #  max_tokens=512,
                #streaming=True
           # )
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                max_tokens=None,
                timeout=None,
                max_retries=2,
    
)
            loader = DirectoryLoader(
                "./data",
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            documents = loader.load()
            if not documents:
                st.error("❌ No documents were loaded from the data directory")
                st.stop()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                length_function=len
            )
            splits = text_splitter.split_documents(documents)
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )
            persist_dir = "./chroma_db"
            if os.path.exists(persist_dir):
                vector_store = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=embeddings
                )
            else:
                vector_store = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    persist_directory=persist_dir
                )
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2}
            )
            memory = ConversationBufferWindowMemory(
                k=10,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            prompt_template = """
You are Fola, an intelligent assistant for Basecamp company. Be friendly. User can refer to Basecamp as company. 

Your job is to answer user questions **only** using:
- Verified information from company documents
- The current conversation history

 Do not guess, invent, or assume anything that is not explicitly stated.
 If the answer is unknown, politely say so and offer alternatives.

---

📚 Company Document Context:
{context}

🗣️ Conversation History:
{chat_history}

❓ Current User Question:
{question}

---

🎯 Based on the information above, provide a clear, helpful, and professional answer. 
If the question is unrelated to the documents or context, say:

"I'm sorry, I couldn't find any information on that in the available company resources. Please check with HR or another department."

Your response:
"""


            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question", "chat_history"]
            )
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt},
                return_source_documents=True,
                verbose=False
            )
            return qa_chain
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG pipeline: {str(e)}")
            st.info("💡 Check your API keys and ensure all dependencies are installed correctly.")
            st.stop()

def display_chat():
    if "messages" not in st.session_state:
        previous_history = load_conversation_history()
        if previous_history and st.sidebar.button("📚 Load Previous Conversation"):
            st.session_state.messages = previous_history["messages"]
            st.session_state.user_context = previous_history.get("user_context", {})
            st.success("Previous conversation loaded!")
            st.rerun()
        else:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm Fola, your enterprise assistant for Basecode company. How can I help you today?"}
            ]
    if "user_context" not in st.session_state:
        st.session_state.user_context = {}

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        current_context = extract_user_context_from_conversation()
        st.session_state.user_context.update(current_context)

        with st.chat_message("assistant"):
            try:
                qa_chain = st.session_state.qa_chain
                greetings = ["hi", "hello", "hey", "good morning", "good evening"]

                user_message = prompt.lower().strip()

                if any(greet in user_message for greet in greetings):
                    # Try to detect if user included their name in the message
                    name = None
                    match = re.search(r"(my name is|i am|i'm)\s+([a-zA-Z]+)", user_message)
                    if match:
                        name = match.group(2).capitalize()
                    if name:
                        response = f"Hey {name}! 👋 Nice to meet you. How can I assist you today?"
                        # Save detected name in context for future use
                        st.session_state.user_context["name_context"] = name
                    else:
                        response = "Hi there! 👋 Welcome to Basecamp! How can I help you today?"
                else:
                    result = qa_chain({"question": prompt})
                    response = result["answer"]

                st.markdown(response)
            except Exception as e:
                response = "Sorry, I encountered an error while processing your question."
                st.error(f"Error: {str(e)}")
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        save_conversation_history()

def main():
    st.title("🧠 Basecamp Information Assistant")
    st.markdown("Ask me anything about the company, policies, or procedures.")
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = initialize_rag_pipeline()
    display_chat()
    with st.sidebar:
        st.header("ℹ️ About Fola")
        st.markdown("""
        **Fola** remembers your conversations and provides personalized assistance.
        
        **Memory Features:**
        - Remembers your name and role
        - Tracks ongoing projects
        - References previous questions
        - Builds context over time
        
        **Tips:**
        - Mention your name and department
        - Ask follow-up questions
        - Reference previous topics
        - Use "it", "that", "the project" - I'll understand!
        """)
        st.markdown("---")
        st.header("🧠 Memory Controls")
        if st.session_state.get("user_context"):
            st.markdown("**Remembered About You:**")
            context = st.session_state.user_context
            if "name_context" in context:
                st.text("✓ Your name/intro")
            if "department_context" in context:
                st.text("✓ Your department")
            if "work_context" in context:
                st.text(f"✓ {len(context['work_context'])} work topics")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = [
                    {"role": "assistant", "content": "Hello! I'm Fola, your enterprise assistant for Basecode company. How can I help you today?"}
                ]
                st.session_state.user_context = {}
                st.rerun()
        with col2:
            if st.button("💾 Save Chat"):
                save_conversation_history()
                st.success("Conversation saved!")
        st.markdown("---")
        st.header("📚 Recent Conversations")
        history_dir = "./conversation_history"
        if os.path.exists(history_dir):
            history_files = sorted(glob.glob(os.path.join(history_dir, "chat_*.json")), reverse=True)[:5]
            for file in history_files:
                timestamp = os.path.basename(file).replace("chat_", "").replace(".json", "")
                if st.button(f"Load {timestamp}", key=file):
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            history_data = json.load(f)
                        st.session_state.messages = history_data["messages"]
                        st.session_state.user_context = history_data.get("user_context", {})
                        st.success("Conversation loaded!")
                        st.rerun()
                    except:
                        st.error("Failed to load conversation")

if __name__ == "__main__":
    main()

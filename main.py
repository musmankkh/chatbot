import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords

from openai import OpenAI

# ==============================
# INITIAL SETUP
# ==============================
load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "files"
HISTORY_FILE = "chat_history.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==============================
# GLOBAL STATE
# ==============================
vector_store = None
uploaded_files = []
chat_history = []
word2vec_model = None

# ==============================
# NLP SETUP
# ==============================
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400
)

# ==============================
# WORD2VEC EMBEDDINGS
# ==============================
class Word2VecEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model
        self.vector_size = model.wv.vector_size

    def _vectorize(self, text):
        tokens = [t for t in simple_preprocess(text) if t not in STOP_WORDS]
        vectors = [self.model.wv[t] for t in tokens if t in self.model.wv]
        return sum(vectors) / len(vectors) if vectors else [0] * self.vector_size

    def embed_documents(self, texts):
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text):
        return self._vectorize(text)

# ==============================
# LOAD DOCUMENTS
# ==============================
def load_documents():
    global vector_store, word2vec_model

    documents = []

    for file in os.listdir(UPLOAD_FOLDER):
        path = os.path.join(UPLOAD_FOLDER, file)

        if file.endswith(".pdf"):
            docs = PyPDFLoader(path).load()
        elif file.endswith(".txt"):
            docs = TextLoader(path).load()
        elif file.endswith(".docx"):
            docs = Docx2txtLoader(path).load()
        elif file.endswith(".csv"):
            docs = CSVLoader(path).load()
        else:
            continue

        for d in docs:
            d.metadata["filename"] = file

        documents.extend(docs)
        uploaded_files.append(file)

    if not documents:
        return

    sentences = [simple_preprocess(d.page_content) for d in documents]
    word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)

    embeddings = Word2VecEmbeddings(word2vec_model)
    vector_store = FAISS.from_documents(documents, embeddings)

# ==============================
# SUMMARY & QA LOGIC
# ==============================
def is_summary_request(q):
    return any(word in q.lower() for word in ["summarize", "summary", "overview"])

def generate_generic_summary():
    if not vector_store:
        return "No documents available to summarize."

    docs = vector_store.similarity_search("", k=15)
    content = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Provide a clear, concise, high-level summary of the following documents.
Focus on the main topics, purpose, and key information.

DOCUMENT CONTENT:
{content}
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a professional document summarization assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=700
    )

    return response.choices[0].message.content


def answer_question(question):
    docs = vector_store.similarity_search(question, k=8)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Use the following information to answer the question.

CONTENT:
{context}

QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

# ==============================
# API ROUTES
# ==============================
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Empty question"}), 400

    if is_summary_request(question):
        answer = generate_generic_summary()
    else:
        answer = answer_question(question)

    chat_history.append({
        "question": question,
        "answer": answer,
        "timestamp": datetime.now().isoformat()
    })

    return jsonify({"status": "success", "answer": answer})


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "documents_loaded": len(uploaded_files),
        "history_size": len(chat_history)
    })


# ==============================
# APP START
# ==============================
if __name__ == "__main__":
    load_documents()
    app.run(host="0.0.0.0", port=5000, debug=True)

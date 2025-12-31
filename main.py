import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import json
from datetime import datetime
import pandas as pd
# Document loaders
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings and vector store
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# OpenAI
from openai import OpenAI

# Document schema
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'files'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['HISTORY_FILE'] = 'chat_history.json'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'csv', 'odt'}

# Global variables
vector_stores = {}
merged_vector_store = None
uploaded_files = []
chat_history = []
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
    length_function=len,
)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_chat_history():
    global chat_history
    try:
        if os.path.exists(app.config['HISTORY_FILE']):
            with open(app.config['HISTORY_FILE'], 'r', encoding='utf-8') as f:
                chat_history = json.load(f)
    except Exception as e:
        print(f"Error loading chat history: {e}")
        chat_history = []


def save_chat_history():
    try:
        with open(app.config['HISTORY_FILE'], 'w', encoding='utf-8') as f:
            json.dump(chat_history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving chat history: {e}")


def add_to_history(question, answer, sources=None):
    chat_history.append({
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'answer': answer,
        'sources': sources
    })
    save_chat_history()


def load_document(file_path):
    """Load any document and create rich, comprehensive content for AI"""
    try:
        filename = os.path.basename(file_path)
        
        if file_path.endswith(".csv"):
            # Load CSV and create comprehensive text representation
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                
                # Create ONE comprehensive document with ALL data
                full_content = f"=== CSV FILE: {filename} ===\n\n"
                full_content += f"This file contains {len(df)} rows and {len(df.columns)} columns.\n\n"
                full_content += f"Column names: {', '.join(df.columns.tolist())}\n\n"
                
                # Add detailed column information
                full_content += "DETAILED COLUMN INFORMATION:\n\n"
                for col in df.columns:
                    full_content += f"Column: {col}\n"
                    unique_values = df[col].dropna().unique()
                    full_content += f"  - Total entries: {df[col].count()}\n"
                    full_content += f"  - Unique values: {len(unique_values)}\n"
                    full_content += f"  - All unique values: {', '.join([str(v) for v in unique_values])}\n\n"
                
                # Add complete data (all rows)
                full_content += f"\n\nCOMPLETE DATA ({len(df)} rows):\n\n"
                full_content += df.to_string(index=False)
                
                # Split into manageable chunks for vector store
                docs = text_splitter.create_documents(
                    [full_content],
                    metadatas=[{'filename': filename, 'source': file_path}]
                )
                
                print(f"  📊 CSV loaded: {len(df)} rows, {len(df.columns)} columns → {len(docs)} chunks")
                return docs
                
            except Exception as e:
                print(f"Error loading CSV: {e}")
                loader = CSVLoader(file_path, encoding='utf-8')
                docs = loader.load()
                
        elif file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            docs = text_splitter.split_documents(docs)
            
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            docs = text_splitter.split_documents(docs)
            
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            docs = text_splitter.split_documents(docs)
            
        elif file_path.endswith(".odt"):
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            docs = text_splitter.split_documents(docs)
            
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        # Add filename to all documents
        for doc in docs:
            doc.metadata['filename'] = filename
            doc.metadata['source'] = file_path
        
        return docs
        
    except Exception as e:
        print(f"Error loading document {file_path}: {str(e)}")
        return None


def create_vector_store(docs):
    try:
        return FAISS.from_documents(docs, embeddings)
    except Exception as e:
        raise Exception(f"Error creating vector store: {str(e)}")


def rebuild_merged_vector_store():
    """Merge all vector stores"""
    global merged_vector_store
    
    if not vector_stores:
        merged_vector_store = None
        return
    
    all_docs = []
    for filename, docs in vector_stores.items():
        all_docs.extend(docs)
    
    if all_docs:
        merged_vector_store = create_vector_store(all_docs)
        print(f"✅ Merged vector store created with {len(all_docs)} chunks")


def load_all_files_from_folder():
    """Load all documents from files folder"""
    global uploaded_files, vector_stores
    
    uploaded_files = []
    vector_stores = {}
    
    print(f"🔍 Scanning folder: {app.config['UPLOAD_FOLDER']}")
    
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        print(f"📁 Created folder: {app.config['UPLOAD_FOLDER']}")
        return
    
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    
    if not files:
        print("⚠️ No files found in the folder")
        return
    
    loaded_count = 0
    
    for filename in files:
        if not allowed_file(filename):
            print(f"⏭️ Skipping: {filename}")
            continue
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            print(f"📄 Loading: {filename}")
            
            docs = load_document(file_path)
            
            if not docs:
                print(f"⚠️ Could not load {filename}")
                continue
            
            file_size = os.path.getsize(file_path)
            file_size_mb = round(file_size / (1024 * 1024), 2)
            
            vector_stores[filename] = docs
            
            uploaded_files.append({
                'filename': filename,
                'size_mb': file_size_mb,
                'chunks': len(docs),
                'loaded_at': datetime.now().isoformat()
            })
            
            loaded_count += 1
            print(f"✅ Loaded: {filename} ({file_size_mb} MB, {len(docs)} chunks)")
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {str(e)}")
    
    rebuild_merged_vector_store()
    
    print(f"\n🎉 Successfully loaded {loaded_count} files")


def get_answer(question):
    """Pure AI-driven answer generation - no hardcoded logic"""
    if not merged_vector_store:
        return "No documents are loaded. Please add files to the 'files' folder and restart the server.", None
    
    try:
        # Search for relevant content (more results for better context)
        docs = merged_vector_store.similarity_search(question, k=15)
        
        if not docs:
            return "I couldn't find relevant information in the loaded documents.", None
        
        # Build rich context from all relevant documents
        context = ""
        sources = set()
        
        for doc in docs:
            filename = doc.metadata.get('filename', 'Unknown')
            sources.add(filename)
            context += f"\n\n=== From: {filename} ===\n"
            context += doc.page_content
        
        # Let AI handle EVERYTHING - no restrictions, no logic
        prompt = f"""You are an intelligent assistant with access to document content. Answer the user's question based on the information provided.

DOCUMENTS CONTENT:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer naturally and comprehensively
- If asked for lists, counts, or specific data, extract and provide exact information from the content
- If asked "how many", count accurately from the data
- If asked for "all" or "list", provide complete lists
- If information is in tables or structured data, parse it carefully
- Cite which document(s) you used
- Be thorough and accurate

YOUR ANSWER:"""
        
        # Call OpenAI with no restrictions - let AI figure it out
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a highly capable AI assistant. You excel at understanding and analyzing any type of document content including text, tables, CSV data, reports, and more. You provide accurate, detailed answers by carefully analyzing the provided information."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        answer = response.choices[0].message.content
        
        return answer, list(sources)
    
    except Exception as e:
        return f"Error processing question: {str(e)}", None


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask_question():
    """Ask any question - AI handles everything"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'status': 'error',
                'error': 'No question provided',
                'message': 'Please enter a question'
            }), 400
        
        question = data['question'].strip()
        
        if not question:
            return jsonify({
                'status': 'error',
                'error': 'Empty question',
                'message': 'Please enter a valid question'
            }), 400
        
        # Get answer - AI handles everything
        answer, sources = get_answer(question)
        
        # Add to history
        add_to_history(question, answer, sources)
        
        return jsonify({
            'status': 'success',
            'answer': answer,
            'sources': sources,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': 'Processing failed',
            'message': f'Failed to process question: {str(e)}'
        }), 500


@app.route('/files', methods=['GET'])
def list_files():
    return jsonify({
        'status': 'success',
        'files': uploaded_files,
        'total': len(uploaded_files)
    }), 200


@app.route('/reload', methods=['POST'])
def reload_files():
    """Reload all files"""
    load_all_files_from_folder()
    
    return jsonify({
        'status': 'success',
        'message': f'Reloaded {len(uploaded_files)} files',
        'files': uploaded_files
    }), 200


@app.route('/history', methods=['GET'])
def get_history():
    limit = request.args.get('limit', type=int, default=50)
    return jsonify({
        'status': 'success',
        'history': chat_history[-limit:],
        'total': len(chat_history)
    }), 200


@app.route('/history', methods=['DELETE'])
def clear_history():
    global chat_history
    chat_history = []
    save_chat_history()
    return jsonify({
        'status': 'success',
        'message': 'Chat history cleared'
    }), 200


@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'success',
        'documents_loaded': len(uploaded_files),
        'total_conversations': len(chat_history),
        'files': [f['filename'] for f in uploaded_files]
    }), 200


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'status': 'error',
        'error': 'Not found',
        'message': 'The requested resource was not found'
    }), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'status': 'error',
        'error': 'Server error',
        'message': 'An internal server error occurred. Please try again.'
    }), 500


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    load_chat_history()
    
    print("\n" + "="*60)
    print("🚀 Starting AI-Powered Document Chatbot")
    print("="*60)
    load_all_files_from_folder()
    print("="*60 + "\n")
    
    print(f"📁 Documents folder: {app.config['UPLOAD_FOLDER']}")
    print(f"💬 Chat history: {app.config['HISTORY_FILE']}")
    print(f"✅ Server ready at http://localhost:5000")
    print("\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
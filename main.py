import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import json
from datetime import datetime
# Document loaders
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

# Embeddings and vector store - UPDATED IMPORT
from langchain_openai import OpenAIEmbeddings  # Changed this line
from langchain_community.vectorstores import FAISS

# OpenAI
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'files'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['HISTORY_FILE'] = 'chat_history.json'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}

# Global variables
vector_stores = {}  # Dictionary to store multiple vector databases
uploaded_files = []  # List to track uploaded files
chat_history = []  # Store conversation history
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


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


def add_to_history(question, answer, filename=None):
 
    chat_history.append({
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'answer': answer,
        'filename': filename
    })
    save_chat_history()


def load_document(file_path):

    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Unsupported file type. Use PDF, TXT, or DOCX")
        
        return loader.load()
    except Exception as e:
        raise Exception(f"Error loading document: {str(e)}")


def create_vector_store(docs):
 
    try:
        return FAISS.from_documents(docs, embeddings)
    except Exception as e:
        raise Exception(f"Error creating vector store: {str(e)}")


def merge_vector_stores():

    if not vector_stores:
        return None
    
    # Start with the first vector store
    merged_store = list(vector_stores.values())[0]
    
    # Merge with remaining stores
    for filename, store in list(vector_stores.items())[1:]:
        try:
            merged_store.merge_from(store)
        except:
            pass
    
    return merged_store


def get_answer(question):
  
    if not vector_stores:
        return "Please upload at least one document first.", None
    
    try:
        # Merge all vector stores
        merged_store = merge_vector_stores()
        
        if merged_store is None:
            return "No documents available to search.", None
        
        # Find relevant documents
        docs = merged_store.similarity_search(question, k=5)
        
        if not docs:
            return "I couldn't find relevant information in the uploaded documents.", None
        
        # Create context from documents
        context = "\n\n".join([d.page_content for d in docs])
        
        # Create prompt
        prompt = f"""Answer the question using ONLY the context below. 
If the answer is not in the context, say "I don't have enough information to answer that based on the uploaded documents."

Context:
{context}

Question: {question}

Answer:"""
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        
        # Get source filenames
        sources = list(set([doc.metadata.get('source', 'Unknown') for doc in docs]))
        
        return answer, sources
    
    except Exception as e:
        return f"Error processing question: {str(e)}", None


def get_file_info(filename):
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        size_mb = round(size / (1024 * 1024), 2)
        return {
            'filename': filename,
            'size': size,
            'size_mb': size_mb,
            'path': file_path
        }
    return None


# Routes
@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process a document"""
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'error': 'No file provided',
            'message': 'Please select a file to upload'
        }), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'error': 'No file selected',
            'message': 'Please choose a file before uploading'
        }), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({
            'status': 'error',
            'error': 'Invalid file type',
            'message': 'Please upload a PDF, TXT, or DOCX file'
        }), 400
    
    try:
        # Save file securely
        filename = secure_filename(file.filename)
        
        # Check if file already exists
        if filename in uploaded_files:
            return jsonify({
                'status': 'error',
                'error': 'Duplicate file',
                'message': f'File "{filename}" is already uploaded. Please delete it first or choose a different file.'
            }), 400
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        file_size_mb = round(file_size / (1024 * 1024), 2)
        
        # Load and process document
        docs = load_document(file_path)
        
        if not docs:
            os.remove(file_path)
            return jsonify({
                'status': 'error',
                'error': 'Empty document',
                'message': f'The file "{filename}" appears to be empty or could not be read.'
            }), 400
        
        # Create vector store for this file
        vector_stores[filename] = create_vector_store(docs)
        
        # Add to uploaded files list
        uploaded_files.append({
            'filename': filename,
            'size_mb': file_size_mb,
            'chunks': len(docs),
            'uploaded_at': datetime.now().isoformat()
        })
        
        return jsonify({
            'status': 'success',
            'message': f'Document "{filename}" uploaded successfully!',
            'filename': filename,
            'size_mb': file_size_mb,
            'chunks': len(docs),
            'total_files': len(uploaded_files)
        }), 200
    
    except Exception as e:
        # Clean up if error occurs
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify({
            'status': 'error',
            'error': 'Processing failed',
            'message': f'Failed to process file: {str(e)}'
        }), 500


@app.route('/ask', methods=['POST'])
def ask_question():
   
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
        
        if len(question) < 3:
            return jsonify({
                'status': 'error',
                'error': 'Question too short',
                'message': 'Please enter a more detailed question'
            }), 400
        
        # Get answer
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


@app.route('/files/<filename>', methods=['DELETE'])
def delete_file(filename):
   
    try:
        # Remove from vector stores
        if filename in vector_stores:
            del vector_stores[filename]
        
        # Remove from uploaded files list
        global uploaded_files
        uploaded_files = [f for f in uploaded_files if f['filename'] != filename]
        
        # Delete physical file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify({
            'status': 'success',
            'message': f'File "{filename}" deleted successfully',
            'remaining_files': len(uploaded_files)
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': 'Delete failed',
            'message': f'Failed to delete file: {str(e)}'
        }), 500


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
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'status': 'error',
        'error': 'File too large',
        'message': 'File size exceeds 16MB limit. Please upload a smaller file.'
    }), 413


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
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Load chat history
    load_chat_history()
    
    # Run the app
    print("🚀 Flask server starting...")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"💬 Chat history file: {app.config['HISTORY_FILE']}")
    app.run(debug=True, host='0.0.0.0', port=5000)
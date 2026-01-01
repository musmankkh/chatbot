import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import json
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import boto3
from botocore.exceptions import ClientError
import io

# OpenAI
from openai import OpenAI

# Doc2Vec
from gensim.models.doc2vec import Doc2Vec

# BM25
from rank_bm25 import BM25Okapi

# Load environment variables
load_dotenv()


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = '../shared/files'
app.config['HISTORY_FILE'] = 'chat_history.json'

# S3 Configuration
app.config['S3_BUCKET_NAME'] = os.getenv('S3_BUCKET_NAME', 'musmankkh-chatbot-models')
app.config['S3_MODEL_KEY'] = 'models/doc2vec_model.bin'
app.config['S3_METADATA_KEY'] = 'models/model_metadata.pkl'
app.config['AWS_REGION'] = os.getenv('AWS_REGION', 'us-east-1')

# Global variables
chat_history = []
openai_api_key = ""
# Model globals
doc2vec_model = None
document_texts = []
document_metadata = []
bm25_model = None
bm25_corpus = []
uploaded_files = []

# S3 Client
s3_client = None

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)


def initialize_s3_client():
    """Initialize S3 client with credentials from environment"""
    global s3_client
    
    try:
        aws_access_key = ""
        aws_secret_key = ""
        aws_region = ""
        
        if aws_access_key and aws_secret_key:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )
            print("✅ S3 client initialized with explicit credentials")
        else:
            # Try using default credentials (IAM role, AWS CLI config, etc.)
            s3_client = boto3.client('s3', region_name=aws_region)
            print("✅ S3 client initialized with default credentials")
        
        return True
    except Exception as e:
        print(f"❌ Failed to initialize S3 client: {e}")
        return False


def read_from_s3(s3_key):
    """Read a file directly from S3 into memory"""
    try:
        bucket_name = app.config['S3_BUCKET_NAME']
        
        print(f"📥 Reading s3://{bucket_name}/{s3_key} directly from S3...")
        
        # Get object from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        
        # Read the content into memory
        file_content = response['Body'].read()
        
        print(f"✅ Read successfully ({len(file_content)} bytes)")
        return file_content
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            print(f"❌ File not found in S3: s3://{bucket_name}/{s3_key}")
        else:
            print(f"❌ S3 read error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error reading from S3: {e}")
        return None


def list_s3_models():
    """List available model files in S3"""
    try:
        bucket_name = app.config['S3_BUCKET_NAME']
        prefix = 'models/'
        
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )
        
        if 'Contents' in response:
            files = []
            print(f"\n📂 Files in s3://{bucket_name}/{prefix}:")
            for obj in response['Contents']:
                size_mb = obj['Size'] / (1024 * 1024)
                print(f"   - {obj['Key']} ({size_mb:.2f} MB)")
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat()
                })
            return files
        else:
            print(f"⚠️ No files found in s3://{bucket_name}/{prefix}")
            return []
    except Exception as e:
        print(f"❌ Error listing S3 files: {e}")
        return []


def preprocess_text(text):
    """Preprocess text for querying"""
    import re
    from nltk.corpus import stopwords
    
    stop_words = set(stopwords.words('english'))
    important_words = {'not', 'no', 'more', 'most', 'very', 'only', 'same', 'different'}
    stop_words = stop_words - important_words
    
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\-\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = text.split()
    
    filtered_tokens = []
    for token in tokens:
        if (token not in stop_words or 
            token.replace('.','').replace('-','').isdigit() or
            len(token) > 1):
            filtered_tokens.append(token)
    
    return filtered_tokens

def load_models():
    """Load pre-trained models from S3 via temporary local files"""
    global doc2vec_model, document_texts, document_metadata
    global bm25_model, bm25_corpus, uploaded_files
    
    print("\n" + "="*60)
    print("📂 LOADING MODELS FROM S3")
    print("="*60)
    
    # Initialize S3 client
    if not initialize_s3_client():
        print("❌ Cannot proceed without S3 access")
        print("💡 Please check your AWS credentials in .env file")
        return False
    
    # List available files
    list_s3_models()
    
    # Create temp directory for model files
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Read Doc2Vec model from S3
        print(f"\n🧠 Reading Doc2Vec model from S3...")
        model_content = read_from_s3(app.config['S3_MODEL_KEY'])
        
        if model_content is None:
            print("❌ Failed to read Doc2Vec model from S3")
            return False
        
        # Save to temporary file (Gensim requires file path)
        temp_model_path = os.path.join(temp_dir, 'doc2vec_model.bin')
        with open(temp_model_path, 'wb') as f:
            f.write(model_content)
        
        print(f"🧠 Loading Doc2Vec model from temp file...")
        doc2vec_model = Doc2Vec.load(temp_model_path)
        
        print(f"✅ Doc2Vec model loaded")
        print(f"   📊 Vocabulary: {len(doc2vec_model.wv)} words")
        print(f"   📐 Vector size: {doc2vec_model.vector_size}")
        
        # Read metadata from S3
        print(f"\n📋 Reading metadata from S3...")
        metadata_content = read_from_s3(app.config['S3_METADATA_KEY'])
        
        if metadata_content is None:
            print("❌ Failed to read metadata from S3")
            return False
        
        # Load metadata from memory buffer
        print(f"📋 Loading metadata from memory...")
        metadata_buffer = io.BytesIO(metadata_content)
        metadata = pickle.load(metadata_buffer)
        
        document_texts = metadata['document_texts']
        document_metadata = metadata['document_metadata']
        bm25_corpus = metadata['bm25_corpus']
        bm25_model = metadata['bm25_model']
        uploaded_files = metadata['uploaded_files']
        
        print(f"✅ Metadata loaded")
        print(f"   📄 Documents: {len(document_texts)}")
        print(f"   📁 Files: {len(uploaded_files)}")
        print(f"   🔍 BM25 corpus: {len(bm25_corpus)} documents")
        print(f"   🕐 Trained at: {metadata.get('trained_at', 'Unknown')}")
        
        print("\n" + "="*60)
        print("✅ MODELS LOADED SUCCESSFULLY FROM S3")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading models from S3: {e}")
        import traceback
        print(traceback.format_exc())
        print("\n💡 Troubleshooting:")
        print("   1. Verify S3 bucket exists: s3://musmankkh-chatbot-models/")
        print("   2. Check model files are uploaded to: s3://musmankkh-chatbot-models/models/")
        print("   3. Verify AWS credentials have S3 read permissions")
        print("   4. Ensure the following files exist:")
        print("      - models/doc2vec_model.bin")
        print("      - models/model_metadata.pkl")
        return False
    
    finally:
        # Clean up temporary files
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"🧹 Cleaned up temporary files")
        except Exception as e:
            print(f"⚠️ Warning: Could not clean temp directory: {e}")
            
def expand_query(query, top_n=3):
    """Expand query with similar words"""
    if doc2vec_model is None:
        return query
    
    tokens = preprocess_text(query)
    expanded = set(tokens)
    
    for token in tokens:
        if token in doc2vec_model.wv:
            try:
                similar = doc2vec_model.wv.most_similar(token, topn=top_n)
                for word, score in similar:
                    if score > 0.65:
                        expanded.add(word)
            except:
                pass
    
    return ' '.join(expanded)


def search_with_doc2vec(query, k=15):
    """Search documents using Doc2Vec"""
    if doc2vec_model is None or len(document_texts) == 0:
        return []
    
    expanded_query = expand_query(query)
    query_tokens = preprocess_text(expanded_query)
    
    if not query_tokens:
        return []
    
    query_vector = doc2vec_model.infer_vector(query_tokens, epochs=20)
    
    similarities = []
    for idx in range(len(document_texts)):
        doc_vector = doc2vec_model.dv[idx]
        sim = cosine_similarity([query_vector], [doc_vector])[0][0]
        similarities.append((idx, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for idx, sim in similarities[:k]:
        if sim > 0.1:
            results.append({
                'content': document_texts[idx],
                'metadata': document_metadata[idx],
                'similarity': float(sim),
                'method': 'doc2vec'
            })
    
    return results


def search_with_bm25(query, k=15):
    """Search documents using BM25"""
    if bm25_model is None:
        return []
    
    query_tokens = preprocess_text(query)
    if not query_tokens:
        return []
    
    scores = bm25_model.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                'content': document_texts[idx],
                'metadata': document_metadata[idx],
                'similarity': float(scores[idx]),
                'method': 'bm25'
            })
    
    return results


def hybrid_search(query, k=15, alpha=0.6):
    """Hybrid search combining Doc2Vec + BM25"""
    if doc2vec_model is None or bm25_model is None:
        return []
    
    doc2vec_results = search_with_doc2vec(query, k=k*2)
    bm25_results = search_with_bm25(query, k=k*2)
    
    doc2vec_scores = {}
    for result in doc2vec_results:
        for idx, text in enumerate(document_texts):
            if text == result['content']:
                doc2vec_scores[idx] = result['similarity']
                break
    
    bm25_scores = {}
    for result in bm25_results:
        for idx, text in enumerate(document_texts):
            if text == result['content']:
                bm25_scores[idx] = result['similarity']
                break
    
    if bm25_scores:
        max_bm25 = max(bm25_scores.values())
        if max_bm25 > 0:
            bm25_scores = {idx: score/max_bm25 for idx, score in bm25_scores.items()}
    
    combined_scores = {}
    all_indices = set(doc2vec_scores.keys()) | set(bm25_scores.keys())
    
    for idx in all_indices:
        d2v_score = doc2vec_scores.get(idx, 0)
        bm25_score = bm25_scores.get(idx, 0)
        combined_scores[idx] = alpha * d2v_score + (1 - alpha) * bm25_score
    
    sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    results = []
    for idx, score in sorted_indices:
        if score > 0.1:
            results.append({
                'content': document_texts[idx],
                'metadata': document_metadata[idx],
                'similarity': float(score),
                'method': 'hybrid'
            })
    
    return results


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


def is_greeting(question):
    """Check if the question is a greeting"""
    greetings = [
        'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon',
        'good evening', 'howdy', 'hola', 'sup', "what's up", 'whats up'
    ]
    question_lower = question.lower().strip()
    
    # Check if the entire message is just a greeting (or greeting + punctuation)
    clean_question = question_lower.rstrip('!?.;, ')
    
    return clean_question in greetings or any(question_lower.startswith(g) for g in greetings)


def is_summary_request(question):
    """Check if the question is asking for a summary"""
    summary_keywords = [
        'summary', 'summarize', 'summarise', 'overview', 'brief', 
        'gist', 'main points', 'key points', 'in short', 'tldr',
        'what is this about', 'what does this document', 'explain this document'
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in summary_keywords)


def get_answer(question):
    """Get answer using hybrid search"""
    if doc2vec_model is None:
        return "❌ Models not loaded. Please check S3 connection and reload models.", None
    
    try:
        # Check if it's a greeting
        if is_greeting(question):
            file_list = ", ".join([f['filename'] for f in uploaded_files]) if uploaded_files else "no documents yet"
            greeting_response = f"""Hello! 👋 I'm your AI document assistant. I'm here to help you find information from your documents.

Currently loaded: {file_list}

You can ask me:
- Questions about the content in your documents
- To summarize any document
- To find specific information or data
- To analyze CSV files and extract data

What would you like to know?"""
            return greeting_response, None
        
        is_summary = is_summary_request(question)
        k = 15 if is_summary else 12
        results = hybrid_search(question, k=k, alpha=0.6)
        
        if not results:
            results = search_with_bm25(question, k=k)
            
        if not results:
            available_files = ", ".join(set([m.get('filename', 'Unknown') for m in document_metadata]))
            return f"I couldn't find relevant information. Available documents: {available_files}", None
        
        context = ""
        sources = set()
        max_context_length = 8000
        
        for result in results:
            filename = result['metadata'].get('filename', 'Unknown')
            chunk_type = result['metadata'].get('chunk_type', 'content')
            sources.add(filename)
            
            new_content = f"\n\n=== From: {filename}"
            if chunk_type:
                new_content += f" [{chunk_type}]"
            new_content += f" (relevance: {result['similarity']:.3f}) ===\n"
            new_content += result['content']
            
            if len(context) + len(new_content) > max_context_length:
                remaining = max_context_length - len(context)
                if remaining > 300:
                    new_content = new_content[:remaining] + "\n[Content truncated...]"
                    context += new_content
                break
            
            context += new_content
        
        if is_summary:
            prompt = f"""You are an intelligent assistant analyzing document content. Provide a concise summary.

DOCUMENTS CONTENT:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Provide a SINGLE, CONCISE PARAGRAPH summary (4-6 sentences maximum)
- Capture main themes, key points, and overall message
- Write in clear, flowing narrative style
- Do NOT use bullet points, lists, or multiple paragraphs
- Mention which document(s) the summary is based on

YOUR SUMMARY (ONE PARAGRAPH ONLY):"""
        else:
            prompt = f"""You are an intelligent assistant with access to document content. Answer accurately.

DOCUMENTS CONTENT:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer directly and accurately
- For CSV/tabular data: extract exact values, counts, and lists
- If asked "how many", count ALL matching items
- If asked for "list" or "all", provide COMPLETE lists
- Parse structured data carefully
- Cite which document(s) you used
- Be thorough, precise, and data-driven

YOUR ANSWER:"""
        
        system_message = """You are a capable AI assistant specialized in analyzing documents and structured data. You excel at understanding CSV files, extracting precise information, and providing accurate answers based on provided content."""
        
        if is_summary:
            system_message += " When asked for a summary, provide exactly ONE concise paragraph."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0 if not is_summary else 0.3,
            max_tokens=2000
        )
        
        answer = response.choices[0].message.content
        return answer, list(sources)
    
    except Exception as e:
        import traceback
        print(f"Error: {traceback.format_exc()}")
        return f"Error processing question: {str(e)}", None


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'status': 'error',
                'error': 'No question provided'
            }), 400
        
        question = data['question'].strip()
        
        if not question:
            return jsonify({
                'status': 'error',
                'error': 'Empty question'
            }), 400
        
        answer, sources = get_answer(question)
        add_to_history(question, answer, sources)
        
        return jsonify({
            'status': 'success',
            'answer': answer,
            'sources': sources,
            'method': 'hybrid (doc2vec + bm25)',
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to process: {str(e)}'
        }), 500


@app.route('/files', methods=['GET'])
def list_files():
    return jsonify({
        'status': 'success',
        'files': uploaded_files,
        'total': len(uploaded_files),
        'doc2vec_trained': doc2vec_model is not None,
        'doc2vec_vocab_size': len(doc2vec_model.wv) if doc2vec_model else 0,
        'document_count': len(document_texts),
        'bm25_trained': bm25_model is not None
    }), 200


@app.route('/reload', methods=['POST'])
def reload_files():
    """Reload models from S3"""
    success = load_models()
    
    if success:
        return jsonify({
            'status': 'success',
            'message': 'Models reloaded successfully from S3',
            'files': uploaded_files
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'message': 'Failed to load models from S3. Check logs for details.'
        }), 400


@app.route('/s3/list', methods=['GET'])
def list_s3_files():
    """List files in S3 bucket"""
    if not s3_client:
        if not initialize_s3_client():
            return jsonify({
                'status': 'error',
                'message': 'S3 client not initialized. Check AWS credentials.'
            }), 500
    
    files = list_s3_models()
    return jsonify({
        'status': 'success',
        'bucket': app.config['S3_BUCKET_NAME'],
        'files': files,
        'count': len(files)
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
        'files': [f['filename'] for f in uploaded_files],
        'embedding_method': 'Doc2Vec + BM25 Hybrid',
        'doc2vec': {
            'trained': doc2vec_model is not None,
            'vocabulary_size': len(doc2vec_model.wv) if doc2vec_model else 0,
            'vector_size': doc2vec_model.vector_size if doc2vec_model else 0,
            'document_count': len(document_texts)
        },
        'bm25': {
            'trained': bm25_model is not None,
            'corpus_size': len(bm25_corpus)
        },
        's3': {
            'enabled': s3_client is not None,
            'bucket': app.config['S3_BUCKET_NAME'],
            'model_key': app.config['S3_MODEL_KEY'],
            'metadata_key': app.config['S3_METADATA_KEY']
        }
    }), 200


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    load_chat_history()
    
    print("\n" + "="*60)
    print("🚀 AI-POWERED DOCUMENT CHATBOT (S3 STREAMING MODE)")
    print("="*60)
    
    # Load pre-trained models directly from S3 (no local files)
    models_loaded = load_models()
    
    if not models_loaded:
        print("\n⚠️ WARNING: Models not loaded from S3!")
        print("\n💡 Troubleshooting checklist:")
        print("   1. Check .env file has AWS credentials")
        print("   2. Verify S3 bucket: s3://musmankkh-chatbot-models/")
        print("   3. Confirm model files exist in: s3://musmankkh-chatbot-models/models/")
        print("   4. Test S3 connection: python test_s3_connection.py")
        print("\nServer will start but queries won't work until models are loaded.")
    
    print("\n" + "="*60)
    print(f"📁 Documents folder: {app.config['UPLOAD_FOLDER']}")
    print(f"💬 Chat history: {app.config['HISTORY_FILE']}")
    print(f"☁️  S3 Bucket: {app.config['S3_BUCKET_NAME']}")
    print(f"📦 Model Key: {app.config['S3_MODEL_KEY']}")
    print(f"📦 Metadata Key: {app.config['S3_METADATA_KEY']}")
    print(f"✅ Server ready at http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
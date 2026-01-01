import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

# Document loaders
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords

# BM25
from rank_bm25 import BM25Okapi

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Configuration
UPLOAD_FOLDER = '../shared/files'
D2V_MODEL_PATH = '../models/doc2vec_model.bin'
METADATA_PATH = '../models/model_metadata.pkl'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'csv', 'odt'}

# Setup stopwords
stop_words = set(stopwords.words('english'))
important_words = {'not', 'no', 'more', 'most', 'very', 'only', 'same', 'different'}
stop_words = stop_words - important_words

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_text(text, keep_structure=False):
    """Enhanced preprocessing that keeps more information"""
    import re
    
    if keep_structure:
        return text
    
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


def load_document(file_path):
    """Load any document and create rich content"""
    try:
        filename = os.path.basename(file_path)
        
        if file_path.endswith(".csv"):
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                chunks = []
                
                # CHUNK 1: Overview
                overview = f"=== CSV FILE: {filename} ===\n\n"
                overview += f"This CSV file contains {len(df)} data rows and {len(df.columns)} columns.\n\n"
                overview += f"Total entries including header: {len(df) + 1} rows\n\n"
                overview += f"Column names: {', '.join(df.columns.tolist())}\n\n"
                overview += "COLUMN DETAILS:\n"
                
                for col in df.columns:
                    overview += f"\n{col}:\n"
                    overview += f"  - Type: {df[col].dtype}\n"
                    overview += f"  - Total entries: {df[col].count()}\n"
                    overview += f"  - Missing values: {df[col].isna().sum()}\n"
                    
                    unique_values = df[col].dropna().unique()
                    if len(unique_values) <= 20:
                        overview += f"  - Unique values ({len(unique_values)}): {', '.join([str(v) for v in unique_values])}\n"
                    else:
                        overview += f"  - Unique values: {len(unique_values)} (too many to list)\n"
                        sample = df[col].dropna().head(5).tolist()
                        overview += f"  - Sample values: {', '.join([str(v) for v in sample])}\n"
                
                chunks.append(Document(
                    page_content=overview,
                    metadata={'filename': filename, 'source': file_path, 'chunk_type': 'overview'}
                ))
                
                # CHUNK 2: Statistics
                stats = f"\n=== STATISTICAL SUMMARY: {filename} ===\n\n"
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    stats += "Numeric columns statistics:\n\n"
                    for col in numeric_cols:
                        stats += f"{col}:\n"
                        stats += f"  - Mean: {df[col].mean():.2f}\n"
                        stats += f"  - Min: {df[col].min()}\n"
                        stats += f"  - Max: {df[col].max()}\n"
                        stats += f"  - Median: {df[col].median():.2f}\n\n"
                
                chunks.append(Document(
                    page_content=stats,
                    metadata={'filename': filename, 'source': file_path, 'chunk_type': 'statistics'}
                ))
                
                # CHUNK 3-N: Data rows
                rows_per_chunk = 50
                for start_idx in range(0, len(df), rows_per_chunk):
                    end_idx = min(start_idx + rows_per_chunk, len(df))
                    chunk_df = df.iloc[start_idx:end_idx]
                    
                    data_chunk = f"\n=== DATA ROWS {start_idx+1}-{end_idx}: {filename} ===\n\n"
                    data_chunk += f"Columns: {', '.join(df.columns.tolist())}\n\n"
                    data_chunk += chunk_df.to_string(index=False)
                    
                    chunks.append(Document(
                        page_content=data_chunk,
                        metadata={
                            'filename': filename, 
                            'source': file_path, 
                            'chunk_type': 'data',
                            'row_range': f'{start_idx+1}-{end_idx}'
                        }
                    ))
                
                print(f"  📊 CSV loaded: {len(df)} rows, {len(df.columns)} columns → {len(chunks)} chunks")
                return chunks
                
            except Exception as e:
                print(f"Error loading CSV: {e}, using fallback")
                loader = CSVLoader(file_path, encoding='utf-8')
                docs = loader.load()
                docs = text_splitter.split_documents(docs)
                
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
        
        for doc in docs:
            if 'filename' not in doc.metadata:
                doc.metadata['filename'] = filename
            if 'source' not in doc.metadata:
                doc.metadata['source'] = file_path
        
        return docs
        
    except Exception as e:
        print(f"Error loading document {file_path}: {str(e)}")
        return None


def train_models():
    """Train Doc2Vec and BM25 models on all documents in the files folder"""
    
    print("\n" + "="*60)
    print("🚀 TRAINING MODELS")
    print("="*60)
    
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        print(f"📁 Created folder: {UPLOAD_FOLDER}")
        print("⚠️ No files to train on. Add files to the 'files' folder.")
        return
    
    files = os.listdir(UPLOAD_FOLDER)
    
    if not files:
        print("⚠️ No files found in the folder")
        return
    
    # Load all documents
    all_docs = []
    uploaded_files = []
    
    print(f"\n🔍 Loading documents from {UPLOAD_FOLDER}...")
    
    for filename in files:
        if not allowed_file(filename):
            print(f"⏭️ Skipping: {filename}")
            continue
        
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        try:
            print(f"📄 Loading: {filename}")
            docs = load_document(file_path)
            
            if not docs:
                print(f"⚠️ Could not load {filename}")
                continue
            
            file_size = os.path.getsize(file_path)
            file_size_mb = round(file_size / (1024 * 1024), 2)
            
            all_docs.extend(docs)
            
            uploaded_files.append({
                'filename': filename,
                'size_mb': file_size_mb,
                'chunks': len(docs),
                'loaded_at': datetime.now().isoformat()
            })
            
            print(f"✅ Loaded: {filename} ({file_size_mb} MB, {len(docs)} chunks)")
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {str(e)}")
    
    if not all_docs:
        print("\n⚠️ No documents loaded. Nothing to train.")
        return
    
    print(f"\n📋 Total documents loaded: {len(all_docs)}")
    print("\n🧠 Training Doc2Vec model...")
    
    # Prepare training data
    tagged_documents = []
    document_texts = []
    document_metadata = []
    bm25_corpus = []
    
    for idx, doc in enumerate(all_docs):
        text = doc.page_content
        document_texts.append(text)
        document_metadata.append(doc.metadata)
        
        tokens = preprocess_text(text, keep_structure=False)
        
        if tokens:
            tagged_documents.append(TaggedDocument(tokens, [idx]))
            bm25_corpus.append(tokens)
    
    print(f"📊 Prepared {len(tagged_documents)} document chunks for training")
    
    # Train Doc2Vec
    doc2vec_model = Doc2Vec(
        vector_size=300,
        window=15,
        min_count=1,
        workers=4,
        epochs=50,
        dm=0,
        dbow_words=1,
        negative=20,
        alpha=0.025,
        min_alpha=0.0001,
        sample=0,
        hs=0
    )
    
    doc2vec_model.build_vocab(tagged_documents)
    
    print(f"🔄 Training Doc2Vec (this may take a minute)...")
    doc2vec_model.train(
        tagged_documents, 
        total_examples=doc2vec_model.corpus_count, 
        epochs=doc2vec_model.epochs
    )
    
    print(f"✅ Doc2Vec trained!")
    print(f"   📊 Vocabulary: {len(doc2vec_model.wv)} words")
    print(f"   📐 Vector size: {doc2vec_model.vector_size}")
    
    # Train BM25
    print(f"\n🔍 Creating BM25 index...")
    bm25_model = BM25Okapi(bm25_corpus)
    print(f"✅ BM25 index created")
    
    # Save models and metadata
    print(f"\n💾 Saving models...")
    
    try:
        # Save Doc2Vec model
        doc2vec_model.save(D2V_MODEL_PATH)
        print(f"✅ Doc2Vec model saved to {D2V_MODEL_PATH}")
        
        # Save metadata (BM25, texts, metadata)
        metadata = {
            'document_texts': document_texts,
            'document_metadata': document_metadata,
            'bm25_corpus': bm25_corpus,
            'bm25_model': bm25_model,
            'uploaded_files': uploaded_files,
            'trained_at': datetime.now().isoformat()
        }
        
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Metadata saved to {METADATA_PATH}")
        
    except Exception as e:
        print(f"❌ Error saving models: {e}")
        return
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"📊 Total files: {len(uploaded_files)}")
    print(f"📄 Total chunks: {len(document_texts)}")
    print(f"🧠 Doc2Vec vocabulary: {len(doc2vec_model.wv)} words")
    print(f"🔍 BM25 corpus: {len(bm25_corpus)} documents")
    print("\n💡 You can now start the chatbot with: python app.py")
    print("="*60 + "\n")


if __name__ == '__main__':
    train_models()
#!/bin/bash

# Script to reorganize your existing repository
# Run this in your local repository

echo "🔄 Reorganizing repository structure..."

# Create new directory structure
mkdir -p training chatbot shared scripts .github/workflows

# Move training files
echo "📦 Moving training files..."
mv train_model.py training/
cp requirements.txt training/requirements.txt

# Move chatbot files
echo "🤖 Moving chatbot files..."
mv main.py chatbot/
cp requirements.txt chatbot/requirements.txt

# Move shared files
echo "📁 Moving shared files..."
if [ -d "files" ]; then
    mv files shared/
fi

# Keep models directory but add to gitignore
echo "models/" >> .gitignore
echo "*.pkl" >> .gitignore
echo "*.joblib" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo ".env" >> .gitignore
echo "*.log" >> .gitignore
echo "nohup.out" >> .gitignore

# Create requirements.txt for training (heavy dependencies)
cat > training/requirements.txt << 'EOF'
# Training Dependencies
pandas
numpy
scikit-learn
langchain
langchain-community
openai
gensim
nltk
rank-bm25
PyPDF2
docx2txt
python-docx
boto3
python-dotenv
flask  # For webhook listener
EOF

# Create requirements.txt for chatbot (runtime only)
cat > chatbot/requirements.txt << 'EOF'
# Runtime Dependencies
flask
flask-cors
python-dotenv
pandas
numpy
scikit-learn
langchain
langchain-community
openai
gensim
nltk
rank-bm25
boto3
joblib
EOF

# Create README for training
cat > training/README.md << 'EOF'
# Training Server

This folder contains the model training code.

## Setup
```bash
cd training
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python train_model.py
```

## Deployment
This runs on the Training EC2 instance.
EOF

# Create README for chatbot
cat > chatbot/README.md << 'EOF'
# Chatbot Server

This folder contains the chatbot API.

## Setup
```bash
cd chatbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## Deployment
This runs on the Chatbot EC2 instance.
EOF

# Create deployment script for training
cat > training/deploy_training.sh << 'EOF'
#!/bin/bash

echo "🚀 Deploying Training Server..."

cd /home/ec2-user/chatbot/training

# Activate virtual environment
source ../venv/bin/activate

# Pull latest code
git pull origin main

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo "✅ Training server deployed!"
EOF

chmod +x training/deploy_training.sh

# Create deployment script for chatbot
cat > chatbot/deploy_chatbot.sh << 'EOF'
#!/bin/bash

echo "🚀 Deploying Chatbot Server..."

cd /home/ec2-user/chatbot/chatbot

# Activate virtual environment
source ../venv/bin/activate

# Pull latest code
git pull origin main

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Download latest models from S3
aws s3 sync s3://my-chatbot-models-bucket/models/ ../models/

echo "✅ Chatbot server deployed!"
EOF

chmod +x chatbot/deploy_chatbot.sh

echo "✅ Repository reorganized!"
echo ""
echo "📝 Next steps:"
echo "1. Review the changes"
echo "2. Update import paths in your code if needed"
echo "3. Test locally"
echo "4. Commit and push:"
echo "   git add ."
echo "   git commit -m 'Reorganize for split deployment'"
echo "   git push origin main"
#!/bin/bash

# Training Server Setup Script for musmankkh/chatbot
# Run this on Training EC2

echo "🎓 Setting up Training Server..."

# Update system
sudo yum update -y
sudo yum install -y git python3-pip

# Clone your repository
cd /home/ec2-user
if [ -d "chatbot" ]; then
    echo "Repository already exists, pulling latest..."
    cd chatbot
    git pull origin main
else
    echo "Cloning repository..."
    git clone https://github.com/musmankkh/chatbot.git
    cd chatbot
fi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Create requirements.txt for training if it doesn't exist
cd training
if [ ! -f "requirements.txt" ]; then
    cat > requirements.txt << 'EOF'
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
psutil
EOF
fi

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Create models directory in repo root
mkdir -p ../models
mkdir -p logs

# Configure AWS CLI
echo ""
echo "⚙️ Configure AWS CLI for S3 access:"
aws configure

# Create S3 upload script
cat > deploy_to_s3.sh << 'EOF'
#!/bin/bash

# S3 Upload Script
S3_BUCKET="${S3_BUCKET:-your-bucket-name}"
MODELS_DIR="/home/ec2-user/chatbot/models"

echo "📤 Uploading models to S3..."

# Upload models
aws s3 sync $MODELS_DIR s3://$S3_BUCKET/models/ \
    --exclude "*.pyc" \
    --exclude "__pycache__/*" \
    --delete

# Upload timestamp
echo "$(date)" > /tmp/last_trained.txt
aws s3 cp /tmp/last_trained.txt s3://$S3_BUCKET/

echo "✅ Models uploaded to S3: s3://$S3_BUCKET/models/"
EOF

chmod +x deploy_to_s3.sh

# Create training and deployment script
cat > train_and_deploy.sh << 'EOF'
#!/bin/bash

# Configuration
LOG_DIR="/home/ec2-user/chatbot/training/logs"
LOG_FILE="$LOG_DIR/training.log"
MODELS_DIR="/home/ec2-user/chatbot/models"
S3_BUCKET="${S3_BUCKET:-your-bucket-name}"
CHATBOT_SERVER="${CHATBOT_SERVER:-http://YOUR_CHATBOT_IP:5001}"

# Create logs directory
mkdir -p $LOG_DIR

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

log "=========================================="
log "Starting training and deployment..."
log "=========================================="

cd /home/ec2-user/chatbot

# Pull latest code
log "Pulling latest code from GitHub..."
git pull origin main >> $LOG_FILE 2>&1

if [ $? -ne 0 ]; then
    log "❌ Failed to pull code!"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
log "Installing dependencies..."
cd training
pip install -q -r requirements.txt >> $LOG_FILE 2>&1

# Train models
log "Starting model training..."
log "This may take 30-60 minutes..."
python train_model.py >> $LOG_FILE 2>&1

if [ $? -eq 0 ]; then
    log "✅ Training completed successfully!"
    
    # Upload models to S3
    log "Uploading models to S3 bucket: $S3_BUCKET"
    ./deploy_to_s3.sh >> $LOG_FILE 2>&1
    
    if [ $? -eq 0 ]; then
        log "✅ Models uploaded to S3!"
        
        # Notify chatbot server to reload
        log "Notifying chatbot server at: $CHATBOT_SERVER"
        RESPONSE=$(curl -s -X POST $CHATBOT_SERVER/reload-models -w "%{http_code}" -o /tmp/reload_response.txt)
        
        if [ "$RESPONSE" = "200" ]; then
            log "✅ Chatbot server notified successfully!"
        else
            log "⚠️ Failed to notify chatbot server (HTTP $RESPONSE)"
            log "Chatbot will use new models on next restart"
        fi
        
        log "=========================================="
        log "✅ Training and deployment completed!"
        log "=========================================="
    else
        log "❌ Failed to upload models to S3!"
        exit 1
    fi
else
    log "❌ Training failed! Check logs for details."
    exit 1
fi
EOF

chmod +x train_and_deploy.sh

# Create webhook listener
cat > webhook_training.py << 'EOF'
from flask import Flask, request, jsonify
import subprocess
import hmac
import hashlib
import os
from datetime import datetime
import psutil

app = Flask(__name__)

# GitHub webhook secret
SECRET = os.environ.get('WEBHOOK_SECRET', 'your-secret-here')

def verify_signature(payload, signature):
    """Verify GitHub webhook signature"""
    if not signature:
        return False
    expected = 'sha256=' + hmac.new(
        SECRET.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)

def is_training_running():
    """Check if training is currently running"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'train_model.py' in cmdline:
                return True, proc.info['pid']
        except:
            pass
    return False, None

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle GitHub webhook"""
    # Verify GitHub signature
    signature = request.headers.get('X-Hub-Signature-256')
    if not verify_signature(request.data, signature):
        return jsonify({"error": "Invalid signature"}), 401
    
    # Get payload
    payload = request.json
    
    # Only process push events to main branch
    if payload.get('ref') != 'refs/heads/main':
        return jsonify({
            "status": "ignored",
            "reason": "Not main branch"
        }), 200
    
    # Check if training is already running
    running, pid = is_training_running()
    if running:
        return jsonify({
            "status": "skipped",
            "reason": f"Training already in progress (PID: {pid})",
            "message": "Wait for current training to complete"
        }), 200
    
    print(f"[{datetime.now()}] Received push to main branch")
    print(f"[{datetime.now()}] Starting training...")
    
    # Trigger training in background
    subprocess.Popen([
        '/bin/bash',
        '/home/ec2-user/chatbot/training/train_and_deploy.sh'
    ])
    
    return jsonify({
        "status": "training_started",
        "message": "Models will be trained and uploaded to S3",
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "training-webhook",
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/status', methods=['GET'])
def status():
    """Get training status"""
    running, pid = is_training_running()
    
    # Read last few lines of log
    log_file = '/home/ec2-user/chatbot/training/logs/training.log'
    last_logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                last_logs = f.readlines()[-10:]
        except:
            pass
    
    return jsonify({
        "training_running": running,
        "pid": pid if running else None,
        "last_logs": [log.strip() for log in last_logs]
    }), 200

@app.route('/logs', methods=['GET'])
def logs():
    """Get training logs"""
    lines = int(request.args.get('lines', 50))
    log_file = '/home/ec2-user/chatbot/training/logs/training.log'
    
    if not os.path.exists(log_file):
        return jsonify({"error": "Log file not found"}), 404
    
    try:
        with open(log_file, 'r') as f:
            all_logs = f.readlines()
            recent_logs = all_logs[-lines:]
        
        return jsonify({
            "logs": [log.strip() for log in recent_logs],
            "total_lines": len(all_logs)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"[{datetime.now()}] Starting Training Webhook Listener on port 5002...")
    app.run(host='0.0.0.0', port=5002, debug=False)
EOF

# Create systemd service for webhook
sudo tee /etc/systemd/system/training-webhook.service > /dev/null << 'EOF'
[Unit]
Description=Training Webhook Listener
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/chatbot/training
Environment="PATH=/home/ec2-user/chatbot/venv/bin:/usr/bin"
Environment="WEBHOOK_SECRET=your-secret-here"
Environment="S3_BUCKET=your-bucket-name"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/home/ec2-user/chatbot/venv/bin/python webhook_training.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Create management script
cat > manage.sh << 'EOF'
#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

case "$1" in
    start)
        echo "🚀 Starting training webhook..."
        sudo systemctl start training-webhook
        sleep 1
        sudo systemctl status training-webhook --no-pager
        ;;
    stop)
        echo "🛑 Stopping training webhook..."
        sudo systemctl stop training-webhook
        ;;
    restart)
        echo "🔄 Restarting training webhook..."
        sudo systemctl restart training-webhook
        sleep 1
        sudo systemctl status training-webhook --no-pager
        ;;
    status)
        echo "📊 Training Webhook Status:"
        sudo systemctl status training-webhook --no-pager
        echo ""
        curl -s http://localhost:5002/status | python3 -m json.tool
        ;;
    logs)
        if [ "$2" = "live" ]; then
            tail -f logs/training.log
        else
            tail -50 logs/training.log
        fi
        ;;
    webhook-logs)
        sudo journalctl -u training-webhook -f
        ;;
    train)
        echo "🎓 Starting manual training..."
        ./train_and_deploy.sh
        ;;
    health)
        curl http://localhost:5002/health | python3 -m json.tool
        ;;
    *)
        echo "Training Server Management"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|logs|webhook-logs|train|health}"
        echo ""
        echo "Commands:"
        echo "  start          - Start webhook listener"
        echo "  stop           - Stop webhook listener"
        echo "  restart        - Restart webhook listener"
        echo "  status         - Check webhook status and training progress"
        echo "  logs           - View training logs (add 'live' for tail -f)"
        echo "  webhook-logs   - View webhook listener logs"
        echo "  train          - Run training manually"
        echo "  health         - Check webhook health"
        exit 1
        ;;
esac
EOF

chmod +x manage.sh

# Create .env template
cat > .env.example << 'EOF'
# AWS Configuration
S3_BUCKET=your-bucket-name

# Webhook Configuration
WEBHOOK_SECRET=your-webhook-secret

# Chatbot Server
CHATBOT_SERVER=http://YOUR_CHATBOT_IP:5001
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable training-webhook
sudo systemctl start training-webhook

echo ""
echo "✅ Training server setup complete!"
echo ""
echo "📝 Configuration needed:"
echo "1. Update environment variables:"
echo "   sudo nano /etc/systemd/system/training-webhook.service"
echo "   - Change S3_BUCKET=your-bucket-name"
echo "   - Change WEBHOOK_SECRET=your-secret"
echo ""
echo "2. Update chatbot server IP:"
echo "   nano train_and_deploy.sh"
echo "   - Change CHATBOT_SERVER=http://YOUR_CHATBOT_IP:5001"
echo ""
echo "3. Restart webhook:"
echo "   ./manage.sh restart"
echo ""
echo "4. Add training documents:"
echo "   cd ../shared/files/"
echo "   # Upload your documents here"
echo ""
echo "5. Run first training:"
echo "   ./manage.sh train"
echo ""
echo "📊 Management commands:"
echo "  ./manage.sh start          - Start webhook"
echo "  ./manage.sh status         - Check status"
echo "  ./manage.sh logs           - View training logs"
echo "  ./manage.sh logs live      - Live training logs"
echo "  ./manage.sh train          - Run training manually"
echo ""
echo "🌐 Webhook URL:"
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
echo "  http://$PUBLIC_IP:5002/webhook"
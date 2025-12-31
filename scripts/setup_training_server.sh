#!/bin/bash

# Training Server Setup Script
# Run this on Training EC2

echo "🎓 Setting up Training Server..."

# Install git if not present
sudo yum update -y
sudo yum install -y git

# Clone repository
cd /home/ec2-user
git clone https://github.com/YOUR_USERNAME/your-chatbot-repo.git chatbot
cd chatbot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install training dependencies
cd training
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Create models directory
mkdir -p ../models

# Configure AWS CLI
echo "⚙️ Configure AWS CLI for S3 access:"
aws configure

# Create training webhook listener
cat > webhook_training.py << 'EOF'
from flask import Flask, request, jsonify
import subprocess
import hmac
import hashlib
import os
from datetime import datetime

app = Flask(__name__)

# GitHub webhook secret
SECRET = os.environ.get('WEBHOOK_SECRET', 'your-secret-here')

def verify_signature(payload, signature):
    if not signature:
        return False
    expected = 'sha256=' + hmac.new(
        SECRET.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)

@app.route('/webhook', methods=['POST'])
def webhook():
    # Verify GitHub signature
    signature = request.headers.get('X-Hub-Signature-256')
    if not verify_signature(request.data, signature):
        return jsonify({"error": "Invalid signature"}), 401
    
    # Get payload
    payload = request.json
    
    # Only process push events to main branch
    if payload.get('ref') == 'refs/heads/main':
        print(f"[{datetime.now()}] Received push to main branch")
        
        # Trigger training in background
        subprocess.Popen([
            '/bin/bash',
            '/home/ec2-user/chatbot/training/train_and_deploy.sh'
        ])
        
        return jsonify({
            "status": "Training started",
            "message": "Models will be trained and uploaded to S3"
        }), 200
    
    return jsonify({"status": "Ignored", "reason": "Not main branch"}), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "training-webhook"
    }), 200

@app.route('/status', methods=['GET'])
def status():
    # Check if training is running
    import psutil
    
    training_running = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'train_model.py' in cmdline:
                training_running = True
                break
        except:
            pass
    
    # Check last training log
    log_file = '/home/ec2-user/chatbot/training/logs/training.log'
    last_line = ""
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
    
    return jsonify({
        "training_running": training_running,
        "last_log": last_line
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
EOF

# Create train and deploy script
cat > train_and_deploy.sh << 'EOF'
#!/bin/bash

# Training and Deployment Script
LOG_DIR="/home/ec2-user/chatbot/training/logs"
LOG_FILE="$LOG_DIR/training.log"
MODELS_DIR="/home/ec2-user/chatbot/models"
S3_BUCKET="your-bucket-name"

# Create logs directory
mkdir -p $LOG_DIR

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

log "=========================================="
log "Starting training process..."
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
pip install -r requirements.txt >> $LOG_FILE 2>&1

# Train models
log "Starting model training..."
python train_model.py >> $LOG_FILE 2>&1

if [ $? -eq 0 ]; then
    log "✅ Training completed successfully!"
    
    # Upload models to S3
    log "Uploading models to S3..."
    aws s3 sync $MODELS_DIR s3://$S3_BUCKET/models/ \
        --exclude "*.pyc" \
        --exclude "__pycache__/*" \
        --delete >> $LOG_FILE 2>&1
    
    if [ $? -eq 0 ]; then
        log "✅ Models uploaded to S3!"
        
        # Create timestamp file
        echo "$(date)" > /tmp/last_trained.txt
        aws s3 cp /tmp/last_trained.txt s3://$S3_BUCKET/
        
        # Notify chatbot server to reload (optional)
        CHATBOT_SERVER="http://YOUR_CHATBOT_IP:5001"
        log "Notifying chatbot server to reload models..."
        curl -X POST $CHATBOT_SERVER/reload-models >> $LOG_FILE 2>&1
        
        log "=========================================="
        log "✅ Training and deployment completed!"
        log "=========================================="
    else
        log "❌ Failed to upload models to S3!"
        exit 1
    fi
else
    log "❌ Training failed!"
    exit 1
fi
EOF

chmod +x train_and_deploy.sh

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
ExecStart=/home/ec2-user/chatbot/venv/bin/python webhook_training.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create management script
cat > manage.sh << 'EOF'
#!/bin/bash

case "$1" in
    start)
        echo "Starting training webhook..."
        sudo systemctl start training-webhook
        sudo systemctl status training-webhook
        ;;
    stop)
        echo "Stopping training webhook..."
        sudo systemctl stop training-webhook
        ;;
    restart)
        echo "Restarting training webhook..."
        sudo systemctl restart training-webhook
        ;;
    status)
        sudo systemctl status training-webhook
        ;;
    logs)
        tail -f logs/training.log
        ;;
    webhook-logs)
        sudo journalctl -u training-webhook -f
        ;;
    train)
        echo "Starting manual training..."
        ./train_and_deploy.sh
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|webhook-logs|train}"
        exit 1
        ;;
esac
EOF

chmod +x manage.sh

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable training-webhook
sudo systemctl start training-webhook

echo ""
echo "✅ Training server setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Update S3_BUCKET in train_and_deploy.sh"
echo "2. Update WEBHOOK_SECRET in /etc/systemd/system/training-webhook.service"
echo "3. Update CHATBOT_SERVER IP in train_and_deploy.sh"
echo "4. Restart webhook: ./manage.sh restart"
echo "5. Add documents to ../shared/files/"
echo "6. Run first training: ./manage.sh train"
echo ""
echo "📊 Management commands:"
echo "  ./manage.sh start          - Start webhook listener"
echo "  ./manage.sh stop           - Stop webhook listener"
echo "  ./manage.sh status         - Check webhook status"
echo "  ./manage.sh logs           - View training logs"
echo "  ./manage.sh webhook-logs   - View webhook logs"
echo "  ./manage.sh train          - Run training manually"
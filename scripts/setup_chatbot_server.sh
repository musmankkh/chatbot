#!/bin/bash

# Chatbot Server Setup Script
# Run this on Chatbot EC2

echo "🤖 Setting up Chatbot Server..."

# Install git if not present
sudo yum update -y
sudo yum install -y git

# Clone repository
cd /home/ec2-user
git clone https://github.com/musmankkh/chatbot.git chatbot
cd chatbot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install chatbot dependencies (lighter than training)
cd chatbot
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Configure AWS CLI
echo "⚙️ Configure AWS CLI for S3 access:"
aws configure

# Download initial models from S3
echo "📥 Downloading initial models..."
mkdir -p ../models
aws s3 sync s3://my-chatbot-models-bucket/models/ ../models/

# Create model downloader script
cat > download_models.sh << 'EOF'
#!/bin/bash

S3_BUCKET="my-chatbot-models-bucket"
MODELS_DIR="/home/ec2-user/chatbot/models"

echo "📥 Downloading models from S3..."

aws s3 sync s3://$S3_BUCKET/models/ $MODELS_DIR/ --delete

if [ $? -eq 0 ]; then
    echo "✅ Models downloaded successfully!"
    
    # Check last trained time
    aws s3 cp s3://$S3_BUCKET/last_trained.txt /tmp/ 2>/dev/null
    if [ -f /tmp/last_trained.txt ]; then
        echo "Last trained: $(cat /tmp/last_trained.txt)"
    fi
else
    echo "❌ Failed to download models!"
    exit 1
fi
EOF

chmod +x download_models.sh

# Create chatbot webhook listener
cat > webhook_chatbot.py << 'EOF'
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
        
        # Trigger reload in background
        subprocess.Popen([
            '/bin/bash',
            '/home/ec2-user/chatbot/chatbot/reload.sh'
        ])
        
        return jsonify({
            "status": "Reload started",
            "message": "Chatbot will be updated and restarted"
        }), 200
    
    return jsonify({"status": "Ignored", "reason": "Not main branch"}), 200

@app.route('/reload-models', methods=['POST'])
def reload_models():
    """Called by training server when new models are ready"""
    print(f"[{datetime.now()}] Received model reload request")
    
    subprocess.Popen([
        '/bin/bash',
        '/home/ec2-user/chatbot/chatbot/reload.sh'
    ])
    
    return jsonify({
        "status": "Reloading models",
        "message": "Chatbot will reload with new models"
    }), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "chatbot-webhook"
    }), 200

@app.route('/status', methods=['GET'])
def status():
    # Check if chatbot is running
    import psutil
    
    chatbot_running = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'main.py' in cmdline and 'chatbot' in cmdline:
                chatbot_running = True
                break
        except:
            pass
    
    return jsonify({
        "chatbot_running": chatbot_running
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
EOF

# Create reload script
cat > reload.sh << 'EOF'
#!/bin/bash

LOG_FILE="/home/ec2-user/chatbot/chatbot/logs/reload.log"
mkdir -p /home/ec2-user/chatbot/chatbot/logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

log "=========================================="
log "Starting reload process..."
log "=========================================="

cd /home/ec2-user/chatbot

# Pull latest code
log "Pulling latest code..."
git pull origin main >> $LOG_FILE 2>&1

# Activate virtual environment
source venv/bin/activate

# Update dependencies
log "Updating dependencies..."
cd chatbot
pip install -r requirements.txt >> $LOG_FILE 2>&1

# Download latest models
log "Downloading latest models from S3..."
./download_models.sh >> $LOG_FILE 2>&1

if [ $? -ne 0 ]; then
    log "❌ Failed to download models!"
    exit 1
fi

# Restart chatbot
log "Restarting chatbot..."

# Find and kill old process
CHATBOT_PID=$(cat /tmp/chatbot.pid 2>/dev/null)
if [ ! -z "$CHATBOT_PID" ]; then
    log "Stopping old chatbot (PID: $CHATBOT_PID)"
    kill $CHATBOT_PID 2>/dev/null
    sleep 2
fi

# Start new process
log "Starting new chatbot..."
cd /home/ec2-user/chatbot/chatbot
nohup python main.py > logs/app.log 2>&1 &
NEW_PID=$!
echo $NEW_PID > /tmp/chatbot.pid

log "✅ Chatbot restarted (PID: $NEW_PID)"
log "=========================================="
EOF

chmod +x reload.sh

# Create systemd service for chatbot
sudo tee /etc/systemd/system/chatbot.service > /dev/null << 'EOF'
[Unit]
Description=Chatbot API
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/chatbot/chatbot
Environment="PATH=/home/ec2-user/chatbot/venv/bin:/usr/bin"
ExecStart=/home/ec2-user/chatbot/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create systemd service for webhook
sudo tee /etc/systemd/system/chatbot-webhook.service > /dev/null << 'EOF'
[Unit]
Description=Chatbot Webhook Listener
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/chatbot/chatbot
Environment="PATH=/home/ec2-user/chatbot/venv/bin:/usr/bin"
Environment="WEBHOOK_SECRET=your-secret-here"
ExecStart=/home/ec2-user/chatbot/venv/bin/python webhook_chatbot.py
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
        echo "Starting chatbot services..."
        sudo systemctl start chatbot
        sudo systemctl start chatbot-webhook
        sleep 2
        $0 status
        ;;
    stop)
        echo "Stopping chatbot services..."
        sudo systemctl stop chatbot
        sudo systemctl stop chatbot-webhook
        ;;
    restart)
        echo "Restarting chatbot services..."
        sudo systemctl restart chatbot
        sudo systemctl restart chatbot-webhook
        ;;
    status)
        echo "=== Chatbot Status ==="
        sudo systemctl status chatbot --no-pager
        echo ""
        echo "=== Webhook Status ==="
        sudo systemctl status chatbot-webhook --no-pager
        ;;
    logs)
        case "$2" in
            chatbot)
                tail -f logs/app.log
                ;;
            webhook)
                sudo journalctl -u chatbot-webhook -f
                ;;
            reload)
                tail -f logs/reload.log
                ;;
            *)
                echo "Usage: $0 logs {chatbot|webhook|reload}"
                ;;
        esac
        ;;
    download)
        echo "Downloading latest models..."
        ./download_models.sh
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|download}"
        echo ""
        echo "  start    - Start all services"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  status   - Show service status"
        echo "  logs     - View logs (chatbot|webhook|reload)"
        echo "  download - Download latest models from S3"
        exit 1
        ;;
esac
EOF

chmod +x manage.sh

# Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable chatbot
sudo systemctl enable chatbot-webhook
sudo systemctl start chatbot
sudo systemctl start chatbot-webhook

echo ""
echo "✅ Chatbot server setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Update S3_BUCKET in download_models.sh"
echo "2. Update WEBHOOK_SECRET in /etc/systemd/system/chatbot-webhook.service"
echo "3. Restart services: ./manage.sh restart"
echo ""
echo "📊 Management commands:"
echo "  ./manage.sh start           - Start all services"
echo "  ./manage.sh stop            - Stop all services"
echo "  ./manage.sh status          - Check service status"
echo "  ./manage.sh logs chatbot    - View chatbot logs"
echo "  ./manage.sh logs webhook    - View webhook logs"
echo "  ./manage.sh download        - Download latest models"
echo ""
echo "🌐 Access URLs:"
echo "  Chatbot API:  http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):5000"
echo "  Webhook:      http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):5001/webhook"
#!/bin/bash

# Chatbot Server Setup Script for musmankkh/chatbot
# Run this on Chatbot EC2

echo "🤖 Setting up Chatbot Server..."

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

if [ ! -f "requirements.txt" ]; then
    cat > requirements.txt << 'EOF'
# Chatbot Runtime Dependencies
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
psutil
watchdog
EOF
fi

# Install dependencies
pip install -r requirements.txt

# Create requirements.txt for chatbot if it doesn't exist
cd chatbot

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Create models directory
mkdir -p ../models
mkdir -p logs

# Configure AWS CLI
echo ""
echo "⚙️ Configure AWS CLI for S3 access:"
aws configure

# Create model downloader script
cat > download_models.sh << 'EOF'
#!/bin/bash

S3_BUCKET="${S3_BUCKET:-your-bucket-name}"
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
    
    # List downloaded models
    echo "Models downloaded:"
    ls -lh $MODELS_DIR/
else
    echo "❌ Failed to download models!"
    exit 1
fi
EOF

chmod +x download_models.sh

# Create webhook listener for chatbot
cat > webhook_chatbot.py << 'EOF'
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

def is_chatbot_running():
    """Check if chatbot is running"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'main.py' in cmdline and 'chatbot' in cmdline:
                return True, proc.info['pid']
        except:
            pass
    return False, None

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle GitHub webhook for code updates"""
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
    
    print(f"[{datetime.now()}] Received push to main branch")
    print(f"[{datetime.now()}] Starting reload...")
    
    # Trigger reload in background
    subprocess.Popen([
        '/bin/bash',
        '/home/ec2-user/chatbot/chatbot/reload.sh'
    ])
    
    return jsonify({
        "status": "reload_started",
        "message": "Chatbot will be updated and restarted",
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/reload-models', methods=['POST'])
def reload_models():
    """Reload models after training (called by training server)"""
    print(f"[{datetime.now()}] Received model reload request from training server")
    
    # Trigger reload in background
    subprocess.Popen([
        '/bin/bash',
        '/home/ec2-user/chatbot/chatbot/reload.sh'
    ])
    
    return jsonify({
        "status": "reloading",
        "message": "Downloading new models and restarting chatbot",
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    running, pid = is_chatbot_running()
    return jsonify({
        "status": "healthy",
        "service": "chatbot-webhook",
        "chatbot_running": running,
        "chatbot_pid": pid,
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/status', methods=['GET'])
def status():
    """Get chatbot status"""
    running, pid = is_chatbot_running()
    
    # Read last few lines of log
    log_file = '/home/ec2-user/chatbot/chatbot/logs/app.log'
    last_logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                last_logs = f.readlines()[-10:]
        except:
            pass
    
    return jsonify({
        "chatbot_running": running,
        "pid": pid if running else None,
        "last_logs": [log.strip() for log in last_logs]
    }), 200

@app.route('/logs', methods=['GET'])
def logs():
    """Get chatbot logs"""
    lines = int(request.args.get('lines', 50))
    log_type = request.args.get('type', 'app')  # app, reload, webhook
    
    log_files = {
        'app': '/home/ec2-user/chatbot/chatbot/logs/app.log',
        'reload': '/home/ec2-user/chatbot/chatbot/logs/reload.log'
    }
    
    log_file = log_files.get(log_type)
    if not log_file or not os.path.exists(log_file):
        return jsonify({"error": "Log file not found"}), 404
    
    try:
        with open(log_file, 'r') as f:
            all_logs = f.readlines()
            recent_logs = all_logs[-lines:]
        
        return jsonify({
            "logs": [log.strip() for log in recent_logs],
            "total_lines": len(all_logs),
            "log_type": log_type
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"[{datetime.now()}] Starting Chatbot Webhook Listener on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=False)
EOF

# Create reload script
cat > reload.sh << 'EOF'
#!/bin/bash

# Configuration
LOG_FILE="/home/ec2-user/chatbot/chatbot/logs/reload.log"
S3_BUCKET="${S3_BUCKET:-your-bucket-name}"

# Create logs directory
mkdir -p /home/ec2-user/chatbot/chatbot/logs

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

log "=========================================="
log "Starting reload process..."
log "=========================================="

cd /home/ec2-user/chatbot

# Pull latest code
log "Pulling latest code from GitHub..."
git pull origin main >> $LOG_FILE 2>&1

if [ $? -ne 0 ]; then
    log "⚠️ Failed to pull code, continuing with existing code..."
fi

# Activate virtual environment
source venv/bin/activate

# Update dependencies
log "Updating dependencies..."
cd chatbot
pip install -q -r requirements.txt >> $LOG_FILE 2>&1

# Download latest models from S3
log "Downloading latest models from S3: $S3_BUCKET"
./download_models.sh >> $LOG_FILE 2>&1

if [ $? -ne 0 ]; then
    log "❌ Failed to download models!"
    log "Chatbot will continue with existing models"
fi

# Find and stop old chatbot process
log "Stopping old chatbot process..."
CHATBOT_PID=$(pgrep -f "python.*main.py.*chatbot")

if [ ! -z "$CHATBOT_PID" ]; then
    log "Found chatbot process (PID: $CHATBOT_PID)"
    kill $CHATBOT_PID 2>/dev/null
    sleep 2
    
    # Force kill if still running
    if ps -p $CHATBOT_PID > /dev/null 2>&1; then
        log "Force killing chatbot process..."
        kill -9 $CHATBOT_PID 2>/dev/null
        sleep 1
    fi
    
    log "✅ Old chatbot stopped"
else
    log "No running chatbot process found"
fi

# Start new chatbot process
log "Starting new chatbot..."
cd /home/ec2-user/chatbot/chatbot

nohup python main.py > logs/app.log 2>&1 &
NEW_PID=$!

# Wait a bit and check if it started successfully
sleep 3

if ps -p $NEW_PID > /dev/null 2>&1; then
    echo $NEW_PID > /tmp/chatbot.pid
    log "✅ Chatbot restarted successfully (PID: $NEW_PID)"
    
    # Test health endpoint
    sleep 2
    HEALTH_CHECK=$(curl -s http://localhost:5000/health 2>/dev/null)
    if [ $? -eq 0 ]; then
        log "✅ Health check passed"
    else
        log "⚠️ Health check failed, but process is running"
    fi
else
    log "❌ Failed to start chatbot! Check app.log for errors"
    exit 1
fi

log "=========================================="
log "✅ Reload completed successfully!"
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
Environment="S3_BUCKET=your-bucket-name"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/home/ec2-user/chatbot/venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=append:/home/ec2-user/chatbot/chatbot/logs/app.log
StandardError=append:/home/ec2-user/chatbot/chatbot/logs/app.log

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
Environment="S3_BUCKET=your-bucket-name"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/home/ec2-user/chatbot/venv/bin/python webhook_chatbot.py
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
        echo "🚀 Starting chatbot services..."
        sudo systemctl start chatbot
        sudo systemctl start chatbot-webhook
        sleep 2
        $0 status
        ;;
    stop)
        echo "🛑 Stopping chatbot services..."
        sudo systemctl stop chatbot
        sudo systemctl stop chatbot-webhook
        ;;
    restart)
        echo "🔄 Restarting chatbot services..."
        sudo systemctl restart chatbot
        sudo systemctl restart chatbot-webhook
        sleep 2
        $0 status
        ;;
    status)
        echo "📊 Chatbot Status:"
        sudo systemctl status chatbot --no-pager -l
        echo ""
        echo "📊 Webhook Status:"
        sudo systemctl status chatbot-webhook --no-pager -l
        echo ""
        curl -s http://localhost:5001/status | python3 -m json.tool
        ;;
    logs)
        case "$2" in
            chatbot|app)
                if [ "$3" = "live" ]; then
                    tail -f logs/app.log
                else
                    tail -50 logs/app.log
                fi
                ;;
            reload)
                if [ "$3" = "live" ]; then
                    tail -f logs/reload.log
                else
                    tail -50 logs/reload.log
                fi
                ;;
            webhook)
                sudo journalctl -u chatbot-webhook -f
                ;;
            *)
                echo "Usage: $0 logs {chatbot|reload|webhook} [live]"
                ;;
        esac
        ;;
    download)
        echo "📥 Downloading latest models from S3..."
        ./download_models.sh
        ;;
    reload)
        echo "🔄 Reloading chatbot with new models..."
        ./reload.sh
        ;;
    health)
        echo "🏥 Chatbot Health:"
        curl -s http://localhost:5000/health | python3 -m json.tool
        echo ""
        echo "🏥 Webhook Health:"
        curl -s http://localhost:5001/health | python3 -m json.tool
        ;;
    test)
        echo "🧪 Testing chatbot..."
        curl -X POST http://localhost:5000/chat \
            -H "Content-Type: application/json" \
            -d '{"query":"test"}' | python3 -m json.tool
        ;;
    *)
        echo "Chatbot Server Management"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|logs|download|reload|health|test}"
        echo ""
        echo "Commands:"
        echo "  start          - Start all services"
        echo "  stop           - Stop all services"
        echo "  restart        - Restart all services"
        echo "  status         - Check service status"
        echo "  logs           - View logs (chatbot|reload|webhook) [live]"
        echo "  download       - Download latest models from S3"
        echo "  reload         - Reload chatbot with new models"
        echo "  health         - Check health endpoints"
        echo "  test           - Send test request to chatbot"
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

# Flask Configuration
FLASK_ENV=production
EOF

# Download initial models
echo ""
echo "📥 Attempting to download initial models from S3..."
./download_models.sh 2>/dev/null || echo "⚠️ No models in S3 yet. Run training first."

# Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable chatbot
sudo systemctl enable chatbot-webhook
sudo systemctl start chatbot
sudo systemctl start chatbot-webhook

sleep 2

echo ""
echo "✅ Chatbot server setup complete!"
echo ""
echo "📝 Configuration needed:"
echo "1. Update environment variables in systemd services:"
echo "   sudo nano /etc/systemd/system/chatbot.service"
echo "   sudo nano /etc/systemd/system/chatbot-webhook.service"
echo "   - Change S3_BUCKET=your-bucket-name"
echo "   - Change WEBHOOK_SECRET=your-secret"
echo ""
echo "2. Update .env file in chatbot folder:"
echo "   nano .env"
echo "   # Add your OpenAI key and other configs"
echo ""
echo "3. Restart services:"
echo "   ./manage.sh restart"
echo ""
echo "📊 Management commands:"
echo "  ./manage.sh start              - Start all services"
echo "  ./manage.sh status             - Check status"
echo "  ./manage.sh logs chatbot       - View chatbot logs"
echo "  ./manage.sh logs chatbot live  - Live chatbot logs"
echo "  ./manage.sh download           - Download models"
echo "  ./manage.sh health             - Check health"
echo "  ./manage.sh test               - Test chatbot"
echo ""
echo "🌐 Access URLs:"
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
echo "  Chatbot:  http://$PUBLIC_IP:5000"
echo "  Webhook:  http://$PUBLIC_IP:5001/webhook"
echo "  Health:   http://$PUBLIC_IP:5000/health"
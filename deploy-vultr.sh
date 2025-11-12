#!/bin/bash
# ABC Application Deployment Script for Vultr VPS
# This script sets up the ABC Application on a fresh Ubuntu VPS

set -e  # Exit on any error

echo "🚀 Starting ABC Application deployment on Vultr VPS..."

# Update system
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install required system packages
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    postgresql \
    postgresql-contrib \
    redis-server \
    build-essential \
    curl \
    wget \
    git \
    ufw \
    fail2ban

# Configure firewall
echo "🔒 Configuring firewall..."
sudo ufw allow ssh
sudo ufw allow 8000  # Application port
sudo ufw --force enable

# Configure PostgreSQL
echo "🗄️ Setting up PostgreSQL..."
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql -c "CREATE DATABASE abc_application;" || echo "Database already exists"
sudo -u postgres psql -c "CREATE USER abc_user WITH PASSWORD 'secure_password_here';" || echo "User already exists"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE abc_application TO abc_user;"

# Configure Redis
echo "🔄 Configuring Redis..."
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test Redis connection
redis-cli ping

# Clone or copy application code
echo "📁 Setting up application directory..."
# Assuming code is already uploaded to the server
cd /opt
sudo mkdir -p abc-application
sudo chown $USER: abc-application
cd abc-application

# Create Python virtual environment
echo "🐍 Setting up Python environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📚 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Copy configuration files
echo "⚙️ Setting up configuration..."
# Make sure .env file is uploaded with your API keys
# cp .env.example .env  # Edit .env with your actual keys

# Create logs directory
mkdir -p logs

# Set up systemd service
echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/abc-application.service > /dev/null <<EOF
[Unit]
Description=ABC Application Multi-Agent Trading System
After=network.target postgresql.service redis-server.service

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/abc-application
Environment=PATH=/opt/abc-application/venv/bin
ExecStart=/opt/abc-application/venv/bin/python src/main.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable abc-application
sudo systemctl start abc-application

# Set up log rotation
echo "📝 Configuring log rotation..."
sudo tee /etc/logrotate.d/abc-application > /dev/null <<EOF
/opt/abc-application/logs/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
    postrotate
        systemctl reload abc-application
    endscript
}
EOF

# Set up backup script
echo "💾 Setting up backup system..."
sudo mkdir -p /backups/abc_application
sudo tee /usr/local/bin/abc-backup.sh > /dev/null <<'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/abc_application"

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

# PostgreSQL backup
pg_dump -U abc_user -h localhost abc_application > $BACKUP_DIR/postgres_$DATE.sql

# Redis backup
redis-cli --rdb $BACKUP_DIR/redis_$DATE.rdb

# Configuration backup
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /opt/abc-application/config/

# Application logs backup
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz /opt/abc-application/logs/

# Retention policy (keep last 30 days)
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete
find $BACKUP_DIR -name "*.rdb" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $DATE"
EOF

sudo chmod +x /usr/local/bin/abc-backup.sh

# Set up daily backup cron job
echo "⏰ Setting up automated backups..."
sudo tee /etc/cron.d/abc-application-backup > /dev/null <<EOF
# Daily backup at 2 AM
0 2 * * * $USER /usr/local/bin/abc-backup.sh
EOF

# Health check script
echo "🏥 Setting up health monitoring..."
sudo tee /usr/local/bin/abc-health-check.sh > /dev/null <<'EOF'
#!/bin/bash
# Health check script for ABC Application

HEALTH_STATUS=0

# Check if service is running
if ! systemctl is-active --quiet abc-application; then
    echo "❌ ABC Application service is not running"
    HEALTH_STATUS=1
fi

# Check database connection
if ! pg_isready -h localhost -U abc_user -d abc_application >/dev/null 2>&1; then
    echo "❌ PostgreSQL connection failed"
    HEALTH_STATUS=1
fi

# Check Redis connection
if ! redis-cli ping >/dev/null 2>&1; then
    echo "❌ Redis connection failed"
    HEALTH_STATUS=1
fi

# Check application health endpoint
if ! curl -f http://localhost:8000/health >/dev/null 2>&1; then
    echo "❌ Application health check failed"
    HEALTH_STATUS=1
fi

if [ $HEALTH_STATUS -eq 0 ]; then
    echo "✅ All systems healthy"
else
    echo "❌ Health check failed - check system status"
    exit 1
fi
EOF

sudo chmod +x /usr/local/bin/abc-health-check.sh

# Set up health check cron job (every 5 minutes)
echo "0,5,10,15,20,25,30,35,40,45,50,55 * * * * $USER /usr/local/bin/abc-health-check.sh" | sudo tee /etc/cron.d/abc-application-health > /dev/null

echo "🎉 ABC Application deployment completed!"
echo ""
echo "📋 Next steps:"
echo "1. Upload your .env file with API keys to /opt/abc-application/"
echo "2. Test the application: curl http://localhost:8000/health"
echo "3. Check logs: journalctl -u abc-application -f"
echo "4. Monitor backups: ls -la /backups/abc_application/"
echo ""
echo "🔧 Useful commands:"
echo "- Start service: sudo systemctl start abc-application"
echo "- Stop service: sudo systemctl stop abc-application"
echo "- Restart service: sudo systemctl restart abc-application"
echo "- View logs: journalctl -u abc-application -f"
echo "- Manual backup: /usr/local/bin/abc-backup.sh"
echo "- Health check: /usr/local/bin/abc-health-check.sh"
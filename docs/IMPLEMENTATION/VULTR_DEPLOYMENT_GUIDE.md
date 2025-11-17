# 🚀 ABC Application Vultr Deployment Guide

## Prerequisites
- Vultr account (free to create)
- IBKR paper trading account
- SSH keys (already configured)

## Step 1: Create Vultr VPS Instance

### 1.1 Create Vultr Account
1. Go to [vultr.com](https://vultr.com)
2. Sign up for a free account
3. Add payment method (credit card required, but you get $100 free credit)

### 1.2 Deploy Ubuntu VPS
1. Click "Deploy" → "Cloud Compute"
2. Choose server location (closest to you)
3. Select Ubuntu 22.04 LTS x64
4. Choose plan: **$12/month (2GB RAM, 1 vCPU)** - sufficient for ABC Application
5. Add SSH key:
   - Click "Add SSH Key"
   - Name: `ABC-Application-Key`
   - Paste your public key: `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMSTF3wQ2nbLZAKpICBUcMShFetH12FoKeg/il7rG3yV nvick@NoahsDesktop`
6. Enable backups (optional but recommended)
7. Click "Deploy Now"

### 1.3 Get VPS Details
After deployment (5-10 minutes), note:
- **IP Address**: `YOUR_VPS_IP` (e.g., 123.456.789.0)
- **Username**: `root`
- **SSH Key**: Already configured

## Step 2: Initial VPS Setup

### 2.1 Connect to VPS
```bash
ssh -i ABCSSH root@YOUR_VPS_IP
```

### 2.2 Update System
```bash
sudo apt update && sudo apt upgrade -y
```

### 2.3 Create Application User
```bash
# Create user
sudo adduser abcuser --gecos "" --disabled-password

# Add to sudo group
sudo usermod -aG sudo abcuser

# Setup SSH for abcuser
sudo mkdir -p /home/abcuser/.ssh
sudo cp ~/.ssh/authorized_keys /home/abcuser/.ssh/
sudo chown -R abcuser:abcuser /home/abcuser/.ssh
sudo chmod 700 /home/abcuser/.ssh
sudo chmod 600 /home/abcuser/.ssh/authorized_keys

# Test login as abcuser
su - abcuser
exit
```

## Step 3: Upload Application Code

### 3.1 From Your Local Machine
```bash
# Create deployment archive (exclude sensitive files)
tar -czf abc-deploy.tar.gz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='data/*.json' \
    --exclude='logs/*' \
    .

# Upload to VPS
scp -i ABCSSH abc-deploy.tar.gz root@YOUR_VPS_IP:~/
```

### 3.2 On VPS - Extract Code
```bash
# SSH to VPS as root
ssh -i ABCSSH root@YOUR_VPS_IP

# Extract application
sudo mkdir -p /opt/abc-application
sudo tar -xzf abc-deploy.tar.gz -C /opt/abc-application/
sudo chown -R abcuser:abcuser /opt/abc-application

# Clean up
rm abc-deploy.tar.gz
```

## Step 4: Configure Environment

### 4.1 Create .env File
```bash
# Switch to abcuser
su - abcuser

# Create .env file
cd /opt/abc-application
cp .env.template .env

# Edit .env with your credentials
nano .env
```

**Required .env variables:**
```bash
# Database
DATABASE_URL=postgresql://abc_user:secure_password_here@localhost/abc_application

# Redis
REDIS_URL=redis://localhost:6379

# IBKR Paper Trading
IBKR_USERNAME=your_ibkr_username
IBKR_PASSWORD=your_ibkr_password
IBKR_ACCOUNT_ID=your_paper_account_id
IBKR_HOST=localhost
IBKR_PORT=7497

# Other API keys (add as needed)
OPENAI_API_KEY=your_openai_key
# ... other keys
```

## Step 5: Run Deployment Script

### 5.1 Execute Deployment
```bash
# As abcuser
cd /opt/abc-application
chmod +x deploy-vultr.sh
./deploy-vultr.sh
```

### 5.2 Monitor Deployment
```bash
# Check service status
sudo systemctl status abc-application
sudo systemctl status ibkr-tws

# View logs
journalctl -u abc-application -f
journalctl -u ibkr-tws -f
```

## Step 6: Configure IBKR TWS

### 6.1 Start IBKR TWS
```bash
sudo systemctl start ibkr-tws
```

### 6.2 Connect to TWS GUI (Optional)
If you need to configure TWS manually:
```bash
# Install VNC or similar for GUI access
# Or use the automated setup script
/usr/local/bin/setup-ibkr-paper.sh
```

### 6.3 Verify IBKR Connection
```bash
# Run the test script
python test_ibkr_paper_trading.py
```

## Step 7: Final Verification

### 7.1 Health Checks
```bash
# Application health
curl http://localhost:8000/health

# Manual health check
/usr/local/bin/abc-health-check.sh

# Service status
sudo systemctl status abc-application ibkr-tws postgresql redis-server
```

### 7.2 Test Trading Operations
```bash
# Run comprehensive tests
python test_ibkr_paper_trading.py
```

## Troubleshooting

### Common Issues

**SSH Connection Failed:**
```bash
# Check SSH key permissions
chmod 600 ABCSSH
chmod 644 ABCSSH.pub
```

**IBKR TWS Won't Start:**
```bash
# Check Xvfb
ps aux | grep xvfb

# Restart TWS service
sudo systemctl restart ibkr-tws
```

**Application Won't Start:**
```bash
# Check logs
journalctl -u abc-application -f

# Check Python environment
cd /opt/abc-application
source venv/bin/activate
python -c "import src.main"
```

**Database Connection Issues:**
```bash
# Check PostgreSQL
sudo systemctl status postgresql
sudo -u postgres psql -c "SELECT version();"
```

## Security Notes

- ✅ SSH key authentication only (password disabled)
- ✅ UFW firewall configured
- ✅ Fail2ban installed for SSH protection
- ✅ Services run as non-root user
- ⚠️  Remember to change default PostgreSQL password
- ⚠️  Regularly update system packages

## Monitoring & Maintenance

```bash
# View application logs
journalctl -u abc-application -f

# View IBKR logs
journalctl -u ibkr-tws -f

# Manual backup
/usr/local/bin/abc-backup.sh

# Health check
/usr/local/bin/abc-health-check.sh
```

## Cost Estimate
- **Vultr VPS**: $12/month (2GB RAM)
- **IBKR Paper Trading**: Free
- **Total**: ~$12/month for full paper trading setup
# [LABEL:DOC:24_6_setup] [LABEL:FRAMEWORK:discord] [LABEL:INFRA:vps]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-11-19] [LABEL:REVIEWED:pending]
#
# Purpose: Setup guide for 24/6 continuous operation with Discord output
# Dependencies: Discord bot, systemd, VPS deployment
# Related: src/agents/unified_workflow_orchestrator.py, setup/deploy-vultr.sh
#
# 24/6 Continuous Operation Setup Guide
# =====================================

## Overview
The 24/6 Continuous Workflow Orchestrator provides automated, market-aware trading analysis workflows that run continuously with real-time Discord output for monitoring and intervention.

## Features
- **Continuous Operation**: Runs 24/6 with market-aware scheduling
- **Discord Integration**: Real-time output and human intervention capabilities
- **Automated Workflows**: Scheduled analysis based on market hours and conditions
- **Market Monitoring**: Triggers emergency analysis for significant market events
- **System Health**: Continuous monitoring with automated alerts

## Prerequisites
1. **VPS Deployment**: Follow the main deployment guide (`setup/deploy-vultr.sh`)
2. **Discord Bot**: Create a Discord application at https://discord.com/developers/applications
3. **Environment Variables**: Configure required secrets in vault/.env

## Discord Bot Setup

### 1. Create Discord Application
1. Go to https://discord.com/developers/applications
2. Click "New Application"
3. Name it "ABC Trading Orchestrator"
4. Go to "Bot" section and create a bot
5. Copy the bot token

### 2. Bot Permissions
The bot needs these permissions (integer value: 414464658496):
- Send Messages
- Use Slash Commands
- Read Message History
- Mention Everyone
- Use External Emojis
- Embed Links

### 3. Invite Bot to Server
Generate invite URL: `https://discord.com/api/oauth2/authorize?client_id=YOUR_CLIENT_ID&permissions=414464658496&scope=bot`

## Environment Configuration

### Required Environment Variables
Add these to your vault or .env file:

```bash
# Discord Configuration
DISCORD_ORCHESTRATOR_TOKEN=your_bot_token_here
DISCORD_GUILD_ID=your_server_id_here

# Existing IBKR and API keys
IBKR_USERNAME=your_ibkr_username
IBKR_PASSWORD=your_ibkr_password
# ... other existing keys
```

### Finding Discord Guild ID
1. Enable Developer Mode in Discord (User Settings → App Settings → Advanced → Developer Mode)
2. Right-click your server name and select "Copy ID"

## Scheduled Workflows

The orchestrator runs these automated workflows:

### Trading Days (Mon-Fri, Eastern Time - ET)
- **5:30 AM ET**: Early Monday Prep - Extra early Monday market regime assessment
- **6:00 AM ET**: Pre-Market Prep - Early pre-market analysis and data collection
- **7:30 AM ET**: Market Open Prep - Final pre-open analysis and position setup (**2+ hours before 9:30 AM open**)
- **12:00 PM ET**: Midday Check - Intraday performance and adjustment analysis
- **4:30 PM ET**: Market Close Review - End-of-day performance review
- **5:00 PM ET**: Post-Market Review - Post-market analysis and next-day preparation

**Note**: The schedule provides **2+ hours of preparation time** before the 9:30 AM ET market open, allowing comprehensive analysis and position setup.

## Preparation Time Strategy

The 24/6 orchestrator implements a **staggered preparation approach** to ensure thorough analysis before market open:

### Phase 1: Early Preparation (5:30-6:00 AM ET)
- **Data Collection**: Pull economic indicators, news sentiment, and preliminary market data
- **Regime Assessment**: Determine current market regime (bull/bear/sideways)
- **Risk Parameters**: Set volatility expectations and risk limits

### Phase 2: Pre-Market Analysis (6:00-7:30 AM ET)
- **Technical Analysis**: Complete chart analysis and technical indicator calculations
- **Fundamental Review**: Review earnings, economic data, and sector performance
- **Strategy Development**: Generate initial trade hypotheses and position sizing

### Phase 3: Final Preparation (7:30 AM ET)
- **Position Setup**: Final trade selection and position sizing calculations
- **Execution Planning**: Prepare order types, entry/exit criteria, and risk management
- **Contingency Planning**: Define adjustment triggers and stop-loss levels

This multi-phase approach ensures the system is fully prepared with comprehensive analysis by market open.

### Emergency Triggers
- **VIX > 30**: High volatility analysis
- **Market Move > ±2%**: Significant market movement response
- **System Health Issues**: Automated health checks and alerts

## Manual Discord Commands

Even in 24/6 mode, you can manually control the system:

```
!start_workflow     - Trigger immediate full analysis workflow
!pause_workflow     - Pause current automated workflow
!resume_workflow    - Resume paused workflow
!stop_workflow      - Stop current workflow
!workflow_status    - Check current workflow status
!status            - System health and agent status
!analyze <query>   - Request specific analysis from agents
```

## Deployment Steps

### 1. Run Deployment Script
```bash
# On your VPS
sudo ./setup/deploy-vultr.sh
```

### 2. Configure Environment
```bash
# Add Discord credentials to vault
sudo -u abc-user /opt/abc-application/venv/bin/python -c "
from src.utils.vault_client import store_vault_secret
store_vault_secret('DISCORD_ORCHESTRATOR_TOKEN', 'your_token')
store_vault_secret('DISCORD_GUILD_ID', 'your_guild_id')
"
```

### 3. Start Services
```bash
sudo systemctl start abc-24-6-orchestrator
sudo systemctl enable abc-24-6-orchestrator
```

### 4. Verify Operation
```bash
# Check service status
sudo systemctl status abc-24-6-orchestrator

# View logs
journalctl -u abc-24-6-orchestrator -f

# Check Discord for startup message
```

## Monitoring and Maintenance

### Log Files
- **Application Logs**: `logs/24_6_orchestrator.log`
- **System Logs**: `journalctl -u abc-24-6-orchestrator`
- **Workflow Results**: `data/live_workflow_results.json`

### Health Checks
```bash
# Manual health check
sudo /usr/local/bin/abc-health-check.sh

# Check Discord bot responsiveness
# Send !status in Discord
```

### Troubleshooting

#### Bot Not Responding
1. Check token: `sudo -u abc-user /opt/abc-application/venv/bin/python -c "from src.utils.vault_client import get_vault_secret; print('Token exists:', bool(get_vault_secret('DISCORD_ORCHESTRATOR_TOKEN')))"`
2. Check guild ID: `sudo -u abc-user /opt/abc-application/venv/bin/python -c "from src.utils.vault_client import get_vault_secret; print('Guild ID:', get_vault_secret('DISCORD_GUILD_ID'))"`
3. Check permissions: Ensure bot has proper permissions in Discord server
4. Restart service: `sudo systemctl restart abc-24-6-orchestrator`

#### Workflows Not Starting
1. Check market calendar: Ensure it's a trading day
2. Check system time: `date` (should be correct timezone)
3. Check agent health: Send `!status` in Discord
4. Check logs for errors

#### High Resource Usage
1. Monitor with: `htop` or `top`
2. Check concurrent workflows
3. Adjust schedule if needed
4. Consider VPS upgrade for higher resource requirements

## Integration with Main Application

The 24/6 orchestrator works alongside the main ABC application:

- **Main App** (`abc-application.service`): Core trading logic and execution
- **24/6 Orchestrator** (`abc-24-6-orchestrator.service`): Workflow coordination and Discord output
- **IBKR TWS** (`ibkr-tws.service`): Trading platform connectivity

## Timezone Handling

All scheduled workflows use **Eastern Time (ET)** - the timezone of the NYSE:

- **Automatic Detection**: The system uses `exchange_calendars` to determine NYSE trading hours and holidays
- **Trading Day Validation**: Workflows only run on valid trading days (excluding weekends and holidays)
- **Market Hours Awareness**: Intraday workflows check if the market is currently open
- **VPS Timezone**: Ensure your VPS system clock is set to the correct timezone, or configure NTP properly

### Timezone Configuration
```bash
# Check current timezone
timedatectl

# Set to Eastern Time if needed
sudo timedatectl set-timezone America/New_York

# Or set to UTC and let the application handle conversion
sudo timedatectl set-timezone UTC
```

The orchestrator automatically handles timezone conversions using the NYSE calendar, ensuring workflows run at the correct times regardless of your server's local timezone.

## Backup and Recovery

The deployment script includes automated backups. For the 24/6 orchestrator:

- **Configuration**: Backed up in `/backups/abc_application/config_*.tar.gz`
- **Logs**: Backed up in `/backups/abc_application/logs_*.tar.gz`
- **Workflow Data**: Stored in `data/` directory

## Next Steps

1. **Test Basic Operation**: Start with manual workflows
2. **Verify Scheduling**: Wait for automated workflows to trigger
3. **Monitor Performance**: Check resource usage and response times
4. **Customize Schedules**: Adjust timing based on your preferences
5. **Add Emergency Triggers**: Implement market-specific alert conditions

The 24/6 orchestrator provides continuous market awareness while maintaining human oversight capabilities through Discord integration.</content>
<parameter name="filePath">c:\Users\nvick\ABC-Application\docs\IMPLEMENTATION\24_6_CONTINUOUS_OPERATION.md
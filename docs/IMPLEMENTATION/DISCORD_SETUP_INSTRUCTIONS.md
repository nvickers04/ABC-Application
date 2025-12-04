# Discord Integration Setup Instructions

## 1. Create Discord Server

### Step-by-Step Server Creation:
1. **Open Discord** and log into your account
2. **Click the "+" button** on the left sidebar (Add a Server)
3. **Select "Create My Own"**
4. **Choose "For me and my friends"** (or "For a club or community")
5. **Name your server** (e.g., "ABC Trading Agents")
6. **Upload an icon** (optional)
7. **Click "Create"**

### Get Your Server ID (Guild ID):
**Important: You need to enable Developer Mode first!**

**Method 1: Standard Navigation**
1. Click the **gear icon** ⚙️ (User Settings) at the bottom left
2. Look for **"Advanced"** in the left sidebar
3. Toggle **"Developer Mode"** to **ON**

**Method 2: If "Advanced" isn't visible**
1. Click the **gear icon** ⚙️ at bottom left
2. Scroll down and look for **"Advanced"** section
3. Or search for "Developer" in the settings search bar (top of settings)

**Method 3: Alternative path**
1. Click the **gear icon** ⚙️ at bottom left
2. Look for **"App Settings"** → **"Behavior"** (older Discord versions)
3. Toggle **"Developer Mode"** to **ON**

**Method 4: Quick access**
- Press `Ctrl + Shift + I` (opens developer tools, but that's not what you want)
- Instead, just search for "Developer Mode" in Discord's settings search

**Once Developer Mode is enabled:**
1. **Right-click your server name** (top-left of the server list)
2. **Select "Copy Server ID"**
3. **Paste it into your `.env` file:**
   ```
   DISCORD_GUILD_ID=your_actual_server_id_here
   ```

**If you still can't find it:**
- Update Discord to the latest version
- Try the web version at discord.com/app
- The setting might be called "Developer Mode" or "Enable Developer Mode"

## 2. Create Bot Application

The system uses a single Discord bot for orchestration and command handling:

1. **Go to:** https://discord.com/developers/applications
2. **Click "New Application"**
   - Name: "ABC Orchestrator" or similar

3. **Bot Configuration:**
   - Go to the **"Bot"** section
   - Click **"Reset Token"** to generate a new bot token
   - **Copy the token** and add to your `.env` file:
     ```
     DISCORD_ORCHESTRATOR_TOKEN=your_bot_token_here
     ```

## 3. Configure Bot Permissions

In the Discord developer portal:

- Go to **OAuth2 → URL Generator**
- Select scopes: `bot`, `applications.commands`
- Select permissions:
  - ✅ Send Messages
  - ✅ Use Slash Commands
  - ✅ Read Message History
  - ✅ Add Reactions
  - ✅ Mention Everyone (for alerts)
  - ✅ Create Polls (for consensus voting)
  - ✅ Embed Links
  - ✅ Attach Files

## 4. Invite Bot to Server

### Step-by-Step Bot Invitation:

1. **Go back to Discord Developer Portal:**
   - Visit: https://discord.com/developers/applications
   - Select your "ABC Orchestrator" bot application

2. **Generate Invite Link:**
   - Click **"OAuth2"** in the left sidebar
   - Click **"URL Generator"** sub-tab
   - Under **"Scopes"**, check:
     - ✅ `bot`
     - ✅ `applications.commands`
   - Under **"Bot Permissions"**, check the permissions listed above

3. **Copy the Generated URL:**
   - The URL will appear at the bottom
   - It should look like: `https://discord.com/api/oauth2/authorize?client_id=...`

4. **Invite the Bot:**
   - **Open the URL in your browser**
   - **Select your server** from the dropdown
   - **Click "Authorize"**
   - **Complete the CAPTCHA** if prompted

### Verification Steps:

After inviting the bot:
1. **Check your Discord server** - you should see the "ABC Orchestrator" bot online
2. **The bot should appear in the member list** on the right sidebar

### Troubleshooting Bot Invites:

**Bot not appearing in server?**
- Double-check you selected the correct server when authorizing
- Make sure you're the server owner/admin
- Try refreshing Discord (Ctrl + R)

**"Missing Permissions" error?**
- Go back to OAuth2 → URL Generator
- Ensure all required permissions are checked
- Regenerate the URL and try again

**Bot shows as offline?**
- This is normal - bots only show as online when your Python script is running
- The script will bring them online when you run `python src/agents/discord_agents.py`

**Can't find the bot in Developer Portal?**
- Make sure you're logged into the correct Discord account
- Check that you created the applications with the right names

## 5. Create System Channels

### Step-by-Step Channel Creation:

1. **In your Discord server**, right-click in the channel list
2. **Select "Create Channel"**
3. **Choose "Text"** for all channels
4. **Create these channels** (copy the names exactly):

**System Channels:**
- `#general` - Main discussion and workflow orchestration
- `#alerts` - System alerts and notifications
- `#ranked-trades` - Ranked trade proposals and signals
- `#commands` - Command reference and documentation

### Channel Setup Tips:

- **Keep channels organized** - put system channels in a "🤖 System" category
- **Set permissions** if needed (bot should work with default permissions)
- **Channel names are case-sensitive** in some contexts, but Discord allows any case

### Optional: Get Channel IDs

If you want to configure specific channels for the bot:
1. **Enable Developer Mode** (from earlier steps)
2. **Right-click each channel** → **"Copy Channel ID"**
3. **Add to `.env`** (optional - system works without this):
   ```
   DISCORD_HEALTH_CHANNEL_ID=your_health_monitoring_channel_id_here
   DISCORD_ALERTS_CHANNEL_ID=your_alerts_channel_id_here
   DISCORD_RANKED_TRADES_CHANNEL_ID=your_ranked_trades_channel_id_here
   DISCORD_COMMANDS_CHANNEL_ID=your_commands_channel_id_here
   ```

## 6. Configure Channel IDs (Optional)

To enable the bot to use specific channels:

1. **Right-click each system channel** → **Copy Channel ID**
2. **Update your `.env` file** (optional - bot will use default channels without this):
   ```
   # Example channel IDs (replace with your actual IDs)
   # Health monitoring channel for system status, API health, memory usage
   DISCORD_HEALTH_CHANNEL_ID=1234567890123456789
   DISCORD_ALERTS_CHANNEL_ID=1234567890123456789
   DISCORD_RANKED_TRADES_CHANNEL_ID=1234567890123456789
   DISCORD_COMMANDS_CHANNEL_ID=1234567890123456789
   ```

## 7. Test Your Setup

### Quick Verification Steps:

1. **Check bot is in your server:**
   - Look at the member list (right sidebar)
   - You should see the "ABC Orchestrator" bot online

2. **Test basic connectivity:**
   - Run the orchestrator script (check your run configuration or use the VS Code task)
   - Bot should show as "online" in Discord
   - Check terminal for connection messages

3. **Test commands in Discord:**
   - Go to `#health-monitoring` or any channel
   - Type: `!status`
   - Should get a system status response

4. **Test command reference:**
   - Type: `/commands` or `!commands`
   - Should show available commands
   - Use: `!set_commands_channel 1445767031482744902` (replace with your commands channel ID)
   - This will configure the #commands channel for detailed documentation

### If Something Goes Wrong:

- **Bot not responding?** Check terminal for error messages
- **Permission errors?** Re-invite bot with correct permissions
- **Connection issues?** Verify DISCORD_ORCHESTRATOR_TOKEN and DISCORD_GUILD_ID in `.env`
- **Commands not working?** Make sure bot has "Use Slash Commands" permission

---

**Ready to launch?** Once the bot is invited and channels created, start the LiveWorkflowOrchestrator to begin using the system!

## Troubleshooting

- **Bot not responding?** Check that token is correct and bot is online
- **Missing permissions?** Re-invite bot with proper permissions
- **Channel issues?** Ensure bot can see and send messages in channels
- **Polls not working?** Verify "Create Polls" permission is granted
- **@everyone not working?** Check "Mention Everyone" permission

---

**Need help with any step?** Check the logs for detailed error messages!

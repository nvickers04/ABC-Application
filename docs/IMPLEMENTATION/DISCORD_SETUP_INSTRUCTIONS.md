# Discord Agent Integration Setup Instructions

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

## 2. Create Bot Applications

For each main agent, create a separate Discord bot application. The system currently supports 5 core agents with Discord integration:

1. **Go to:** https://discord.com/developers/applications
2. **Click "New Application"** for each agent:
   - **Macro Agent** (Required)
   - **Data Agent** (Required)  
   - **Strategy Agent** (Required)
   - **Reflection Agent** (Required)
   - **Execution Agent** (Required)
   - **Risk Agent** (Optional - currently disabled due to TensorFlow dependencies)
   - **Learning Agent** (Optional - currently disabled due to TensorFlow dependencies)

3. **For each application:**
   - Go to the **"Bot"** section
   - Click **"Reset Token"** to generate a new bot token
   - **Copy the token** (this goes in your `.env` file)

## 3. Configure Bot Permissions

In the Discord developer portal for each bot:

- Go to **OAuth2 → URL Generator**
- Select scopes: `bot`, `applications.commands`
- Select permissions:
  - ✅ Send Messages
  - ✅ Use Slash Commands  
  - ✅ Read Message History
  - ✅ Add Reactions
  - ✅ Mention Everyone (for alerts)
  - ✅ Create Polls (for voting)

## 4. Invite Bots to Server

### Step-by-Step Bot Invitation:

**For each of your 5 core agents (Macro, Data, Strategy, Reflection, Execution):**

1. **Go back to Discord Developer Portal:**
   - Visit: https://discord.com/developers/applications
   - Select the bot application you created (e.g., "Macro Agent")

2. **Generate Invite Link:**
   - Click **"OAuth2"** in the left sidebar
   - Click **"URL Generator"** sub-tab
   - Under **"Scopes"**, check:
     - ✅ `bot`
     - ✅ `applications.commands`
   - Under **"Bot Permissions"**, check:
     - ✅ Send Messages
     - ✅ Use Slash Commands
     - ✅ Read Message History
     - ✅ Add Reactions
     - ✅ Mention Everyone (for alerts)
     - ✅ Create Polls (for voting)
     - ✅ Read Messages/View Channels

3. **Copy the Generated URL:**
   - The URL will appear at the bottom
   - It should look like: `https://discord.com/api/oauth2/authorize?client_id=...`

4. **Invite Each Bot:**
   - **Open the URL in your browser**
   - **Select your server** from the dropdown (should show your "ABC Trading Agents" server)
   - **Click "Authorize"**
   - **Complete the CAPTCHA** if prompted
   - **Repeat for each of the 5 core bots**

### Optional Agents:
- **Risk Agent** and **Learning Agent** bots can be added later when TensorFlow dependencies are resolved
- These agents will integrate seamlessly once their Discord tokens are configured

### Verification Steps:

After inviting all bots:
1. **Check your Discord server** - you should see 5 new bot users online (Macro, Data, Strategy, Reflection, Execution)
2. **Each bot should have a name** like "Macro Agent#1234"
3. **Bots should appear in the member list** on the right sidebar

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

## 5. Create Agent Channels

### Step-by-Step Channel Creation:

1. **In your Discord server**, right-click in the channel list
2. **Select "Create Channel"**
3. **Choose "Text"** for all channels
4. **Create these channels** (copy the names exactly):

**Agent-Specific Channels:**
- `#macro-agent` - Macro economic analysis and market context
- `#data-agent` - Real-time data collection and market intelligence  
- `#strategy-agent` - Trading strategies and signal generation
- `#reflection-agent` - System oversight and decision validation
- `#execution-agent` - Trade execution and position management
- `#risk-agent` - Risk assessment and position limits (optional)
- `#learning-agent` - System learning and performance optimization (optional)

**Collaboration Channels:**
- `#debates` - Agent debates and human-agent discussions
- `#alerts` - Critical system alerts and notifications
- `#general` - General discussion (if not already created)

### Channel Setup Tips:

- **Keep channels organized** - put agent channels in a "🤖 Agents" category
- **Set permissions** if needed (bots should work with default permissions)
- **Channel names are case-sensitive** in some contexts, but Discord allows any case

### Optional: Get Channel IDs

If you want status updates in specific channels:
1. **Enable Developer Mode** (from earlier steps)
2. **Right-click each channel** → **"Copy Channel ID"**
3. **Add to `.env`** (optional - system works without this):
   ```
   DISCORD_MACRO_CHANNEL_ID=your_channel_id_here
   DISCORD_DATA_CHANNEL_ID=your_channel_id_here
   # etc.
   ```

## 6. Configure Channel IDs (Optional)

To enable status updates in specific channels:

1. **Right-click each agent channel** → **Copy Channel ID**
2. **Update your `.env` file** (optional - bots will work without this):
   ```
   # Example channel IDs (replace with your actual IDs)
   DISCORD_MACRO_CHANNEL_ID=1234567890123456789
   DISCORD_DATA_CHANNEL_ID=1234567890123456789
   # ... etc for each agent
   ```

## 8. Test Your Setup

### Quick Verification Steps:

1. **Check bots are in your server:**
   - Look at the member list (right sidebar)
   - You should see 5 bot users online (Macro, Data, Strategy, Reflection, Execution)

2. **Test basic connectivity:**
   - Run: `python src/agents/discord_agents.py`
   - Bots should show as "online" in Discord
   - Check terminal for connection messages

3. **Test commands in Discord:**
   - Go to `#general` or any channel
   - Type: `!status`
   - Should get a response from agents

4. **Test enhanced features:**
   - `!create_poll "Test?" "Yes" "No"` - Creates Discord native poll
   - `!agent_vote "Decision?" "Buy" "Sell"` - Creates reaction-based vote
   - `!debate "Should we test trading?"` - Starts agent debate

### If Something Goes Wrong:

- **Bots not responding?** Check terminal for error messages
- **Permission errors?** Re-invite bots with correct permissions
- **Connection issues?** Verify tokens and guild ID in `.env`
- **Commands not working?** Make sure bots have "Use Slash Commands" permission

---

**Ready to launch?** Once all bots are invited and channels created, run `python src/agents/discord_agents.py` to start your AI trading agents!

## Troubleshooting

- **Bot not responding?** Check that tokens are correct and bots are online
- **Missing permissions?** Re-invite bots with proper permissions
- **Channel issues?** Ensure bots can see and send messages in channels
- **Polls not working?** Verify "Create Polls" permission is granted
- **@everyone not working?** Check "Mention Everyone" permission

---

**Need help with any step?** The setup script will validate your configuration once everything is set up!
